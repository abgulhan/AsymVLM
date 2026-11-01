"""Idle-residue repair, retaining saved fits and the original four-hour hard stop."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from . import attention_pilot_v19 as pilot
from . import attention_resume_v19a as previous
from . import attention_operations_v19 as operations
from .compression_budget_operations_v17a import probe_gpu
from .compression_matrix_v8 import checked, seal
from .evidence_catalog_v18b import cached_catalog
from .followup_common import write_immutable
from .state import file_digest
from .worker import verify_sources

ROOT = previous.ROOT.with_name('attention_v19b')
_load_records = pilot.load_records
_wait_stage = pilot.wait_stage
_source_pins = pilot.source_pins
RESIDUE_MIB = 4


def observation_idle(value):
    """Four MiB alone is not a workload; processes or utilization always veto."""
    return (value.get('apps') == ''
            and type(value.get('memory_mib')) is int
            and 0 <= value['memory_mib'] <= RESIDUE_MIB
            and type(value.get('utilization_percent')) is int
            and value['utilization_percent'] == 0)


def gpu_idle(gpu, *, allow_cooldown):
    if type(gpu) is not int or gpu not in range(8):
        raise ValueError('explicit ClusterB GPU index required')
    deadline = time.monotonic() + (15 if allow_cooldown else 6)
    clean = 0
    while time.monotonic() < deadline:
        try:
            value = probe_gpu(gpu, timeout=min(3, max(.1, (deadline-time.monotonic())/2)))
        except (OSError, ValueError, subprocess.SubprocessError):
            return False
        if value.get('apps'):
            return False
        clean = clean + 1 if observation_idle(value) else 0
        if clean >= 2:
            return True
        if not clean and not allow_cooldown:
            return False
        time.sleep(min(.5, max(0, deadline-time.monotonic())))
    return False


def source_pins():
    script = Path(__file__).resolve().parents[2] / 'scripts/mlsys_attention_v19b_idle_clusterb.sh'
    return {**_source_pins(), str(Path(__file__).resolve()): file_digest(__file__),
            str(script): file_digest(script)}


def validate_protocol(path):
    value = checked(path)
    root = Path(value['cwd']).resolve() / ROOT
    parent_path = Path(value['cwd']).resolve() / previous.ROOT / 'protocol.json'
    parent = checked(parent_path)
    hard_stop = parent['deadline_utc_epoch']
    if (Path(path).resolve() != root/'protocol.json' or Path(value['output_root']).resolve() != root
            or value.get('experiment') != 'attention_v19' or value.get('maximum_gpu_workers') != 8
            or value.get('resume_parent') != str(parent_path)
            or value.get('resume_parent_sha256') != parent['sha256']
            or value['created_utc_epoch'] != parent['created_utc_epoch']
            or value['deadline_utc_epoch'] != hard_stop
            or value['physical_deadline_utc_epoch'] != hard_stop-300
            or value['stop_new_work_utc_epoch'] != hard_stop-600):
        raise ValueError('recovery identity or original hard stop changed')
    return value


def load_records(protocol):
    extraction = checked(protocol['extraction_protocol'])
    if (extraction['sha256'] != protocol['extraction_protocol_sha256']
            or any(extraction[key] != protocol[key] for key in
                   ('manifest', 'model', 'training', 'diagnostic_ids',
                    'created_utc_epoch', 'deadline_utc_epoch'))):
        raise ValueError('extraction inputs, training recipe, or hard stop changed')
    return _load_records(extraction)


def saved_fits(protocol, records):
    parent = checked(protocol['resume_parent'])
    if len(records) != protocol['reused_label_rows']:
        raise ValueError('saved fit panel changed')
    result = {}
    for arm in pilot.ARMS:
        path = Path(parent['output_root'])/'training'/arm/'latest.pt'
        if file_digest(path) != protocol['input_hashes'][str(path)]:
            raise ValueError('saved scorer changed')
        _, payload = pilot.learning.load_scorer(path)
        if (payload['identity']['protocol_sha256'] != parent['sha256']
                or payload['identity']['arm'] != arm or payload['epoch'] != 3):
            raise ValueError('saved scorer recipe changed')
        result[arm] = str(path)
    return result


def diagnostic_plan(protocol, records):
    return Path(protocol['resume_parent']).parent/'plans/02-sink-diagnostic/task.json'


def wait_stage(protocol, name):
    if name != '02-sink-diagnostic':
        return _wait_stage(protocol, name)
    result = pilot.reporting.collect_task(diagnostic_plan(protocol, None))
    if not result['status']['complete']:
        raise ValueError('saved diagnostics are incomplete')
    return result


def prepare(project):
    project = Path(project).resolve()
    path = project/ROOT/'protocol.json'
    if path.exists():
        return validate_protocol(path)
    parent_path = project/previous.ROOT/'protocol.json'
    parent = checked(parent_path)
    verify_sources(parent)
    if time.time() >= parent['deadline_utc_epoch']-600:
        raise TimeoutError('no time remains inside the original hard stop')
    pins = {**parent['input_hashes'], **source_pins(), str(parent_path): file_digest(parent_path)}
    extraction = checked(parent['extraction_protocol'])
    records = _load_records(extraction)
    for arm in pilot.ARMS:
        for name in ('latest.pt', 'complete.json', 'temperature.json'):
            p = parent_path.parent/'training'/arm/name
            pins[str(p)] = file_digest(p)
    diagnostic_task = parent_path.parent/'plans/02-sink-diagnostic/task.json'
    diagnostic = pilot.reporting.collect_task(diagnostic_task)
    if not diagnostic['status']['complete']:
        raise ValueError('cannot reuse incomplete physical diagnostics')
    pins[str(diagnostic_task)] = file_digest(diagnostic_task)
    for p in (parent_path.parent/'physical/02-sink-diagnostic').glob('rank*.complete.json'):
        pins[str(p)] = file_digest(p)
        pins.update(checked(p)['input_hashes'])
    hard_stop = parent['deadline_utc_epoch']
    value = seal({**{k: v for k, v in parent.items() if k != 'sha256'},
        'output_root': str(path.parent), 'input_hashes': pins,
        'resume_parent': str(parent_path), 'resume_parent_sha256': parent['sha256'],
        'stop_new_work_utc_epoch': hard_stop-600, 'physical_deadline_utc_epoch': hard_stop-300,
        'resume_reason': 'idle_4_MiB_residue; reopen_internal_cutoffs_with_original_hard_stop',
        'idle_gpu_rule': 'two_process_free_zero_utilization_probes_with_memory_at_most_4_MiB',
        'reused_label_rows': len(records), 'reused_diagnostic_measurements': len(diagnostic['rows'])})
    saved_fits(value, records)
    for rank in range(8):
        marker = checked(Path(extraction['output_root'])/'labels'/f'rank{rank}.complete.json')
        write_immutable(path.parent/'labels'/f'rank{rank}.complete.json', marker)
    write_immutable(path, value)
    return value


def configure():
    pilot.load_records = load_records
    pilot.fit = saved_fits
    pilot.diagnostic_plan = diagnostic_plan
    pilot.wait_stage = wait_stage
    pilot.source_pins = source_pins
    operations.ROOT = ROOT
    operations.configure()
    operations.base.validate_protocol = validate_protocol
    operations.base.gpu_idle = gpu_idle
    operations.base.JOB_NAME = 'mlsys-attention-v19b'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'pipeline', 'clusterb-worker'))
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args(argv)
    configure()
    root = args.project.resolve()/ROOT
    with cached_catalog():
        if args.phase == 'prepare':
            protocol = prepare(args.project)
            config = operations.base.build_config(root/'protocol.json')
            result = dict(status='prepared', config_sha256=config['sha256'],
                reused_label_rows=protocol['reused_label_rows'], reused_scorers=3,
                reused_diagnostic_measurements=protocol['reused_diagnostic_measurements'])
        elif args.phase == 'pipeline':
            operations.base.read_config(root/'operations/config.json')
            result = pilot.pipeline(checked(root/'protocol.json'))
        else:
            config = operations.base.read_config(root/'operations/config.json')
            result = operations.clusterb_worker(config, args.gpu, 12000)
    print(json.dumps({k: v for k, v in result.items() if k != 'screening_gate'}), flush=True)


if __name__ == '__main__':
    main()
