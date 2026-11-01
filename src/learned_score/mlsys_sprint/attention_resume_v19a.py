"""Resume the failed attention pilot without changing its inputs or deadline.

The original run, source files, terminal receipt, and extracted labels remain
unchanged. Only the duplicate diagnostic warmup argument is removed; the
shared evaluator retains its standard one warmup. No jobs are submitted here.
"""
import argparse
import json
from pathlib import Path
import time

from . import attention_pilot_v19 as pilot
from . import attention_operations_v19 as operations
from .compression_matrix_v8 import checked, seal
from .evidence_catalog_v18b import cached_catalog
from .followup_common import write_immutable
from .state import file_digest

PARENT_ROOT = pilot.ROOT
ROOT = PARENT_ROOT.with_name('attention_v19a')
_evaluate = pilot.evaluate
_load_records = pilot.load_records
_source_pins = pilot.source_pins


def source_pins():
    script = Path(__file__).resolve().parents[2] / 'scripts/mlsys_attention_v19a_idle_clusterb.sh'
    return {**_source_pins(), str(Path(__file__).resolve()): file_digest(__file__),
            str(script): file_digest(script)}


def evaluate(protocol, name, split, cells, **kwargs):
    extra = dict(kwargs.get('extra') or {})
    if name == '02-sink-diagnostic' and extra.get('warmup') == 0:
        extra.pop('warmup')
        kwargs['extra'] = extra
    return _evaluate(protocol, name, split, cells, **kwargs)


def load_records(protocol):
    parent = checked(protocol['extraction_protocol'])
    if (parent['sha256'] != protocol['extraction_protocol_sha256']
            or any(parent[key] != protocol[key] for key in
                   ('manifest', 'model', 'training', 'diagnostic_ids',
                    'created_utc_epoch', 'stop_new_work_utc_epoch',
                    'physical_deadline_utc_epoch', 'deadline_utc_epoch'))):
        raise ValueError('resumed extraction identity, recipe, or deadline changed')
    return _load_records(parent)


def prepare(project):
    project = Path(project).resolve()
    root = project / ROOT
    if (root / 'protocol.json').exists():
        return checked(root / 'protocol.json')
    parent_path = project / PARENT_ROOT / 'protocol.json'
    parent = checked(parent_path)
    terminal_path = parent_path.parent / 'pipeline-terminal.json'
    terminal = checked(terminal_path)
    if (terminal.get('protocol_sha256') != parent['sha256']
            or terminal.get('status') != 'failed'
            or terminal.get('error') != "dict() got multiple values for keyword argument 'warmup'"):
        raise ValueError('only the recorded duplicate-warmup failure may be resumed')
    if time.time() >= parent['stop_new_work_utc_epoch']:
        raise TimeoutError('original admission deadline has passed')
    for path, sha in parent['input_hashes'].items():
        if file_digest(path) != sha:
            raise ValueError('original input changed: ' + path)
    records = _load_records(parent)
    pins = {**parent['input_hashes'], **source_pins(),
            str(parent_path): file_digest(parent_path),
            str(terminal_path): file_digest(terminal_path)}
    markers = []
    for rank in range(8):
        path = parent_path.parent / 'labels' / f'rank{rank}.complete.json'
        marker = checked(path)
        markers.append((path.name, marker))
        pins.update(marker['input_hashes'])
        pins[str(path)] = file_digest(path)
    protocol = seal({**{k: v for k, v in parent.items() if k != 'sha256'},
        'output_root': str(root), 'input_hashes': pins,
        'extraction_protocol': str(parent_path),
        'extraction_protocol_sha256': parent['sha256'],
        'resume_reason': 'remove_duplicate_warmup_keep_shared_evaluator_default',
        'reused_label_rows': len(records)})
    # These are readiness references to the original extraction, not new labels.
    for name, marker in markers:
        write_immutable(root / 'labels' / name, marker)
    write_immutable(root / 'protocol.json', protocol)
    return protocol


def configure():
    pilot.evaluate = evaluate
    pilot.load_records = load_records
    pilot.source_pins = source_pins
    operations.ROOT = ROOT
    operations.configure()
    operations.base.JOB_NAME = 'mlsys-attention-v19a'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'pipeline', 'clusterb-worker'))
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args(argv)
    configure()
    project = args.project.resolve()
    protocol_path = project / ROOT / 'protocol.json'
    config_path = project / ROOT / 'operations/config.json'
    with cached_catalog():
        if args.phase == 'prepare':
            protocol = prepare(project)
            config = operations.base.build_config(protocol_path)
            result = dict(status='prepared', reused_label_rows=protocol['reused_label_rows'],
                          config_sha256=config['sha256'])
        elif args.phase == 'pipeline':
            operations.base.read_config(config_path)
            result = pilot.pipeline(checked(protocol_path))
        else:
            result = operations.clusterb_worker(operations.base.read_config(config_path), args.gpu, 12000)
    print(json.dumps({k: v for k, v in result.items() if k != 'screening_gate'}), flush=True)


if __name__ == '__main__':
    main()
