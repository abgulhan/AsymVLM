"""Complete frozen-method benchmarks without reopening development selection.

The short-answer panel is explicitly post-hoc. Unopened confirmation identities
are reserved by a separate provenance audit; no inference may silently consume
them. The text pilot uses development pages and never tunes the frozen rule.
"""
import argparse
import fcntl
import json
from pathlib import Path
import re
import time

from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest
from .worker import read_plan, verify_sources

ROOT = Path('reproduction_results/mlsys_compression_20261022_v14')
SOURCE = Path('reproduction_results/mlsys_compression_20261022_v13/study.json')
FORMAT = 'mlsys_compression_campaign_v14'
MODELS = ('qwen25', 'llava_next')


def verify(study, *, sources=True):
    from .compression_finalize_v13 import verify as verify_source
    if study.get('sha256') != digest({k: v for k, v in study.items() if k != 'sha256'}):
        raise ValueError('campaign integrity mismatch')
    cwd = Path(study['cwd']).resolve()
    if (study.get('format') != FORMAT or Path(study['output_root']).resolve() != cwd / ROOT
            or study.get('scorer_training') is not False or study.get('rule_retuning') is not False
            or study.get('confirmation_labels_opened') is not False
            or study.get('independent_confirmation') is not False
            or study.get('dataset_is_policy_input') is not False
            or not 0 < study['deadline_utc_epoch'] - study['created_utc_epoch'] <= 12 * 3600
            or type(study.get('world_size')) is not int or not 1 <= study['world_size'] <= 64):
        raise ValueError('campaign scope changed')
    source = checked(study['source_study'])
    verify_source(source, sources=sources)
    decision = checked(study['decision_path'])
    if (study['models'] != source['models'] or set(study['models']) != set(MODELS)
            or decision['study_sha256'] != source['sha256']
            or decision['selected_method'] != 'scorer_adaptive'
            or decision['variant_id'] != 'rank32_coverage__conservative'
            or decision['sha256'] != study['decision_sha256']
            or not Path(study['reservation_manifest']).is_file()):
        raise ValueError('frozen method or reserved data identity changed')
    if sources:
        for path, sha in study['input_hashes'].items():
            if file_digest(path) != sha:
                raise ValueError('pinned campaign input changed: ' + path)
    return source


def python(study, module):
    return [str(Path(study['cwd']) / '.conda/asymvlm/bin/python'), '-B', '-u', '-m',
            'learned_score.mlsys_sprint.' + module]


def make_task(study, name, argv, artifacts, *, model, resource='gpu', minimum_start_seconds=300):
    from .adaptive_workflow_v4 import environment
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or model not in MODELS:
        raise ValueError('invalid campaign task identity')
    return {'id': 'compression-v14-' + name, 'tag': 'compression-v14-' + resource,
            'model_tag': model, 'resource': resource, 'minimum_start_seconds': minimum_start_seconds,
            'depends_on': [], 'argv': argv,
            'env': {**environment(Path(study['cwd'])), 'PYTHONDONTWRITEBYTECODE': '1'},
            'artifacts': [{'path': str(p)} for p in artifacts]}


def publish_plan(study, name, tasks, pins=None):
    verify(study)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('unsafe plan name')
    root = Path(study['output_root'])
    path = root / 'plans' / name / 'plan.json'
    all_pins = {**study['input_hashes'], str(root / 'study.json'): file_digest(root / 'study.json')}
    for key, sha in (pins or {}).items():
        if key in all_pins and all_pins[key] != sha:
            raise ValueError('cannot override a frozen input')
        all_pins[key] = sha
    if path.exists():
        plan = read_plan(path)
        verify_sources(plan)
        if plan['tasks'] != tasks or plan['input_hashes'] != all_pins:
            raise ValueError('cannot replace published commands or provenance')
        return path
    if time.time() >= study['deadline_utc_epoch']:
        raise ValueError('campaign publication deadline reached')
    value = seal({'format': FORMAT + '_plan', 'cwd': study['cwd'], 'tasks': tasks,
                  'input_hashes': all_pins, 'unexpected_retries': 1,
                  'created_utc_epoch': study['created_utc_epoch'],
                  'deadline_utc_epoch': study['deadline_utc_epoch'], 'evidence': study['evidence']})
    write_immutable(path, value)
    verify_sources(read_plan(path))
    return path


def prepare(cwd, reservation_manifest, *, hours=12, world_size=16, now=None):
    from .compression_finalize_v13 import verify as verify_source
    cwd = Path(cwd).absolute()
    path = cwd / ROOT / 'study.json'
    if path.exists():
        value = checked(path)
        verify(value)
        return value
    if not 0 < hours <= 12 or type(world_size) is not int or not 1 <= world_size <= 64:
        raise ValueError('bounded campaign required')
    source_path = cwd / SOURCE
    source = checked(source_path)
    verify_source(source)
    cwd = Path(source['cwd'])
    decision_path = Path(source['output_root']) / 'decision.json'
    decision = checked(decision_path)
    reservation = Path(reservation_manifest).absolute()
    from learned_score.mlsys_eval.common import load_manifest
    load_manifest(reservation)
    pins = {**source['input_hashes'], str(source_path): file_digest(source_path),
            str(decision_path): file_digest(decision_path), str(reservation): file_digest(reservation)}
    modules = ('compression_campaign_v14', 'compression_operations_v14',
               'compression_full_suite_v14', 'compression_text_v14',
               'compression_operations_v13a')
    for module in modules:
        p = cwd / 'learned_score/mlsys_sprint' / (module + '.py')
        pins[str(p)] = file_digest(p)
    for script in ('mlsys_compression_v14_clustera.pbs', 'mlsys_compression_v14_clustera_worker.sh',
                   'mlsys_compression_v14_idle_worker.sh'):
        p = cwd / 'scripts' / script
        pins[str(p)] = file_digest(p)
    now = time.time() if now is None else now
    value = {k: source[k] for k in ('models', 'manifest', 'manifest_sha256', 'bytes_per_token',
             'generation_reserve', 'max_new_tokens', 'datasets')}
    value.update(format=FORMAT, cwd=str(cwd), output_root=str(cwd / ROOT),
                 source_study=str(source_path), decision_path=str(decision_path),
                 decision_sha256=decision['sha256'], reservation_manifest=str(reservation),
                 input_hashes=pins, created_utc_epoch=now, deadline_utc_epoch=now + hours * 3600,
                 world_size=world_size, scorer_training=False, rule_retuning=False,
                 dataset_is_policy_input=False, confirmation_labels_opened=False,
                 independent_confirmation=False, text_pilot=True,
                 evidence='post-hoc transfer with the frozen v13 rule; not independent confirmation',
                 allocation_comparison='identical aggregate KV capacity within each dataset; same selector and cache policy')
    value = seal(value)
    write_immutable(path, value)
    verify(value)
    return value


def advance(study):
    from . import compression_full_suite_v14 as full
    from . import compression_text_v14 as text
    verify(study)
    root = Path(study['output_root'])
    # Independent branches: a reporting failure must not block publication of
    # the other work. Persist the error visibly; never declare partial success.
    outcomes, errors = {}, {}
    for name, action in (('text', text.advance), ('full_suite', full.advance)):
        try:
            outcomes[name] = action(study)
        except Exception as exc:
            errors[name] = {'type': type(exc).__name__, 'error': str(exc)}
    complete = not errors and all(v.get('status') == 'complete' for v in outcomes.values())
    result = seal({'study_sha256': study['sha256'], 'time': time.time(),
                   'status': 'complete' if complete else 'running', 'branches': outcomes,
                   'errors': errors, 'independent_confirmation': False,
                   'deadline_utc_epoch': study['deadline_utc_epoch']})
    atomic_json(root / 'status.json', result)
    if complete:
        write_immutable(root / 'completion.json', seal({'study_sha256': study['sha256'],
                        'all_declared_physical_work_complete': True, 'independent_confirmation': False,
                        'frozen_decision_sha256': study['decision_sha256']}))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'advance', 'watch', 'verify'))
    p.add_argument('--study', type=Path, default=ROOT / 'study.json')
    p.add_argument('--reservation-manifest', type=Path)
    p.add_argument('--hours', type=float, default=12)
    p.add_argument('--world-size', type=int, default=16)
    a = p.parse_args(argv)
    if a.phase == 'prepare':
        if a.reservation_manifest is None:
            p.error('--reservation-manifest is required')
        study = prepare(Path.cwd(), a.reservation_manifest, hours=a.hours, world_size=a.world_size)
        print(json.dumps({'study': str(Path(study['output_root']) / 'study.json'), 'sha256': study['sha256']}))
        return
    study = checked(a.study)
    verify(study)
    if a.phase == 'verify':
        print(json.dumps({'verified': True, 'sha256': study['sha256']}))
        return
    with (Path(study['output_root']) / 'coordinator.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.time() < study['deadline_utc_epoch']:
            status = advance(study)
            print(json.dumps(status, sort_keys=True), flush=True)
            if status['status'] == 'complete' or a.phase != 'watch':
                return
            time.sleep(30)
    atomic_json(Path(study['output_root']) / 'deadline.json', {'time': time.time(),
                'complete': False, 'resume_requires_new_bounded_campaign': True})


if __name__ == '__main__':
    main()
