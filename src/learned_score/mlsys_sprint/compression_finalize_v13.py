"""Bounded method finalization, preserving all historical experiments.

Development selection is never independent confirmation. Failed promotion does
not prevent diagnostic comparisons, and incomplete cohorts never elect a winner.
"""
import argparse
import fcntl
import json
from pathlib import Path
import time

from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest

ROOT = Path('reproduction_results/mlsys_compression_20261022_v13')
SOURCE = Path('reproduction_results/mlsys_compression_20261022_v12/study.json')
FORMAT = 'mlsys_compression_finalize_v13'
MODELS = ('qwen25', 'llava_next')


def verify(study, *, sources=True):
    from .compression_policy_v13 import VARIANTS
    if study.get('sha256') != digest({k: v for k, v in study.items() if k != 'sha256'}):
        raise ValueError('v13 study integrity mismatch')
    cwd = Path(study['cwd']).resolve()
    if (study.get('format') != FORMAT or Path(study['output_root']).resolve() != cwd / ROOT
            or study.get('confirmation_labels_opened') is not False
            or study.get('scorer_training') is not False
            or study.get('dataset_is_policy_input') is not False
            or study.get('evaluation_constraints') != {'mean_loss_pp': 1., 'domain_loss_pp': 2.}
            or tuple(study.get('candidates', ())) != tuple(VARIANTS)
            or not 0 < study['deadline_utc_epoch'] - study['created_utc_epoch'] <= 18000
            or study['method_freeze_utc_epoch'] > study['deadline_utc_epoch']):
        raise ValueError('invalid bounded finalization scope')
    from .compression_study_v12 import verify as verify_source
    source = checked(study['source_study'])
    verify_source(source, sources=sources)
    if (study['models'] != source['models'] or study['manifest_sha256'] != source['manifest_sha256']
            or file_digest(study['source_study']) != study['source_study_sha256']):
        raise ValueError('historical model/data identity changed')
    if sources:
        for path, expected in study['input_hashes'].items():
            if file_digest(path) != expected:
                raise ValueError('pinned finalization input changed: ' + path)
    return source


def prepare(cwd, *, now=None, hours=5, world_size=16):
    from .compression_study_v12 import verify as verify_source
    from .compression_policy_v13 import VARIANTS
    cwd = Path(cwd).absolute()
    path = cwd / ROOT / 'study.json'
    if path.exists():
        value = checked(path)
        verify(value)
        return value
    if not 0 < hours <= 5 or type(world_size) is not int or not 1 <= world_size <= 64:
        raise ValueError('bounded duration and shard count required')
    source_path = cwd / SOURCE
    source = checked(source_path)
    verify_source(source)
    # Keep the shared filesystem spelling from the source, not a host-local alias.
    cwd = Path(source['cwd'])
    pins = {str(source_path): file_digest(source_path), **source['input_hashes']}
    for model in MODELS:
        for suffix in ('rows.json', 'report.json'):
            p = Path(source['output_root']) / model / 'development' / suffix
            pins[str(p)] = file_digest(p)
    reconciliation = Path(source['output_root']) / 'admission/provenance-reconciliation.json'
    pins[str(reconciliation)] = file_digest(reconciliation)
    for module in ('compression_finalize_v13', 'compression_policy_v13', 'compression_execution_v13',
                   'compression_operations_v13', 'compression_report_v13', 'compression_systems_v13'):
        p = cwd / 'learned_score/mlsys_sprint' / (module + '.py')
        pins[str(p)] = file_digest(p)
    for name in ('mlsys_compression_v13_clustera.pbs', 'mlsys_compression_v13_clustera_worker.sh',
                 'mlsys_compression_v13_idle_worker.sh'):
        p = cwd / 'scripts' / name
        pins[str(p)] = file_digest(p)
    now = time.time() if now is None else now
    value = {k: source[k] for k in ('models', 'manifest', 'manifest_sha256', 'bytes_per_token',
        'generation_reserve', 'max_new_tokens', 'datasets', 'outer_folds', 'inner_folds',
        'maximum_latency_overhead_fraction', 'cost_tie_fraction')}
    value.update(format=FORMAT, cwd=str(cwd), output_root=str(cwd / ROOT),
        source_study=str(source_path), source_study_sha256=file_digest(source_path),
        created_utc_epoch=now, deadline_utc_epoch=now + hours * 3600,
        method_freeze_utc_epoch=now + min(4, hours) * 3600,
        calibration_deadline_utc_epoch=now + min(1, hours) * 3600,
        world_size=world_size, input_hashes=pins, candidates=VARIANTS,
        evaluation_constraints={'mean_loss_pp': 1., 'domain_loss_pp': 2.},
        max_mean_loss_pp=1., max_domain_loss_pp=2.,
        keep_percent=list(range(10, 101, 10)),
        scorer_training=False, dataset_is_policy_input=False, confirmation_labels_opened=False,
        independent_confirmation=False, text_eviction=False, text_pilot=False,
        allow_divprune_replacement=True, allow_static_winner=True,
        evidence='development selection on previously examined physical outcomes',
        full_suite='blocked_historical_identity_admission_not_reopened',
        selection_objective='minimum_equal_model_mean_allocated_kv_bytes',
        selection_tie_break='within_1pct_fixed_then_worst_loss_then_mean_loss_then_method_id')
    value = seal(value)
    write_immutable(cwd / ROOT / 'study.json', value)
    verify(value)
    return value


def calibrate(study, model):
    from . import compression_policy_v13 as policy
    source = verify(study)
    root = Path(study['output_root'])
    path = root / model / 'development/report.json'
    if path.exists():
        result = checked(path)
        if result['study_sha256'] != study['sha256']:
            raise ValueError('calibration belongs to a different study')
        return result
    if time.time() >= study['calibration_deadline_utc_epoch']:
        raise ValueError('calibration deadline reached; candidate catalog remains frozen')
    original = checked(Path(source['output_root']) / model / 'development/rows.json')
    rows = original['rows']
    result = policy.development_report(rows, model=model, bytes_per_token=study['bytes_per_token'][model],
        generation_reserve=study['generation_reserve'], outer_folds=study['outer_folds'], inner_folds=study['inner_folds'])
    write_immutable(root / model / 'development/rows.json', original)
    value = seal({'study_sha256': study['sha256'], 'model': model, 'rows_sha256': digest(rows),
        'result': result, 'evidence': study['evidence'], 'independent_confirmation': False})
    write_immutable(path, value)
    return value


def advance(study):
    from . import compression_policy_v13 as policy
    from . import compression_execution_v13 as execution
    from . import compression_report_v13 as reporting
    verify(study)
    root = Path(study['output_root'])
    reports = {model: calibrate(study, model) for model in MODELS}
    profile_path = root / 'profiles.json'
    if profile_path.exists():
        profiles = checked(profile_path)
    else:
        profiles = policy.select_profiles({m: r['result'] for m, r in reports.items()})
        write_immutable(profile_path, profiles)
    for model in MODELS:
        diagnostic_path = root / model / 'development/failure-diagnostics.json'
        if not diagnostic_path.exists():
            rows = checked(root / model / 'development/rows.json')['rows']
            variants = sorted({'rank32_coverage__original', profiles['forms']['adaptive']['variant_id']})
            diagnostics = {variant: policy.failure_diagnostics(rows,
                reports[model]['result']['procedures'][variant]['counts'],
                bytes_per_token=study['bytes_per_token'][model], generation_reserve=study['generation_reserve'])
                for variant in variants}
            write_immutable(diagnostic_path, seal({'study_sha256': study['sha256'],
                'model': model, 'variants': diagnostics, 'diagnostic_only': True}))
    plans = {model: execution.publish_comparison(study, model, reports[model], profiles) for model in MODELS}
    report = reporting.refresh(study, reports=reports, profiles=profiles)
    decision_path = root / 'decision.json'
    if not decision_path.exists() and (report['complete'] or time.time() >= study['method_freeze_utc_epoch']):
        decision = reporting.choose_final(study, report, reports, profiles)
        write_immutable(decision_path, decision)
    decision = checked(decision_path) if decision_path.exists() else None
    systems = {}
    if decision and decision.get('selected_method') and time.time() < study['deadline_utc_epoch']:
        from . import compression_systems_v13 as timing
        for model in MODELS:
            systems[model] = timing.advance(study, model, decision, report=reports[model])
    status = seal({'study_sha256': study['sha256'], 'time': time.time(), 'plans': plans,
        'stage': 'decision_frozen' if decision else 'physical_comparisons',
        'decision': decision, 'systems': systems, 'physical_comparisons_complete': report['complete'],
        'independent_confirmation': False, 'deadline_utc_epoch': study['deadline_utc_epoch']})
    atomic_json(root / 'status.json', status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'calibrate', 'advance', 'watch', 'report'))
    parser.add_argument('--cwd', type=Path, default=Path.cwd())
    parser.add_argument('--study', type=Path)
    parser.add_argument('--hours', type=float, default=5)
    parser.add_argument('--world-size', type=int, default=16)
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        study = prepare(args.cwd, hours=args.hours, world_size=args.world_size)
        print(json.dumps({'study': str(Path(study['output_root']) / 'study.json'), 'sha256': study['sha256']}), flush=True)
        return
    study = checked(args.study or args.cwd / ROOT / 'study.json')
    verify(study)
    if args.action == 'calibrate':
        if not args.model:
            parser.error('--model required for calibration')
        result = calibrate(study, args.model)
        print(json.dumps({'model': args.model, 'report_sha256': result['sha256']}), flush=True)
        return
    if args.action == 'report':
        from .compression_report_v13 import refresh
        print(json.dumps(refresh(study), sort_keys=True), flush=True)
        return
    root = Path(study['output_root'])
    with (root / 'coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                status = advance(study)
                print(json.dumps(status, sort_keys=True), flush=True)
            except Exception as exc:
                atomic_json(root / 'coordinator-error.json', {'time': time.time(), 'error': repr(exc)})
                raise
            if args.action != 'watch' or time.time() >= study['deadline_utc_epoch']:
                return
            if (status.get('decision') and status['physical_comparisons_complete']
                    and all(v.get('status') == 'complete' for v in status.get('systems', {}).values())):
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
