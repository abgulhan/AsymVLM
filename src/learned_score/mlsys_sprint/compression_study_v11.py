"""Restartable selector/stopping-rule development; v10 is immutable input.

No confirmation labels are opened. Publication does not launch GPU jobs. The
explicit watch CLI advances CPU analysis and publishes physical worker tasks.
"""
import argparse
from collections import Counter
import copy
import fcntl
import json
from pathlib import Path
import time

from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest

ROOT = Path('reproduction_results/mlsys_compression_20261020_v11')
PREVIOUS = Path('reproduction_results/mlsys_compression_20261020_v10/study.json')
FORMAT = 'mlsys_compression_study_v11'
MODELS = ('qwen25', 'llava_next')
TARGETS = (50, 70, 90)


def prepare(cwd, *, hours=24):
    cwd = Path(cwd).resolve()
    path = cwd / ROOT / 'study.json'
    if path.exists():
        study = checked(path)
        verify(study)
        return study
    if not 0 < hours <= 24:
        raise ValueError('bounded execution window must be at most 24 hours')
    from . import compression_study_v10 as previous
    old_path = cwd / PREVIOUS
    old = checked(old_path)
    previous.verify(old)
    pins = {**old['input_hashes'], str(old_path): file_digest(old_path)}
    now = time.time()
    study = seal({'format': FORMAT, 'cwd': str(cwd), 'output_root': str(cwd / ROOT),
        'v10_study': str(old_path), 'v9_study': old['v9_study'], 'v8_study': old['v8_study'],
        'old_study': old['old_study'], 'models': old['models'], 'manifest': old['manifest'],
        'manifest_sha256': old['manifest_sha256'], 'input_hashes': pins,
        'world_size': 16, 'created_utc_epoch': now, 'deadline_utc_epoch': now + hours*3600,
        'keep_percent': list(range(10, 101, 10)), 'targets_percent': list(TARGETS),
        'datasets': old['datasets'], 'max_new_tokens': 128, 'generated_budget': 512,
        'text_eviction': False, 'scorer_training': False, 'dataset_is_policy_input': False,
        'outer_folds': 5, 'inner_folds': 3, 'accuracy_loss_reference_pp': 2.,
        'accuracy_loss_reference_is_hard_gate': False,
        'selection_objective': 'equal-benchmark accuracy at a reported compression; exact matched physical controls',
        'selector_scope': 'projected_feature_adaptations_not_official_pipeline',
        'selector_pool': {'qwen25': ['scorer', 'divprune'], 'llava_next': ['scorer', 'divprune', 'cdpruner']},
        'baseline_scope': old['baseline_scope'], 'confirmation_labels_opened': False,
        'confirmation_status': 'blocked_until_historical_identity_audit_resolved',
        'independent_confirmation': False, 'evidence': 'development on previously inspected data'})
    write_immutable(path, study)
    verify(study)
    return study


def verify(study):
    from . import compression_execution_v11 as execution
    if (study.get('format') != FORMAT or Path(study['output_root']) != Path(study['cwd']) / ROOT
            or study.get('confirmation_labels_opened') is not False
            or study.get('dataset_is_policy_input') is not False):
        raise ValueError('invalid development study scope')
    execution.verify(study)


def diagnostics(study, model):
    from . import compression_diagnostics_v11 as analysis
    path = Path(study['output_root']) / 'diagnostics' / (model + '.json')
    if path.exists():
        report = checked(path)
    else:
        report = analysis.build_report(model, study['v10_study'])
        write_immutable(path, report)
    if report['provenance']['study_sha256'] != checked(study['v10_study'])['sha256']:
        raise ValueError('diagnosis refers to a different source study')
    for source, expected in report['provenance']['input_hashes'].items():
        if file_digest(source) != expected:
            raise ValueError('diagnostic input changed: ' + source)
    return report


def publish_initial(study, model, report):
    from . import compression_execution_v11 as execution
    from . import compression_diagnostics_v11 as analysis
    alternative = report['strongest_cached_alternative']
    _, _, index, provenance = analysis.load_development(model, study['v10_study'])
    root = Path(study['output_root'])
    reuse_path = root / model / 'development/reuse-audit.json'
    write_immutable(reuse_path, seal({'model': model, 'study_sha256': study['sha256'],
        'physical_inventory_keys': [list(k) for k in sorted(index)], 'provenance': provenance,
        'scope': 'frozen original selector implementation only; no repaired-ranker reuse'}))
    plans = execution.prepare_features(study, model, selectors=('scorer', alternative))['plans']
    for split in execution.SPLITS:
        _, _, _, features = execution.development_features(study, model, split)
        exclusions = [k for k in index if k[0] in features]
        schedule = execution.selector_schedule(study, model, split, alternative, exclude_pairs=exclusions)
        if schedule['rows']:
            plans.append(str(execution.publish_schedule(study, model, split+'-selector-grid', split, schedule)))
    return plans


def _merged_rows(study, model, selector, features):
    from . import compression_diagnostics_v11 as analysis
    from . import compression_execution_v11 as execution
    from .compression_study_v8 import physical_index
    from .compression_policy_v10 import canonical_dataset
    rows, _, index, _ = analysis.load_development(model, study['v10_study'])
    if selector != 'scorer':
        for split in execution.SPLITS:
            added = execution.collect_stage(study, model, split+'-selector-grid')
            if added is None:
                # A split fully covered by audited historical outputs publishes
                # no new task. Do not wait forever for an intentionally absent
                # plan, but require every needed selector/count to be present.
                _, _, _, native = execution.development_features(study,model,split)
                missing = [(request,'full' if (n*p+99)//100==n else selector,(n*p+99)//100)
                    for request,feature in native.items() for n in [feature['input_visual_tokens']]
                    for p in range(10,101,10)]
                if any(key not in index for key in missing):
                    return None
                continue
            normalized = [{**row,'dataset':canonical_dataset(row['dataset'])} for row in added]
            # Equality includes patch indices, predictions, native geometry and
            # the runtime contract, not just an accidentally equal quality score.
            index = physical_index([*index.values(),*normalized])
        rows = analysis.selector_rows(rows, index, selector, tuple(range(10, 101, 10)))
    if set(features) != {r['row_hash'] for r in rows}:
        raise ValueError('stopping features must cover the entire declared development cohort')
    result = copy.deepcopy(rows)
    for row in result:
        value = features[row['row_hash']]
        if (any(value.get(k)!=row[k] for k in ('input_visual_tokens','input_text_tokens','group_id'))
                or canonical_dataset(value['dataset'])!=row['dataset']):
            raise ValueError('feature/quality geometry mismatch')
        row['curves'] = value['curves']
    return result


def calibrate(study, model, alternative, features):
    from . import compression_policy_v11 as policy
    root = Path(study['output_root']) / model / 'development'
    results, input_rows = {}, {}
    for selector in ('scorer', alternative):
        rows = _merged_rows(study, model, selector, features)
        if rows is None:
            return None
        input_rows[selector] = rows
        for target in TARGETS:
            path = root / f'{selector}-p{target:03d}-nested.json'
            identity = digest({'rows': rows, 'selector': selector, 'target': target,
                               'policy_source': file_digest(policy.__file__)})
            if path.exists():
                value = checked(path)
                if value['input_identity'] != identity:
                    raise ValueError('cannot replace a calibrated development result')
            else:
                result = policy.nested_validate(rows, selector, target/100)
                value = seal({'study_sha256': study['sha256'], 'model': model,
                    'input_identity': identity, 'policy_source_sha256': file_digest(policy.__file__), 'result': result})
                write_immutable(path, value)
            results[(selector, target)] = value['result']
    # One allocation-feature/selector family across targets and datasets.
    # All choices are development choices; physical comparisons remain required.
    winner = min(('scorer', alternative), key=lambda s: (
        -sum(results[(s,t)]['summary']['macro_accuracy'] for t in TARGETS)/len(TARGETS),
        -min(results[(s,t)]['summary']['worst_delta_full_pp'] for t in TARGETS), s))
    decision = seal({'study_sha256': study['sha256'], 'model': model, 'alternative_selector': alternative,
        'allocation_feature_selector': winner, 'targets': {str(t): results[(winner,t)]['sha256'] for t in TARGETS},
        'evidence': 'development-selected; no independent confirmation', 'dataset_is_policy_input': False})
    write_immutable(root/'selected.json', decision)
    return decision, results, input_rows


def physical_inventory(study, model):
    from . import compression_diagnostics_v11 as analysis
    from . import compression_execution_v11 as execution
    from .compression_policy_v10 import canonical_dataset
    from .compression_study_v8 import physical_index
    _, _, index, _ = analysis.load_development(model, study['v10_study'])
    for split in execution.SPLITS:
        added = execution.collect_stage(study, model, split+'-selector-grid')
        if added is not None:
            index = physical_index([*index.values(), *[{**r, 'dataset': canonical_dataset(r['dataset'])} for r in added]])
    return index


def publish_factorial(study, model, alternative, decision, results, input_rows, features=None):
    from . import compression_policy_v11 as policy
    from . import compression_execution_v11 as execution
    from .compression_reuse_v11 import split_missing
    from .adaptive_planning_v4 import _geometry
    root, paths = Path(study['output_root']), []
    rows = input_rows['scorer']
    inventory = physical_inventory(study, model)
    for target in TARGETS:
        counts = results[(decision['allocation_feature_selector'], target)]['counts']
        factorial = policy.factorial_assignments(rows, counts, alternative)
        write_immutable(root/model/f'development/factorial-p{target:03d}.json', seal(factorial))
        for split in execution.SPLITS:
            _, _, metadata, native_features = execution.development_features(study, model, split)
            scheduled = {}
            for request, feature in sorted(native_features.items()):
                n = feature['input_visual_tokens']
                cells = []
                for key, cell in factorial['cells'].items():
                    count = cell['counts'][request]
                    cells.append({'key': key, 'selector': 'full' if count == n else cell['selector'],
                        'selected_count': count, 'mode': 'exact', 'policy': 'v11_frozen_nested_development_counts'})
                scheduled[request] = {**_geometry(feature), 'cells': cells}
            schedule = seal({'format': execution.FORMAT+'_schedule', 'study_sha256': study['sha256'],
                'model': model, 'split': split, 'candidate': 'current_rank32',
                'manifest_sha256': study['manifest_sha256'], 'checkpoint_sha256': metadata['checkpoint_sha256'],
                'cache_contract': metadata['cache_contract'], 'features_sha256': digest(native_features),
                'rows': scheduled, 'answers_used_to_choose_counts': False,
                'training_labels_used_inside_other_folds': True,
                'evidence': 'development-selected allocation; individual counts from excluded-fold rules'})
            stage = f'{split}-matched-p{target:03d}'
            write_immutable(root/model/f'schedules/{stage}-master-v11.json', schedule)
            missing, reuse = split_missing(schedule, inventory, features)
            write_immutable(root/model/f'schedules/{stage}-reuse-v11.json', reuse)
            if missing['rows']:
                paths.append(str(execution.publish_schedule(study, model, stage, split, missing)))
    return paths


def worker_summary(study):
    from .worker import read_plan, _state
    records = []
    for path in sorted((Path(study['output_root'])/'plans').glob('*/plan.json')):
        plan = read_plan(path)
        for task in plan['tasks']:
            state = _state(path.parent/'state'/(task['id']+'.json'))
            records.append({'task': task['id'], 'status': state.get('status', 'queued'),
                'failures': state.get('failures', 0), 'log': state.get('log')})
    return {'counts': dict(Counter(r['status'] for r in records)), 'tasks': records}


def physical_report(study, model, alternative):
    from . import compression_execution_v11 as execution
    from . import compression_report_v11 as reporting
    root = Path(study['output_root'])
    destination = root/'reports'/(model+'-matched.json')
    if destination.exists():
        return checked(destination)
    measured, expected = [], {}
    for target in TARGETS:
        for split in execution.SPLITS:
            stage = f'{split}-matched-p{target:03d}'
            receipt_path = root/model/f'schedules/{stage}-reuse-v11.json'
            if not receipt_path.exists():
                return None
            receipt = checked(receipt_path)
            for source, expected_hash in receipt['source_pins'].items():
                if file_digest(source) != expected_hash:
                    raise ValueError('reused matched physical source changed: '+source)
            if receipt['counts']['missing_unique_cells'] == 0:
                continue
            added = execution.collect_stage(study, model, stage)
            if added is None:
                return None
            measured.extend(added)
        expected[str(target)] = checked(root/model/f'development/factorial-p{target:03d}.json')['counts']
    index = physical_inventory(study, model)
    report = reporting.report(study, model, [*index.values(), *measured], expected,
                              alternative_selector=alternative)
    write_immutable(destination, report)
    if hasattr(reporting, 'markdown'):
        from .adaptive_report_v1 import _atomic_text
        _atomic_text(root/'reports'/(model+'-TABLES.md'), reporting.markdown(report))
    return report


def configure(study):
    from . import compression_execution_v11 as execution
    from . import compression_operations_v11 as operations
    root = Path(study['output_root'])
    for role in ('clusterb', 'clustera'):
        path = root/'coordinator'/(role+'-config.json')
        if path.exists():
            operations.read_config(path)
            continue
        plans = [root/'plans'/name/'plan.json' for name in execution.plan_names()
                 if role == 'clusterb' or 'llava_next' in name]
        config = operations.build_config(study['cwd'], plans, role=role,
            deadline_utc_epoch=study['deadline_utc_epoch'], created_utc_epoch=study['created_utc_epoch'], study_root=root)
        write_immutable(path, config)


def advance(study):
    from . import compression_execution_v11 as execution
    verify(study)
    status = {'study_sha256': study['sha256'], 'time': time.time(), 'models': {},
        'independent_confirmation': False, 'confirmation_status': study['confirmation_status']}
    for model in MODELS:
        report = diagnostics(study, model)
        alternative = report['strongest_cached_alternative']
        publish_initial(study, model, report)
        entry = {'alternative_selector': alternative, 'status': 'waiting_for_physical_features_and_selector_grid'}
        features = execution.collect_features(study, model)
        if features is not None:
            fitted = calibrate(study, model, alternative, features)
            if fitted is not None:
                decision, results, input_rows = fitted
                publish_factorial(study, model, alternative, decision, results, input_rows, features)
                entry.update(status='waiting_for_exact_matched_physical_comparisons', decision=decision)
                measured = physical_report(study, model, alternative)
                if measured is not None:
                    entry.update(status='quality_development_complete_systems_pending',
                        matched_report=str(Path(study['output_root'])/'reports'/(model+'-matched.json')),
                        matched_report_sha256=measured['sha256'], quality_complete=True)
                    from . import compression_systems_v11 as systems
                    entry['systems'] = systems.publish_systems(study, model)
                    timing = systems.collect(study, model)
                    if timing is not None:
                        timing_report = seal(systems.summarize(timing))
                        timing_path = Path(study['output_root'])/'reports'/(model+'-systems.json')
                        write_immutable(timing_path, timing_report)
                        entry.update(status='development_complete', systems_complete=True,
                            systems_report=str(timing_path))
        status['models'][model] = entry
    status['workers'] = worker_summary(study)
    status['development_complete'] = all(e.get('quality_complete') and e.get('systems_complete')
                                         for e in status['models'].values())
    status['development_terminal'] = all(e.get('quality_complete') and (e.get('systems_complete') or
        (e.get('systems') or {}).get('status') == 'not_ready') for e in status['models'].values())
    status['remaining_after_physical'] = ['independent data admission before scientific confirmation']
    if not status['development_complete']:
        status['remaining_after_physical'].insert(0, 'finish exact matched quality and exclusive systems measurements')
    atomic_json(Path(study['output_root'])/'coordinator/status.json', status)
    return status


def watch(study, interval=30):
    if not 5 <= interval <= 60:
        raise ValueError('watch interval must be 5..60 seconds')
    root = Path(study['output_root'])/'coordinator'
    root.mkdir(parents=True, exist_ok=True)
    with (root/'coordinator.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.time() < study['deadline_utc_epoch']:
            try:
                status = advance(study)
            except Exception as error:
                atomic_json(root/'error.json', {'time': time.time(), 'type': type(error).__name__, 'error': str(error)})
                raise
            print(json.dumps({'models': status['models'], 'workers': status['workers']['counts']}), flush=True)
            if status['development_terminal']:
                return
            time.sleep(min(interval, max(0, study['deadline_utc_epoch']-time.time())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'advance', 'watch', 'configure', 'status'))
    parser.add_argument('--cwd', type=Path, default=Path.cwd())
    parser.add_argument('--hours', type=float, default=24)
    args = parser.parse_args(argv)
    study = prepare(args.cwd, hours=args.hours)
    if args.phase == 'prepare':
        result = {'study': str(Path(study['output_root'])/'study.json'), 'sha256': study['sha256']}
    elif args.phase == 'configure':
        configure(study)
        result = {'configured': True}
    elif args.phase == 'watch':
        result = watch(study)
    elif args.phase == 'status':
        result = worker_summary(study)
    else:
        result = advance(study)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
