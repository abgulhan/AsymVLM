"""Eight-hour, additive coverage-diagnosis campaign; historical runs stay frozen."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from .compression_matrix_v8 import seal, checked
from .followup_common import write_immutable
from .state import atomic_json, file_digest
from learned_score.mlsys_eval.common import load_manifest, seal_manifest

ROOT = Path('reproduction_results/mlsys_compression_20261025_v16')
MODELS = ('qwen25', 'llava_next', 'phi35')
START = datetime(2026, 10, 25, 3, 15, 3, tzinfo=timezone.utc).timestamp()
FREEZE = START + 3 * 3600
END = START + 8 * 3600
FORMAT = 'mlsys_compression_campaign_v16'


def initialize(project):
    project = Path(project).absolute()
    root = project / ROOT
    if (root / 'study.json').exists():
        return checked(root / 'study.json')
    visual_path = project / 'reproduction_results/mlsys_compression_20261023_v15/visual/study.json'
    phi_path = project / 'reproduction_results/mlsys_compression_20261023_v15/third-model/study.json'
    old_path = project / 'reproduction_results/mlsys_compression_20261022_v13/study.json'
    audit_path = project / 'reproduction_results/mlsys_compression_20261022_v12/admission/provenance-reconciliation.json'
    visual, phi, old, audit = (checked(p) for p in (visual_path, phi_path, old_path, audit_path))
    a, b = load_manifest(visual['manifest']), load_manifest(phi['manifest'])
    available = {r['row_hash'] for r in b['splits']['evaluation']}
    groups = defaultdict(list)
    for row in a['splits']['evaluation']:
        groups[row['group_id']].append(row)
    common = sorted((r for rows in groups.values() if all(r['row_hash'] in available for r in rows)
                     for r in rows), key=lambda r: r['row_hash'])
    if not common or any(r.get('known_training_calibration_overlap') is not False for r in common):
        raise ValueError('nonempty, previously admitted training-excluded common panel required')
    for key, rows in groups.items():
        retained = [r for r in common if r['group_id'] == key]
        if retained and rows[0]['dataset'] == 'MME' and len(retained) != 2:
            raise ValueError('MME image pairs must remain intact')
    manifest = seal_manifest({'format': a['format'], 'dataset': a['dataset'],
        'splits': {'fit': [], 'screening': [], 'evaluation': common, 'phi_train': []},
        'metadata': {'independent_confirmation': False,
            'evidence': 'frozen_followup_on_previously_examined_common_sources_not_independent_confirmation',
            'known_training_calibration_excluded': True,
            'source_manifests': [visual['manifest'], phi['manifest']],
            'source_manifest_sha256': [a['manifest_sha256'], b['manifest_sha256']],
            'selection_uses_labels': False, 'complete_group_sampling': True}})
    manifest_path = root / 'data/common-evaluation.json'
    write_immutable(manifest_path, manifest)
    unresolved = [{k: r[k] for k in ('artifact_path', 'artifact_sha256', 'possible_domains', 'reason')}
                  for r in audit['receipts'] if not r['resolved']]
    audit_result = seal({'format': FORMAT + '_identity_audit', 'started_utc_epoch': START,
        'maximum_audit_seconds': 2700, 'observed_utc_epoch': time.time(),
        'source': str(audit_path), 'source_sha256': file_digest(audit_path),
        'unresolved_artifacts': unresolved, 'independent_confirmation_authorized': False,
        'protected_reservations_opened': False, 'labels_loaded': False,
        'action': 'use_prior_approved_common_panel_without_relabeling_as_independent',
        'requests_by_dataset': dict(Counter(r['dataset'] for r in common))})
    write_immutable(root / 'admission/initial.json', audit_result)
    model_specs = {**visual['models'], 'phi35': phi['model']}
    paths = [visual_path, phi_path, old_path, audit_path, manifest_path,
             Path(visual['manifest']), Path(phi['manifest']), Path(old['manifest'])]
    study = seal({'format': FORMAT, 'cwd': str(project), 'output_root': str(root),
        'created_utc_epoch': START, 'method_freeze_utc_epoch': FREEZE, 'deadline_utc_epoch': END,
        'models': model_specs, 'model_order': list(MODELS),
        'development_manifests': {'qwen25': old['manifest'], 'llava_next': old['manifest'], 'phi35': phi['manifest']},
        'evaluation_manifest': str(manifest_path),
        'source_studies': {'visual': str(visual_path), 'phi35': str(phi_path), 'incumbent': str(old_path)},
        'profiles': {'conservative': {'mean_loss_pp': .5, 'domain_loss_pp': 1.},
                     'primary': {'mean_loss_pp': 1., 'domain_loss_pp': 2.},
                     'aggressive': {'mean_loss_pp': 2., 'domain_loss_pp': 3.}},
        'outer_folds': 5, 'retentions': list(range(10, 101, 10)),
        'importance_floor': .1, 'representation_floor': .1,
        'max_new_tokens': 128, 'generated_text_budget': 512,
        'text_eviction_active_in_vision_runs': False, 'scorer_training': False,
        'independent_confirmation': False, 'final_selection_uses_evaluation_labels': False,
        'input_hashes': {str(p): file_digest(p) for p in paths}})
    write_immutable(root / 'study.json', study)
    return study


def publish_worker_plan(study, model, stage, phase, split, *, world_size=8,
                        request_ids=None, retentions=tuple(range(10, 101, 10)),
                        schedule=None, feature_root=None, manifest=None):
    """Publish one immutable existing-worker queue; no scheduler side effects."""
    from . import compression_execution_v16 as execution
    from .worker import read_plan, verify_sources
    from .adaptive_workflow_v4 import environment
    root = Path(study['output_root'])
    directory = root / 'plans' / (stage + '-' + model)
    plan_path = directory / 'plan.json'
    if plan_path.exists():
        plan = read_plan(plan_path)
        verify_sources(plan)
        return plan_path
    output = root / model / stage
    task = execution.build_task(study, phase, model, split, output_root=output,
        request_ids=request_ids, schedule=schedule, feature_root=feature_root,
        retentions=retentions, manifest=manifest)
    if study.get('supplemental_study_path'):
        task = seal({**{k: v for k, v in task.items() if k != 'sha256'},
            'input_hashes': {**task['input_hashes'],
                str(Path(study['supplemental_study_path']).resolve()): file_digest(study['supplemental_study_path'])}})
        execution.validate_task(task, phase)
    task_path = directory / 'task.json'
    write_immutable(task_path, task)
    pins = {**task['input_hashes'], str(task_path): file_digest(task_path)}
    tasks = []
    for rank in range(world_size):
        command = [str(Path(study['cwd']) / '.conda/asymvlm/bin/python'), '-B', '-u', '-m',
            'learned_score.mlsys_sprint.compression_execution_v16', phase,
            '--task', str(task_path), '--rank', str(rank), '--world-size', str(world_size)]
        tasks.append({'id': f'compression-v16-{stage}-{model}-{rank}', 'tag': 'compression-v16',
            'resource': 'gpu', 'model_tag': model, 'depends_on': [],
            'minimum_start_seconds': 300 if phase == 'physical' else 120,
            'argv': command, 'env': {**environment(Path(study['cwd'])), 'PYTHONDONTWRITEBYTECODE': '1'},
            'artifacts': [{'path': str(output / f'rank{rank}.complete.json'), 'passed': True}]})
    plan = seal({'format': FORMAT + '_plan', 'cwd': study['cwd'], 'tasks': tasks,
        'input_hashes': pins, 'unexpected_retries': 1, 'created_utc_epoch': study['created_utc_epoch'],
        'deadline_utc_epoch': study['deadline_utc_epoch'],
        'evidence': 'development_or_previously_examined_benchmark_not_independent_confirmation'})
    write_immutable(plan_path, plan)
    verify_sources(read_plan(plan_path))
    return plan_path


def supplemental_phi(study):
    """Admit existing ChartQA/MME development sources, never evaluation labels."""
    from .compression_data_v9 import qualify_keys, group_closure
    root = Path(study['output_root'])
    path = root / 'supplemental-phi-study.json'
    if path.exists():
        return checked(path)
    qwen = load_manifest(study['development_manifests']['qwen25'])
    phi = load_manifest(study['development_manifests']['phi35'])
    excluded = set()
    for rows in phi['splits'].values():
        for row in rows:
            excluded.update(qualify_keys(row))
    # Also close against every evaluation identity, including image aliases.
    for row in load_manifest(study['evaluation_manifest'])['splits']['evaluation']:
        excluded.update(qualify_keys(row))
    splits, proofs = {'evaluation': [], 'phi_train': []}, {}
    for split in ('fit', 'screening'):
        proposed = [r for r in qwen['splits'][split] if r['dataset'] in ('ChartQA_TEST', 'MME')]
        clean, overlapping = group_closure(proposed, excluded)
        splits[split] = clean
        proofs[split] = {'proposed': len(proposed), 'admitted': len(clean),
            'overlapping_groups': sorted(overlapping), 'domains': dict(Counter(r['dataset'] for r in clean))}
        for row in clean:
            excluded.update(qualify_keys(row))
    manifest = seal_manifest({'format': qwen['format'], 'dataset': qwen['dataset'], 'splits': splits,
        'metadata': {'development_only': True, 'independent_confirmation': False,
            'source_manifest_sha256': qwen['manifest_sha256'],
            'phi_exclusion_manifest_sha256': phi['manifest_sha256'],
            'identity_only_admission': True, 'labels_loaded': False,
            'selection_uses_labels': False, 'group_closure_proofs': proofs}})
    manifest_path = root / 'data/phi-supplemental-development.json'
    write_immutable(manifest_path, manifest)
    derived = seal({**{k: v for k, v in study.items() if k != 'sha256'},
        'parent_study_sha256': study['sha256'], 'supplemental_study_path': str(path),
        'supplemental_development_only': True,
        'development_manifests': {**study['development_manifests'], 'phi35': str(manifest_path)},
        'input_hashes': {**study['input_hashes'], str(manifest_path): file_digest(manifest_path)}})
    write_immutable(path, derived)
    return derived


def complete_plan(path):
    from .worker import read_plan, _state, _completed
    path = Path(path)
    if not path.exists():
        return False
    plan = read_plan(path)
    return all(_completed(t, _state(path.parent / 'state' / (t['id'] + '.json')), plan)
               for t in plan['tasks'])


def advance_initial(study):
    root, result = Path(study['output_root']), {}
    for model in MODELS:
        canary = publish_worker_plan(study, model, 'canary', 'canary', 'fit', world_size=1)
        if not complete_plan(canary):
            result[model] = 'waiting_for_native_canary'
            continue
        for split, size in (('fit', 8), ('screening', 4), ('evaluation', 16)):
            publish_worker_plan(study, model, 'features-' + split,
                'eval-features' if split == 'evaluation' else 'dev-features', split, world_size=size)
        if model == 'phi35':
            for split, size in (('fit', 24), ('screening', 12)):
                publish_worker_plan(study, model, 'labels-' + split, 'physical', split,
                    world_size=size, retentions=(10, 20, 40, 60, 80))
            supplemental = supplemental_phi(study)
            for split in ('fit', 'screening'):
                publish_worker_plan(supplemental, model, 'features-extra-' + split,
                    'dev-features', split, world_size=4)
                publish_worker_plan(supplemental, model, 'labels-extra-' + split,
                    'physical', split, world_size=12)
        result[model] = 'development_and_answer_free_evaluation_published'
    atomic_json(root / 'initial-status.json', {'time': time.time(), 'models': result})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('initialize', 'advance', 'watch'))
    parser.add_argument('--project', default=str(Path.cwd()))
    args = parser.parse_args(argv)
    value = initialize(args.project)
    if args.action == 'initialize':
        print(json.dumps({'study': str(Path(value['output_root']) / 'study.json'),
                          'deadline_utc_epoch': value['deadline_utc_epoch'],
                          'evaluation_manifest': value['evaluation_manifest']}))
    else:
        while time.time() < value['deadline_utc_epoch']:
            print(json.dumps(advance_initial(value)), flush=True)
            if args.action != 'watch':
                break
            time.sleep(min(30, max(0, value['deadline_utc_epoch'] - time.time())))


if __name__ == '__main__':
    main()
