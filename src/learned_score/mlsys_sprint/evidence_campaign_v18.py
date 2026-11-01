"""Immutable preparation for the bounded evidence-supervision pilot.

No outcome or answer is used to select identities. All evaluation is reused
development, not a newly independent confirmation of previously tuned models.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from learned_score.mlsys_eval.common import seal_manifest, load_manifest
from .adaptive_data_v3 import dataset_family, _split_keys
from .followup_common import write_immutable
from .state import digest, file_digest

ROOT = 'reproduction_results/mlsys_evidence_20261025_v18'
BASE = 'reproduction_results/mlsys_20261012/manifest.json'
SOURCE = 'reproduction_results/mlsys_compression_20261025_v16/plans/features-fit-qwen25/task.json'
START = datetime(2026, 10, 25, 22, 9, 28, tzinfo=timezone.utc).timestamp()
SIZES = {'scorer_train': 50, 'scorer_screening': 32,
         'controller_calibration': 32, 'scorer_selection': 64}


def seal(value):
    return {**value, 'sha256': digest(value)}


def select_splits(base):
    """One question/image/document; reserve dev rows for all evaluation stages."""
    used, splits = set(), {}
    for split, per_domain in SIZES.items():
        pool = base['splits']['scorer_train' if split == 'scorer_train' else 'scorer_dev']
        chosen = []
        for domain in ('DocVQA', 'TextVQA'):
            candidates = [r for r in pool if dataset_family(r['dataset']) == domain]
            candidates.sort(key=lambda r: digest(['evidence-v18', split, r['row_hash']]))
            domain_rows = []
            for row in candidates:
                if row['source_split'] != 'train':
                    raise ValueError('pilot permits only official-training sources')
                keys = _split_keys(row)
                if keys & used:
                    continue
                used.update(keys)
                domain_rows.append(dict(row))
                if len(domain_rows) == per_domain:
                    break
            if len(domain_rows) != per_domain:
                raise ValueError(f'insufficient image-disjoint {domain}/{split} rows')
            chosen.extend(domain_rows)
        splits[split] = chosen
    return splits


def prepare(project):
    project = Path(project).resolve()
    root, basepath, sourcepath = project / ROOT, project / BASE, project / SOURCE
    base = load_manifest(basepath)
    source = json.loads(sourcepath.read_text())
    splits = select_splits(base)
    manifest = seal_manifest(dict(format='mlsys_adaptive_data_v3',
        dataset='ASYMVLM_EVIDENCE_V18_DEVELOPMENT', splits=splits,
        metadata={**base['metadata'], 'original_training_manifest': str(basepath),
            'original_training_manifest_sha256': file_digest(basepath),
            'evidence_status': 'reused_development_not_fresh_confirmation',
            'selection': 'identity_hash_only_image_document_disjoint_one_question_per_image',
            'counts': {k: dict(Counter(dataset_family(r['dataset']) for r in v))
                       for k, v in splits.items()},
            'locked_selection_rule': 'no selection labels opened before method freeze',
            'answers_used_for_identity_selection': False}))
    manifestpath = root / 'data/manifest.json'
    write_immutable(manifestpath, manifest)
    # The external teacher and visual audit share a preselected training subset.
    train = splits['scorer_train']
    teacher_ids, audit_ids = [], []
    for domain in ('DocVQA', 'TextVQA'):
        rows = sorted((r for r in train if dataset_family(r['dataset']) == domain),
                      key=lambda r: digest(['evidence-v18-teacher-audit', r['row_hash']]))
        teacher_ids.extend(r['row_hash'] for r in rows[:20])
        audit_ids.extend(r['row_hash'] for r in rows[:10])
    write_immutable(root / 'data/teacher-diagnostic-identities.json',
                    seal(dict(teacher_comparison_ids=teacher_ids, visual_audit_ids=audit_ids)))
    protocol = seal(dict(format='mlsys_evidence_v18_protocol', cwd=str(project),
        output_root=str(root), created_utc_epoch=START, stop_new_work_utc_epoch=START+10800,
        physical_deadline_utc_epoch=START+13800, deadline_utc_epoch=START+14400,
        maximum_gpu_workers=8, model=source['model'], manifest=str(manifestpath),
        training_manifest=str(basepath), source_task=str(sourcepath),
        stage_sizes=SIZES, retained_percentages=[50,70], primary_retention_percent=70,
        annotation_gate=dict(usable_minimum=16, audited=20),
        scorer_gate=dict(mean_gain_over_stronger_control_pp=1., minimum_domain_delta_pp=-1.),
        allocation=dict(retentions=[50,60,70,80,90,100], tau_step=.01,
                        maximum_mean_loss_from_full_pp=1., maximum_domain_loss_pp=2.,
                        matched_uniform_exact_tokens=True),
        final_status='development_pilot_no_production_promotion',
        input_hashes={str(p):file_digest(p) for p in (basepath,manifestpath,sourcepath)}))
    write_immutable(root / 'protocol.json', protocol)
    return protocol


def annotation_plan(protocol, split='scorer_train', world_size=8, teacher='qwen', teacher_model=None):
    project, root = Path(protocol['cwd']), Path(protocol['output_root'])
    stage = {'scorer_train':'train', 'scorer_screening':'screen'}[split]
    name = f'01-annotation-{stage}-{teacher}'
    planroot = root / 'plans' / name
    source = json.loads(Path(protocol['source_task']).read_text())
    # Reuse explicit source/model pins, not any historical result labels.
    pins = dict(source['input_hashes'])
    pins.update(protocol['input_hashes'])
    module = project/'learned_score/mlsys_sprint/evidence_annotation_v18.py'
    pins[str(module)] = file_digest(module)
    shared = project/'learned_score/mlsys_sprint/evidence_learning_v18.py'
    pins[str(shared)] = file_digest(shared)
    task = dict(format='mlsys_evidence_annotation_v18_task', cwd=str(project),
        model=protocol['model'], training_manifest=protocol['training_manifest'],
        manifest=protocol['manifest'], split=split, output_root=str(root/stage),
        mode='annotate_extract' if stage=='train' and teacher=='qwen' else 'annotate',
        teacher=teacher, deadline_utc_epoch=protocol['physical_deadline_utc_epoch'],
        input_hashes=pins)
    diagnostic = json.loads((root/'data/teacher-diagnostic-identities.json').read_text())
    if stage=='train':
        task['overlay_row_hashes'] = diagnostic['visual_audit_ids']
    if teacher_model:
        task['teacher_model'] = teacher_model
        task['entries'] = diagnostic['teacher_comparison_ids']
        snapshot = Path(teacher_model['model_path'])
        for path in sorted(snapshot.rglob('*')):
            if path.is_file() and '.cache' not in path.parts:
                pins[str(path.resolve())] = file_digest(path)
    taskpath = planroot/'task.json'
    write_immutable(taskpath, seal(task))
    tasks = []
    for rank in range(world_size):
        marker = ('index' if task['mode']=='annotate_extract' else 'annotations')+f'-rank{rank}.json'
        environment={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1',
                 'HF_MODULES_CACHE':str(project/'.cache/huggingface/modules'),
                 'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
                 'TOKENIZERS_PARALLELISM':'false'}
        if teacher=='internvl':
            environment['PYTHONPATH']=str(project/'.data/teacher-deps-v18')+':'+str(project)
        tasks.append(dict(id=f'evidence-v18-{stage}-{teacher}-{rank}',resource='gpu',
            model_tag='qwen25' if teacher=='qwen' else 'internvl35', tag='evidence-v18',
            argv=[str(project/'.conda/asymvlm/bin/python'),'-B','-u','-m',
                  'learned_score.mlsys_sprint.evidence_annotation_v18','--task',str(taskpath),
                  '--rank',str(rank),'--world-size',str(world_size)],
            env=environment,
            depends_on=[],minimum_start_seconds=180,
            artifacts=[dict(path=str(root/stage/'annotation'/teacher/marker))]))
    plan = seal(dict(format='mlsys_evidence_v18_worker_plan',cwd=str(project),tasks=tasks,
        input_hashes={**pins,str(taskpath):file_digest(taskpath)},unexpected_retries=1,
        deadline_utc_epoch=protocol['physical_deadline_utc_epoch'],
        evidence='reused_development_not_fresh_confirmation'))
    write_immutable(planroot/'plan.json',plan)
    return str(planroot/'plan.json')


def evaluation_plan(protocol, name, split, cells, world_size=8, request_ids=None,
                    feature_checkpoints=(), extra=None):
    project,root=Path(protocol['cwd']),Path(protocol['output_root'])
    manifest=load_manifest(protocol['manifest'])
    request_ids=request_ids or [r['row_hash'] for r in manifest['splits'][split]]
    source=json.loads(Path(protocol['source_task']).read_text())
    pins={**source['input_hashes'],**protocol['input_hashes']}
    for filename in ('evidence_evaluate_v18.py','evidence_learning_v18.py'):
        path=project/'learned_score/mlsys_sprint'/filename
        pins[str(path)]=file_digest(path)
    checkpoints=set(feature_checkpoints)|{c['checkpoint'] for c in cells if c.get('selector')=='scorer'}
    for checkpoint in checkpoints:
        for path in (Path(checkpoint),Path(checkpoint).parent/'complete.json'):
            pins[str(path.resolve())]=file_digest(path)
    task=dict(format='mlsys_evidence_evaluate_v18_task',cwd=str(project),
        source_task=protocol['source_task'],manifest=protocol['manifest'],split=split,
        request_ids=request_ids,output_root=str(root/'physical'/name),
        deadline_utc_epoch=protocol['physical_deadline_utc_epoch'],warmup=1,cells=cells,
        feature_checkpoints=list(feature_checkpoints),input_hashes=pins,**(extra or {}))
    planroot=root/'plans'/name
    taskpath=planroot/'task.json'
    write_immutable(taskpath,seal(task))
    tasks=[]
    for rank in range(world_size):
        tasks.append(dict(id=f'evidence-v18-{name}-{rank}',resource='gpu',tag='evidence-v18',model_tag='qwen25',
            argv=[str(project/'.conda/asymvlm/bin/python'),'-B','-u','-m',
                  'learned_score.mlsys_sprint.evidence_evaluate_v18','--task',str(taskpath),
                  '--rank',str(rank),'--world-size',str(world_size)],
            env={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1',
                 'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},
            depends_on=[],minimum_start_seconds=180,
            artifacts=[dict(path=str(Path(task['output_root'])/f'rank{rank}.complete.json'),passed=True)]))
    plan=seal(dict(format='mlsys_evidence_v18_worker_plan',cwd=str(project),tasks=tasks,
        input_hashes={**pins,str(taskpath):file_digest(taskpath)},unexpected_retries=1,
        deadline_utc_epoch=protocol['physical_deadline_utc_epoch']))
    write_immutable(planroot/'plan.json',plan)
    return str(planroot/'plan.json')


def repaired_annotation_plan(protocol,name,source_name,output_stage,world_size=8,
                             entries=None,mode=None,module_name='evidence_annotation_v18a',
                             prompt_version='evidence_regions_simple_v18a'):
    """Additive repair publication: the old raw responses and tasks stay fixed."""
    project,root=Path(protocol['cwd']),Path(protocol['output_root'])
    old_task=json.loads((root/'plans'/source_name/'task.json').read_text())
    old_plan=json.loads((root/'plans'/source_name/'plan.json').read_text())
    task={k:v for k,v in old_task.items() if k!='sha256'}
    task.update(output_root=str(root/output_stage),prompt_version=prompt_version)
    if entries is not None: task['entries']=entries
    if mode is not None: task['mode']=mode
    module=project/'learned_score/mlsys_sprint'/(module_name+'.py')
    task['input_hashes'][str(module)]=file_digest(module)
    # A single-row native-gradient canary does not pretend to cover the20imageaudit.
    if entries is not None:
        task['overlay_row_hashes']=[r for r in task.get('overlay_row_hashes',[]) if r in entries]
    taskpath=root/'plans'/name/'task.json'
    write_immutable(taskpath,seal(task))
    tasks=[]
    for rank in range(world_size):
        template=old_plan['tasks'][0]
        marker=('annotations' if task['mode']=='annotate' else 'index')+f'-rank{rank}.json'
        tasks.append({**template,'id':f'evidence-v18-{name}-{rank}',
            'argv':[str(project/'.conda/asymvlm/bin/python'),'-B','-u','-m',
                    'learned_score.mlsys_sprint.'+module_name,'--task',str(taskpath),
                    '--rank',str(rank),'--world-size',str(world_size)],
            'artifacts':[dict(path=str(root/output_stage/'annotation'/task['teacher']/marker))]})
    plan=seal(dict(format='mlsys_evidence_v18_worker_plan',cwd=str(project),tasks=tasks,
        input_hashes={**task['input_hashes'],str(taskpath):file_digest(taskpath)},unexpected_retries=1,
        deadline_utc_epoch=protocol['physical_deadline_utc_epoch'],
        repair_of_plan_sha256=old_plan['sha256']))
    write_immutable(taskpath.parent/'plan.json',plan)
    return str(taskpath.parent/'plan.json')


def prepare_fit(protocol, teacher='qwen', world_size=8, annotation_root=None, source_task=None):
    """Publish both matched arms only after the explicit annotation audit pass."""
    project,root=Path(protocol['cwd']),Path(protocol['output_root'])
    annotation=Path(annotation_root) if annotation_root else root/'train/annotation'/teacher
    if not annotation.resolve().is_relative_to(root.resolve()):
        raise ValueError('annotation cache outside pilot')
    auditpath=annotation/'visual-audit.json'
    audit=json.loads(auditpath.read_text())
    expected=json.loads((root/'data/teacher-diagnostic-identities.json').read_text())['visual_audit_ids']
    if (set(audit['rows'])!=set(expected) or len(audit['rows'])!=20
            or sum(bool(r['usable']) for r in audit['rows'].values())<16):
        raise ValueError('teacher has not passed the predeclared visual annotation audit')
    for row_hash,decision in audit['rows'].items():
        annotation_path=annotation/'rows'/(row_hash+'.json')
        record=json.loads(annotation_path.read_text())
        if (decision.get('annotation_sha256')!=file_digest(annotation_path)
                or type(decision.get('usable')) is not bool
                or not decision.get('reason')
                or decision['usable'] and record['annotation']['status']!='ok'):
            raise ValueError('visual audit is not bound to the actual usable annotation')
    indices=[json.loads((annotation/f'index-rank{rank}.json').read_text()) for rank in range(world_size)]
    if any(index.get('completed') is not True or index.get('world_size')!=world_size
           or index.get('teacher')!=teacher for index in indices):
        raise ValueError('incomplete or inconsistent training annotation shards')
    source_task=Path(source_task or root/'plans'/f'01-annotation-train-{teacher}'/'task.json')
    source_hash=file_digest(source_task)
    if any(index.get('task_sha256')!=source_hash or index.get('rank')!=rank
           for rank,index in enumerate(indices)):
        raise ValueError('training shard does not match its pinned task/rank')
    entries=[e for index in indices for e in index['entries']]
    expected_train={e['row_hash'] for e in load_manifest(protocol['manifest'])['splits']['scorer_train']}
    if len(entries)!=100 or {e['row_hash'] for e in entries}!=expected_train:
        raise ValueError('training needs complete distinct100row annotation/cache panel')
    rejected=[]
    for entry in entries:
        decision=audit['rows'].get(entry['row_hash'])
        if decision and not decision['usable']:
            rejected.append(entry['row_hash'])
            entry['annotation_status']='rejected_visual_audit'
    index=seal(dict(format='mlsys_evidence_cache_index_v18',completed=True,entries=entries,
        teacher=teacher,visual_audit_sha256=file_digest(auditpath),
        rejected_by_visual_audit=rejected,source_task_sha256=source_hash,
        shards={str(annotation/f'index-rank{r}.json'):file_digest(annotation/f'index-rank{r}.json')
                for r in range(world_size)}))
    indexpath=annotation/'training-index.json'
    write_immutable(indexpath,index)
    paths=[]
    for arm in ('control','evidence'):
        path=root/'fits'/arm/'task.json'
        module=project/'learned_score/mlsys_sprint/evidence_learning_v18.py'
        checkpoint=Path(protocol['model']['checkpoint'])
        pins={str(p.resolve()):file_digest(p) for p in (indexpath,module,checkpoint,checkpoint.parent/'complete.json',auditpath)}
        task=seal(dict(format='mlsys_evidence_learning_v18_task',cwd=str(project),arm=arm,
            cache_index=str(indexpath),reference_checkpoint=str(checkpoint),output_root=str(path.parent),
            deadline_utc_epoch=protocol['stop_new_work_utc_epoch'],input_hashes=pins))
        write_immutable(path,task); paths.append(str(path))
    return paths


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','annotation'])
    parser.add_argument('--cwd',default='.')
    parser.add_argument('--split',default='scorer_train')
    parser.add_argument('--world-size',type=int,default=8)
    args=parser.parse_args()
    protocol=prepare(args.cwd)
    if args.action=='annotation':
        print(annotation_plan(protocol,args.split,args.world_size))
    else:
        print(json.dumps({k:protocol[k] for k in ('output_root','created_utc_epoch','deadline_utc_epoch')}))


if __name__=='__main__':
    main()
