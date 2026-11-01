"""Phi transfer study: native parity, physical knots, and fit-only coverage rules.

All historical sources are read-only. This is post-hoc cross-model transfer, not
fresh confirmation. The frozen rank32 scorer and original-position crop adapter
are unchanged. Generated text is not evicted in this short-answer experiment.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import socket
import time
from types import SimpleNamespace

from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import digest, file_digest

FORMAT = 'mlsys_phi_transfer_v15'
ROOT = Path('reproduction_results/mlsys_compression_20261023_v15/third-model')
KEEP = (30, 50, 70, 90, 100)
PROFILES = {'original': (1., 2.), 'conservative': (.5, 1.), 'extra_conservative': (.25, .5)}
SPLITS = ('fit', 'screening', 'evaluation')
EVIDENCE = 'post_selection_development_transfer_not_independent_confirmation'
MODEL = 'phi35'


def _read(path):
    return json.loads(Path(path).read_text())


def family(name):
    from .adaptive_data_v3 import dataset_family
    return dataset_family(name)


def transfer_subset(rows, maximum_groups=256):
    """SHA selection by complete image group; MME balanced across categories."""
    from .compression_data_v9 import identity_only
    identity_only(rows)
    if type(maximum_groups) is not int or maximum_groups < 14:
        raise ValueError('at least fourteen groups required to preserve MME categories')
    groups = defaultdict(list)
    for row in rows:
        groups[(family(row['dataset']), row['group_id'])].append(row)
    chosen = []
    for domain in sorted({key[0] for key in groups}):
        strata = defaultdict(list)
        for (dataset, group), members in groups.items():
            if dataset != domain:
                continue
            category = members[0].get('category', '') if domain == 'MME' else ''
            if domain == 'MME' and (len(members) != 2 or not category or
                    len({r.get('category') for r in members}) != 1):
                raise ValueError('MME requires intact same-category image pairs')
            strata[category].append((group, members))
        for category in strata:
            strata[category].sort(key=lambda x: digest(['phi-v15-transfer', domain, category, x[0]]))
        selected, offset = [], 0
        while len(selected) < maximum_groups:
            batch = [strata[category][offset] for category in sorted(strata)
                     if offset < len(strata[category])]
            if not batch:
                break
            selected.extend(batch[:maximum_groups-len(selected)])
            offset += 1
        chosen.extend(copy.deepcopy(row) for _, members in selected for row in members)
    return sorted(chosen, key=lambda r: (r['dataset'], r['row_hash']))


def validate_disjoint(splits):
    from .compression_data_v9 import qualify_keys
    keysets = {}
    for split, rows in splits.items():
        keysets[split] = set().union(*(qualify_keys(r) for r in rows)) if rows else set()
        if len({r['row_hash'] for r in rows}) != len(rows):
            raise ValueError('duplicate request within ' + split)
    names = list(keysets)
    for i, first in enumerate(names):
        for second in names[i+1:]:
            if keysets[first] & keysets[second]:
                raise ValueError('image/query identity overlap: ' + first + '/' + second)


def source_pins():
    from . import adaptive_experiment_v4 as runtime
    from . import compression_features_v11 as features
    from . import compression_policy_v12 as policy
    base = Path(__file__).resolve().parents[1]
    result = {str(base / name): sha for name, sha in runtime.source_hashes('phi3_vision').items()}
    result.update(features.source_pins('phi3_vision'))
    for module in (Path(__file__), Path(policy.__file__), Path(policy.previous.__file__)):
        result[str(module.resolve())] = file_digest(module)
    return result


def verify(study, *, pins=True):
    if (study.get('format') != FORMAT or study.get('sha256') != digest({k:v for k,v in study.items() if k != 'sha256'})
            or study.get('model_tag') != MODEL or study.get('retentions') != list(KEEP)
            or study.get('profiles') != {k:list(v) for k,v in PROFILES.items()}
            or study.get('site') != 'clusterb' or study.get('independent_confirmation') is not False):
        raise ValueError('third-model study identity changed')
    if pins:
        for path, sha in study['input_hashes'].items():
            if file_digest(path) != sha:
                raise ValueError('third-model frozen input changed: ' + path)
    return study


def prepare(project_root, output_root=None, *, hours=16, world_size=8):
    from learned_score.mlsys_eval.common import load_manifest, seal_manifest
    from .adaptive_experiment_v3 import contract_from_config
    project = Path(project_root).resolve()
    root = Path(output_root or project/ROOT).resolve()
    if not root.is_relative_to(project/'reproduction_results'):
        raise ValueError('output must stay in the reproduction directory')
    if (root/'study.json').exists():
        return verify(checked(root/'study.json'))
    if not 0 < hours <= 24 or type(world_size) is not int or not 1 <= world_size <= 32:
        raise ValueError('invalid bounded execution window or shard count')
    old_path = project/'reproduction_results/mlsys_adaptive_20261016_v3/study.json'
    source = checked(old_path)
    spec = copy.deepcopy(source['models'][MODEL])
    original = Path(source['manifest'])
    old = load_manifest(original)
    panel_path = project/'reproduction_results/mlsys_compression_20261022_v14/full-suite/manifest.json'
    panel = load_manifest(panel_path)
    if (panel['metadata'].get('known_training_calibration_excluded') is not True or
            panel['metadata'].get('independent_confirmation') is not False):
        raise ValueError('requires the actual approved post-hoc transfer panel')
    splits = {name:copy.deepcopy(old['splits'][name]) for name in ('fit', 'screening', 'phi_train')}
    splits['evaluation'] = transfer_subset(panel['splits']['evaluation'])
    validate_disjoint(splits)
    manifest = seal_manifest({'format':'mlsys_adaptive_data_v3', 'dataset':'ASYMVLM_PHI_V15',
        'metadata': {'selection_uses_labels':False, 'independent_confirmation':False,
            'evidence':EVIDENCE, 'source_panel':str(panel_path), 'source_fit':str(original),
            'scorer_training':False, 'maximum_transfer_groups_per_domain':256}, 'splits':splits})
    manifest_path = root/'data/manifest.json'
    write_immutable(manifest_path, manifest)
    ids = [min((r for r in splits['fit'] if family(r['dataset']) == domain),
               key=lambda r:r['row_hash'])['row_hash'] for domain in ('DocVQA', 'TextVQA')]
    write_immutable(root/'parity/request-ids.json', {'row_hashes':ids})
    config_path = Path(spec['model_path'])/'config.json'
    contract = contract_from_config(SimpleNamespace(**_read(config_path)))
    pins = source_pins()
    for path in (old_path, original, panel_path, manifest_path, root/'parity/request-ids.json',
                 Path(spec['checkpoint']), Path(spec['checkpoint']).parent/'complete.json', config_path):
        pins[str(path)] = file_digest(path)
    for path in Path(spec['model_path']).glob('*.py'):
        pins[str(path)] = file_digest(path)
    for name in ('model.safetensors.index.json', 'generation_config.json', 'preprocessor_config.json', 'tokenizer_config.json'):
        path = Path(spec['model_path'])/name
        if path.exists():
            pins[str(path)] = file_digest(path)
    now = time.time()
    study = seal({'format':FORMAT, 'cwd':str(project), 'output_root':str(root),
        'model_tag':MODEL, 'model':spec, 'manifest':str(manifest_path),
        'manifest_sha256':file_digest(manifest_path), 'source_panel':str(panel_path),
        'source_fit_manifest':str(original), 'input_hashes':pins, 'created_utc_epoch':now,
        'deadline_utc_epoch':now+hours*3600, 'world_size':world_size, 'site':'clusterb',
        'retentions':list(KEEP), 'profiles':{k:list(v) for k,v in PROFILES.items()},
        'cache_contract':contract, 'bytes_per_token':contract['kv_bytes_per_slot'],
        'generation_reserve':contract['reserved_generated_slots'], 'parity_ids':ids,
        'split_rows':{k:len(v) for k,v in splits.items()},
        'transfer_domains':{d:{'rows':sum(family(r['dataset'])==d for r in splits['evaluation']),
            'groups':len({r['group_id'] for r in splits['evaluation'] if family(r['dataset'])==d})}
            for d in sorted({family(r['dataset']) for r in splits['evaluation']})},
        'calibration':'fit-only scalar threshold; screening diagnostics; all profiles retained',
        'scorer_training':False, 'interpolation':False, 'independent_confirmation':False,
        'evidence':EVIDENCE, 'text_policy':'protected_prefix_no_generated_text_eviction',
        'baseline_scope':'DivPrune projected-feature selector adaptation, not official end-to-end implementation'})
    write_immutable(root/'study.json', study)
    return verify(study)


def _site():
    host = socket.gethostname().split('.')[0]
    if not host.startswith('clusterb-gpu-'):
        raise ValueError('Phi v15 features and inference must stay on the pinned ClusterB site')
    return host


def _args(study, split, rank=0):
    spec = study['model']
    return SimpleNamespace(adapter='phi3_vision', model=spec['model_path'], model_revision=spec['revision'],
        checkpoint=Path(spec['checkpoint']), manifest=Path(study['manifest']), split=split,
        training_manifest=Path(study['cwd'])/'reproduction_results/mlsys_20261012/manifest.json',
        device='cuda:0', num_crops=16, max_visual_patches=1280, attention_implementation='eager',
        backbone_dtype='bfloat16', dataset_splits=(split,), rank=rank, world_size=study['world_size'])


def parity_passed(study):
    root = Path(study['output_root'])
    gate_path = root/'parity/gate.json'
    if not gate_path.exists():
        return False
    gate = checked(gate_path)
    if gate.get('study_sha256') != study['sha256'] or gate.get('passed') is not True or gate.get('site') != study['site']:
        raise ValueError('invalid Phi native parity gate')
    for path, sha in gate['input_hashes'].items():
        if file_digest(path) != sha:
            raise ValueError('Phi parity proof changed')
    return True


def validate_parity(proofs, study):
    if {p.get('request_id') for p in proofs} != set(study['parity_ids']):
        raise ValueError('parity request identities differ')
    for proof in proofs:
        expected = {'status':'passed', 'manifest_sha256':study['manifest_sha256'],
            'checkpoint_sha256':file_digest(study['model']['checkpoint']),
            'split':'fit', 'world_size':1, 'max_new_tokens':128, 'retentions_percent':list(KEEP)}
        if any(proof.get(k) != v for k,v in expected.items()):
            raise ValueError('Phi parity provenance differs')
        if {o['retention_percent'] for o in proof['outcomes']} != set(KEEP):
            raise ValueError('missing native parity retention')
        for result in proof['outcomes']:
            p = result['retention_percent']
            if (result.get('native_ids_equal') is not True or
                    (p == 100 and result.get('native_generate_ids_equal') is not True) or
                    result['cache']['evicted_tokens'] or result['cache']['compaction_events']):
                raise ValueError('native/bounded Phi token parity failed')
            if result['selected_visual_tokens'] != max(1, proof['input_visual_tokens']*p//100):
                raise ValueError('parity count differs from declared native protocol')
    return True


def parity_worker(study):
    from . import phi_adaptive_v4 as phi
    verify(study)
    host = _site()
    root, spec = Path(study['output_root']), study['model']
    argv = ['--phase','parity','--manifest',study['manifest'],'--checkpoint',spec['checkpoint'],
        '--output',str(root/'parity/proofs'),'--request-ids',str(root/'parity/request-ids.json'),
        '--split','fit','--model',spec['model_path'],'--model-revision',spec['revision'],
        '--device','cuda:0','--rank','0','--world-size','1','--limit','2',
        '--max-new-tokens','128','--retentions',*[str(p) for p in KEEP],'--resume']
    phi.main(argv)
    paths = [root/'parity/proofs'/(key+'.json') for key in study['parity_ids']]
    validate_parity([_read(p) for p in paths], study)
    write_immutable(root/'parity/gate.json',seal({'format':FORMAT+'_native_parity',
        'study_sha256':study['sha256'],'passed':True,'site':study['site'],'host':host,
        'input_hashes':{str(p):file_digest(p) for p in paths},
        'note':'Native gate uses floor counts; feature/quality knots use ceil counts with exact-index checks.'}))


def _require_gate(study):
    verify(study)
    _site()
    if not parity_passed(study):
        raise ValueError('native Phi parity must pass before any feature/quality work')


def feature_worker(study, split, rank):
    import torch
    from torch.nn import functional as F
    from .adaptive_data_v3 import load_manifest
    from . import adaptive_experiment_v3 as runtime
    from . import phi_adaptive_v4 as phi
    from .compression_features_v11 import AnswerFreeDataset, compute_feature_curves
    from learned_score.mlsys_scorer.backbone import frozen_versions, assert_frozen
    _require_gate(study)
    if split not in SPLITS or not 0 <= rank < study['world_size']:
        raise ValueError('invalid feature partition')
    manifest = load_manifest(study['manifest'])
    args = _args(study, split, rank)
    with phi.installed():
        adapter, _, backbone, scorer = runtime.load_runtime(args, manifest)
    dataset = AnswerFreeDataset(study['manifest'], split, project_root=study['cwd'])
    adapter.dataset = dataset
    contract = runtime.contract_from_config(adapter.model.config)
    if contract != study['cache_contract']:
        raise ValueError('actual Phi KV geometry differs from frozen config')
    versions, scorer_versions = frozen_versions(adapter.model), frozen_versions(scorer)
    directory = Path(study['output_root'])/'features'/split
    metadata = {'format':FORMAT+'_features','study_sha256':study['sha256'],'model_tag':MODEL,
        'split':split,'world_size':study['world_size'],'backbone':backbone,'site':study['site'],
        'hardware':torch.cuda.get_device_name(adapter.device),'cache_contract':contract,
        'checkpoint_sha256':file_digest(study['model']['checkpoint']),'answers_loaded':False}
    write_immutable(directory/f'rank{rank}.metadata.json',metadata)
    entries = sorted(manifest.splits[split],key=lambda r:(r['dataset'],r['row_hash']))[rank::study['world_size']]
    pins = {}
    for entry in entries:
        path = directory/'features'/(entry['row_hash']+'.json')
        if path.exists():
            value = _read(path)
            if value['identity_sha256'] != digest(metadata):
                raise ValueError('feature resume changed identity')
        else:
            with torch.no_grad():
                prepared = adapter.prepare_sample(dataset.row_for_entry(entry),include_answer=False)
                values = runtime.feature_summary(prepared,scorer)
                vision = prepared.inputs_embeds[0,prepared.visual_positions]
                question = prepared.inputs_embeds[0,prepared.semantic_text_positions]
                scores = scorer(vision,question).float()
                projected = F.linear(F.normalize(vision.float(),dim=-1),scorer.visual.weight.float())
                curves = compute_feature_curves(scores,vision,projected,selectors=('scorer',),retentions=KEEP)
                value = {'format':FORMAT+'_feature','identity_sha256':digest(metadata),
                    'request_id':entry['row_hash'],'group_id':entry['group_id'],'dataset':entry['dataset'],
                    'image_sha256':entry['image_sha256'],'split':split,'curves':curves,
                    'checkpoint_sha256':metadata['checkpoint_sha256'],'cache_contract':contract,
                    'answers_loaded':False,'answers_consumed_by_feature_extractor':False,**values}
                del prepared, vision, question, scores, projected
            write_immutable(path,value)
        assert_frozen(adapter.model,versions)
        assert_frozen(scorer,scorer_versions)
        pins[str(path)] = file_digest(path)
    write_immutable(directory/f'rank{rank}.complete.json',seal({'format':FORMAT+'_features_complete',
        'study_sha256':study['sha256'],'identity_sha256':digest(metadata),'requests':len(entries),'input_hashes':pins}))


def collect_features(study, split):
    root = Path(study['output_root'])/'features'/split
    markers = [root/f'rank{r}.complete.json' for r in range(study['world_size'])]
    if not all(p.exists() for p in markers):
        return None
    entries = _read(study['manifest'])['splits'][split]
    values = {}
    for rank,path in enumerate(markers):
        marker = checked(path)
        metadata = _read(root/f'rank{rank}.metadata.json')
        if marker['study_sha256'] != study['sha256'] or marker['identity_sha256'] != digest(metadata):
            raise ValueError('feature completion identity mismatch')
        for filename,sha in marker['input_hashes'].items():
            if file_digest(filename) != sha:
                raise ValueError('frozen feature modified')
            row = _read(filename)
            if row['identity_sha256'] != digest(metadata) or row['request_id'] in values:
                raise ValueError('duplicate or invalid feature identity')
            values[row['request_id']] = row
    if set(values) != {r['row_hash'] for r in entries}:
        raise ValueError('feature coverage differs from frozen manifest')
    return values


def knot_schedule(study, split, features):
    rows = {}
    for key, f in sorted(features.items()):
        n = f['input_visual_tokens']
        cells = [{'key':'all_tokens','selector':'full','selected_count':n}]
        for p in KEEP[:-1]:
            for selector in (('scorer','divprune') if split == 'evaluation' else ('scorer',)):
                cells.append({'key':f'{selector}_uniform_{p:03d}','selector':selector,
                    'selected_count':(n*p+99)//100,'retention_percent':p})
        rows[key] = {'dataset':f['dataset'],'input_visual_tokens':n,
            'input_text_tokens':f['input_text_tokens'],'cells':cells}
    return seal({'format':FORMAT+'_schedule','study_sha256':study['sha256'],'split':split,
        'manifest_sha256':study['manifest_sha256'],'checkpoint_sha256':file_digest(study['model']['checkpoint']),
        'cache_contract':study['cache_contract'],'rows':rows,'counts_selected_without_answers':True})


def uniform_matched(features, counts):
    """Identical aggregate image count per domain, largest-remainder rounding."""
    result = {}
    for domain in sorted({family(f['dataset']) for f in features.values()}):
        keys = sorted(k for k,f in features.items() if family(f['dataset']) == domain)
        goal, total = sum(counts[k] for k in keys), sum(features[k]['input_visual_tokens'] for k in keys)
        low = {k:max(1,features[k]['input_visual_tokens']*goal//total) for k in keys}
        residual = goal-sum(low.values())
        if residual < 0:
            raise ValueError('uniform minimum exceeds exact aggregate target')
        order = sorted(keys,key=lambda k:(-(features[k]['input_visual_tokens']*goal % total),k))
        for key in order:
            if residual and low[key] < features[key]['input_visual_tokens']:
                low[key] += 1
                residual -= 1
        if residual or sum(low.values()) != goal:
            raise ValueError('uniform exact token reconciliation failed')
        result.update(low)
    return result


def _policy_rows(study, split, features, physical=None):
    rows = []
    entries = _read(study['manifest'])['splits'][split]
    for entry in entries:
        key = entry['row_hash']
        f = features[key]
        row = {'row_hash':key,'group_id':entry['group_id'],'dataset':family(entry['dataset']),
            'category':entry.get('category',''),'quality_metric':entry['metric'],
            'input_visual_tokens':f['input_visual_tokens'],'input_text_tokens':f['input_text_tokens'],
            'curves':f['curves']}
        if physical is not None:
            row['quality_by_count'] = {}
            for value in physical.values():
                if value['request_id'] == key and value['selector'] in ('scorer','full'):
                    row['quality_by_count'][str(value['selected_count'])] = value['quality_score']
        rows.append(row)
    return rows


def calibrate(study, fit, screening, fit_physical, screen_physical):
    from . import compression_policy_v12 as policy
    fit_rows = _policy_rows(study,'fit',fit,fit_physical)
    screen_rows = _policy_rows(study,'screening',screening,screen_physical)
    rules, diagnostics = {}, {}
    for profile,(mean,worst) in PROFILES.items():
        rule = policy.fit_rule(fit_rows,'rank32_coverage',model=MODEL,
            bytes_per_token=study['bytes_per_token'],generation_reserve=study['generation_reserve'],
            max_mean_loss_pp=mean,max_domain_loss_pp=worst)
        assigned = policy.infer(rule,policy.answer_free(screen_rows))
        rules[profile] = rule
        diagnostics[profile] = {'summary':policy.summary(screen_rows,assigned['counts'],
            bytes_per_token=study['bytes_per_token'],generation_reserve=study['generation_reserve']),
            'allocation':assigned,'screening_does_not_change_rule':True}
    return seal({'format':FORMAT+'_calibration','study_sha256':study['sha256'],'rules':rules,
        'screening':diagnostics,'rule_fit_splits':['fit'],'evaluation_labels_used':False,
        'screening_labels_used_to_change_threshold':False,'all_profiles_retained':True,
        'one_scalar':'query-weighted uncovered mass in frozen rank32 geometry',
        'interpolation':False,'independent_confirmation':False})


def adaptive_schedule(study, features, calibration):
    from . import compression_policy_v12 as policy
    if calibration['study_sha256'] != study['sha256']:
        raise ValueError('calibration belongs to another study')
    inputs = policy.answer_free(_policy_rows(study,'evaluation',features))
    cells = {k:[] for k in features}
    allocations = {}
    for profile,rule in calibration['rules'].items():
        assigned = policy.infer(rule,inputs)
        uniform = uniform_matched(features,assigned['counts'])
        allocations[profile] = {'adaptive':assigned,'uniform_counts':uniform}
        for key in cells:
            for method,selector,count in (('scorer_adaptive','scorer',assigned['counts'][key]),
                    ('scorer_uniform','scorer',uniform[key]),('divprune_uniform','divprune',uniform[key])):
                cells[key].append({'key':method+'__'+profile,'selector':selector,'selected_count':count,
                    'profile':profile,'policy_sha256':rule['sha256']})
    schedule = knot_schedule(study,'evaluation',features)
    schedule.pop('sha256')
    for key in cells:
        schedule['rows'][key]['cells'] = cells[key]
    schedule.update(calibration_sha256=calibration['sha256'],allocations=allocations,
        exact_aggregate_tokens_matched_per_domain=True)
    return seal(schedule)


def physical_worker(study, stage, rank):
    from . import adaptive_experiment_v4 as runtime
    _require_gate(study)
    split = 'evaluation' if stage in ('evaluation','adaptive') else stage
    root = Path(study['output_root'])
    schedule_path = root/'schedules'/(stage+'.json')
    schedule = checked(schedule_path)
    if schedule['study_sha256'] != study['sha256']:
        raise ValueError('physical schedule belongs to another study')
    spec = study['model']
    original = runtime.selector_for
    def selector_for(cell, scorer, feature, observed=None):
        select, internal = original(cell,scorer,feature,observed)
        def checked_select(prepared):
            selected = select(prepared)
            if cell['selector'] == 'scorer':
                saved = _read(root/'features'/split/'features'/(feature['request_id']+'.json'))
                expected = saved['curves']['scorer'].get('selected_indices_sha256',{}).get(str(cell['selected_count']))
                if expected is not None and digest(selected.detach().cpu().tolist()) != expected:
                    raise ValueError('physical learned prefix differs from frozen coverage prefix')
            return selected
        return checked_select,internal
    runtime.selector_for = selector_for
    try:
        runtime.main(['--phase','physical','--split',split,'--manifest',study['manifest'],
            '--checkpoint',spec['checkpoint'],'--output',str(root/'physical'/stage),
            '--schedule',str(schedule_path),'--feature-root',str(root/'features'/split),
            '--adapter','phi3_vision','--model',spec['model_path'],'--model-revision',spec['revision'],
            '--num-crops','16','--device','cuda:0','--rank',str(rank),'--world-size',str(study['world_size']),
            '--warmup','1','--resume'])
    finally:
        runtime.selector_for = original


def collect_physical(study, stage):
    root = Path(study['output_root'])
    schedule_path = root/'schedules'/(stage+'.json')
    directory = root/'physical'/stage
    if not schedule_path.exists() or not all((directory/f'rank{r}.complete.json').exists() for r in range(study['world_size'])):
        return None
    schedule = checked(schedule_path)
    values = {}
    metadata = [_read(directory/f'rank{r}.metadata.json') for r in range(study['world_size'])]
    identities = {digest(m) for m in metadata}
    for rank,m in enumerate(metadata):
        if (m['manifest_sha256'] != study['manifest_sha256'] or m['cache_contract'] != study['cache_contract']
                or m['schedule_sha256'] != file_digest(schedule_path)
                or _read(directory/f'rank{rank}.complete.json')['identity_sha256'] != digest(m)):
            raise ValueError('physical metadata/marker changed')
    for key,row in schedule['rows'].items():
        for cell in row['cells']:
            path = directory/'rows'/f"{key}.{cell['key']}.json"
            v = _read(path)
            expected = (row['input_text_tokens']+cell['selected_count']+study['generation_reserve'])*study['cache_contract']['total_owned_bytes_per_slot']
            if (v.get('status') != 'ok' or v.get('identity_sha256') not in identities
                    or v['schedule_content_sha256'] != schedule['sha256']
                    or v['selected_count'] != cell['selected_count'] or v['expected_cache_bytes'] != expected
                    or v['metrics']['cache']['total_owned_bytes'] != expected):
                raise ValueError('physical result differs from frozen count/cache contract')
            v.update(physical_source_path=str(path),physical_source_sha256=file_digest(path))
            values[(key,cell['key'])] = v
    return values


def plan_names():
    return ['parity-phi35']+[f'features-{s}-phi35' for s in SPLITS]+[f'physical-{s}-phi35' for s in (*SPLITS,'adaptive')]


def _publish(study, name, phase, *, split=None, stage=None, pins=None):
    from .adaptive_workflow_v4 import environment
    from .worker import read_plan
    root = Path(study['output_root'])
    tasks = []
    for rank in range(1 if phase == 'parity' else study['world_size']):
        argv = [str(Path(study['cwd'])/'.conda/asymvlm/bin/python'),'-B','-u','-m',
            'learned_score.mlsys_sprint.compression_third_model_v15',phase,'--study',str(root/'study.json')]
        if split:
            argv += ['--split',split]
        if stage:
            argv += ['--stage',stage]
        if phase != 'parity':
            argv += ['--rank',str(rank)]
        artifact = (root/'parity/gate.json' if phase == 'parity' else
            root/'features'/split/f'rank{rank}.complete.json' if phase == 'features' else
            root/'physical'/stage/f'rank{rank}.complete.json')
        tasks.append({'id':f'compression-v15-{name}-{rank}','tag':'compression-v15-gpu',
            'resource':'gpu','model_tag':MODEL,'depends_on':[],'minimum_start_seconds':300,
            'argv':argv,'env':{**environment(Path(study['cwd'])),'PYTHONDONTWRITEBYTECODE':'1'},
            'artifacts':[{'path':str(artifact),**({'passed':True} if phase == 'parity' else {})}]})
    all_pins = {**study['input_hashes'],str(root/'study.json'):file_digest(root/'study.json'),**(pins or {})}
    plan = seal({'format':FORMAT+'_plan','cwd':study['cwd'],'tasks':tasks,'input_hashes':all_pins,
        'unexpected_retries':1,'created_utc_epoch':study['created_utc_epoch'],'deadline_utc_epoch':study['deadline_utc_epoch']})
    path = root/'plans'/name/'plan.json'
    write_immutable(path,plan)
    read_plan(path)
    return str(path)


def advance(study):
    verify(study)
    root = Path(study['output_root'])
    plans = [_publish(study,'parity-phi35','parity')]
    if not parity_passed(study):
        return {'plans':plans,'waiting':'native_phi_parity','complete':False}
    features, physical = {}, {}
    gate_pin = {str(root/'parity/gate.json'):file_digest(root/'parity/gate.json')}
    for split in SPLITS:
        plans.append(_publish(study,f'features-{split}-phi35','features',split=split,pins=gate_pin))
        features[split] = collect_features(study,split)
        if features[split] is None:
            continue
        schedule_path = root/'schedules'/(split+'.json')
        write_immutable(schedule_path,knot_schedule(study,split,features[split]))
        pins = {**gate_pin,str(schedule_path):file_digest(schedule_path)}
        for rank in range(study['world_size']):
            p = root/'features'/split/f'rank{rank}.complete.json'
            pins[str(p)] = file_digest(p)
        plans.append(_publish(study,f'physical-{split}-phi35','physical',stage=split,pins=pins))
        physical[split] = collect_physical(study,split)
    if all(physical.get(s) is not None for s in ('fit','screening')):
        calpath = root/'calibration.json'
        if not calpath.exists():
            write_immutable(calpath,calibrate(study,features['fit'],features['screening'],physical['fit'],physical['screening']))
        calibration = checked(calpath)
        if features.get('evaluation') is not None:
            schedule_path = root/'schedules/adaptive.json'
            write_immutable(schedule_path,adaptive_schedule(study,features['evaluation'],calibration))
            pins = {**gate_pin,str(schedule_path):file_digest(schedule_path),str(calpath):file_digest(calpath)}
            plans.append(_publish(study,'physical-adaptive-phi35','physical',stage='adaptive',pins=pins))
    report = collect_report(study)
    return {'plans':plans,'complete':report is not None,'waiting':None if report else 'physical_generation_or_calibration'}


def collect_report(study):
    from .adaptive_report_v3 import paired_ci, quality_summary
    from .compression_report_v8 import memory_summary
    static, adaptive = collect_physical(study,'evaluation'), collect_physical(study,'adaptive')
    if static is None or adaptive is None:
        return None
    allrows = {**static,**adaptive}
    domains, contrasts = [], []
    for domain in sorted(study['transfer_domains']):
        ids = sorted(k for k,m in static if m=='all_tokens' and family(static[(k,m)]['dataset']) == domain)
        methods = sorted({method for key,method in allrows if key in ids})
        reference = [allrows[(k,'all_tokens')] for k in ids]
        metric = reference[0]['quality_metric']
        for method in methods:
            values = [allrows[(k,method)] for k in ids]
            quality = quality_summary(values,metric)
            memory = memory_summary(values,reference)
            domains.append({'model':MODEL,'dataset':domain,'method':method,'requests':len(ids),
                'groups':len({r['group_id'] for r in reference}),'score_percent':quality['score_percent'],
                'quality':quality,'memory':memory,'mean_allocated_kv_mib':memory['mean_allocated_kv_bytes']/2**20,
                'kv_reduction_percent':memory['aggregate_allocated_kv_reduction_percent'],
                'percent_requests_with_memory_reduction':100*memory['fraction_compressed'],
                'versus_all_tokens':paired_ci(values,reference,metric,repetitions=2000)})
        for profile in PROFILES:
            candidate = [allrows[(k,'scorer_adaptive__'+profile)] for k in ids]
            for baseline in ('scorer_uniform','divprune_uniform'):
                control = [allrows[(k,baseline+'__'+profile)] for k in ids]
                if sum(r['expected_cache_bytes'] for r in candidate) != sum(r['expected_cache_bytes'] for r in control):
                    raise ValueError('adaptive/control aggregate allocated cache differs')
                contrasts.append({'model':MODEL,'dataset':domain,'profile':profile,'reference':baseline,
                    **paired_ci(candidate,control,metric,repetitions=2000)})
    value = seal({'format':FORMAT+'_report','study_sha256':study['sha256'],'rows':domains,
        'paired_equal_budget_contrasts':contrasts,'calibration_sha256':checked(Path(study['output_root'])/'calibration.json')['sha256'],
        'evidence':EVIDENCE,'independent_confirmation':False,'all_declared_cohorts_complete':True,
        'physical_sources':{v['physical_source_path']:v['physical_source_sha256'] for v in allrows.values()},
        'text_policy':study['text_policy'],'baseline_scope':study['baseline_scope'],
        'memory_definition':'allocated KV tensor capacity, not total GPU memory',
        'interpolation_used':False,'transfer_labels_used_for_calibration':False})
    write_immutable(Path(study['output_root'])/'report.json',value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','advance','parity','features','physical','report'))
    parser.add_argument('--study',type=Path)
    parser.add_argument('--project-root',type=Path,default=Path.cwd())
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--hours',type=float,default=16)
    parser.add_argument('--world-size',type=int,default=8)
    parser.add_argument('--split',choices=SPLITS)
    parser.add_argument('--stage',choices=(*SPLITS,'adaptive'))
    parser.add_argument('--rank',type=int,default=0)
    args = parser.parse_args(argv)
    if args.phase == 'prepare':
        study = prepare(args.project_root,args.output_root,hours=args.hours,world_size=args.world_size)
        print(json.dumps({'study':str(Path(study['output_root'])/'study.json'),'sha256':study['sha256'],
            'split_rows':study['split_rows'],'transfer_domains':study['transfer_domains']}),flush=True)
        return
    if args.study is None:
        parser.error('--study required')
    study = checked(args.study)
    if args.phase == 'parity':
        parity_worker(study)
    elif args.phase == 'features':
        feature_worker(study,args.split,args.rank)
    elif args.phase == 'physical':
        physical_worker(study,args.stage,args.rank)
    elif args.phase == 'advance':
        print(json.dumps(advance(study)),flush=True)
    else:
        print(json.dumps({'complete':collect_report(study) is not None}),flush=True)


if __name__ == '__main__':
    main()
