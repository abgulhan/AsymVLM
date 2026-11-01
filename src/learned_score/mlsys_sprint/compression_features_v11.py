"""Answer-free selector-native stopping curves, with physical-prefix audits.

Only image/question columns are loaded. Curves are proxies for representation
loss, never claimed probabilities of answer error. All physical results remain
development evidence. Historical checkpoints and runtimes are read-only.
"""
import argparse
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

from . import compression_execution_v11 as execution
from . import compression_features_v10 as old_features
from . import adaptive_experiment_v3 as runtime
from .followup_common import write_immutable
from .state import digest, file_digest

FORMAT = 'mlsys_selector_stopping_curves_v11'
RETENTIONS = tuple(range(10, 101, 10))
SELECTORS = ('scorer', 'divprune', 'cdpruner')
FORBIDDEN = old_features.FORBIDDEN
AnswerFreeDataset = old_features.AnswerFreeDataset
CONTRACT = {
    'retention_percentages': RETENTIONS,
    'missing_mass': 'sum of fixed nonnegative normalized weights outside selected prefix',
    'uncovered_mass': 'sum_i weight_i * (1 - max_selected_j max(0,cos(z_i,z_j)))',
    'scorer_weights': 'softmax of population-zscore learned scores; not calibrated probabilities',
    'scorer_geometry': 'normalized frozen rank32 visual projection',
    'divprune_weights': 'uniform; no learned scorer dependency in deployed selector/budget',
    'divprune_geometry': 'native projected visual embeddings used by physical DivPrune',
    'cdpruner_weights': 'uniform',
    'cdpruner_geometry': 'native projected visual embeddings with independent reference CLIP relevance',
    'baseline_scope': 'projected_feature_adaptations_not_official_pipeline',
    'selector_implementation': 'frozen_v5_native_selector; no repaired order paired with historical labels',
    'quality_monotonicity_assumed': False,
}


def validate_selectors(selectors, model=None):
    selectors = tuple(selectors)
    if not selectors or len(selectors) != len(set(selectors)) or any(s not in SELECTORS for s in selectors):
        raise ValueError('declare unique supported selectors')
    if model not in (None, 'llava_next') and 'cdpruner' in selectors:
        raise ValueError('reference-CLIP CDPruner is only audited on LLaVA-NeXT')
    return tuple(s for s in SELECTORS if s in selectors)


def source_pins(adapter):
    from . import compression_stopping_v11 as stopping
    from .selector_experiment_v5 import source_hashes
    root = Path(__file__).resolve().parents[1]
    files = (Path(__file__), Path(execution.__file__), Path(old_features.__file__), Path(stopping.__file__))
    return {**{str(root/name): sha for name, sha in source_hashes(adapter).items()},
            **{str(path.resolve()): file_digest(path) for path in files}}


def _fallback(n, selector, reason):
    from .compression_stopping_v11 import full_fallback_curve
    curve = full_fallback_curve(n,reason,selector=selector,
        geometry=CONTRACT.get(selector+'_geometry','unspecified'))
    curve.update(valid=False,physical_prefix_parity=False,
        selected_indices_sha256={str(n):digest(list(range(n)))})
    return curve


def compute_feature_curves(scores, visual_features, projected_visual_features, *,
                           selectors=('scorer', 'divprune'), cd_relevance=None, retentions=RETENTIONS):
    """Pure tensor calculation; no data/answer access and no model mutation."""
    import torch
    from . import compression_stopping_v11 as stopping
    from .adaptive_benchmark_v1 import select_exact_count
    from learned_score.llava_selectors import divprune_order
    from .stable_cdpruner import stable_cdpruner_order
    selectors = validate_selectors(selectors)
    if not retentions or any(type(p) is not int or not 1 <= p <= 100 for p in retentions):
        raise ValueError('integer percentages required')
    if visual_features.ndim != 2 or len(visual_features) == 0:
        raise ValueError('nonempty native visual geometry required')
    n = len(visual_features)
    counts = sorted({(n*p+99)//100 for p in retentions} | {n})
    nonfull = [k for k in counts if k<n]
    result = {}
    for selector in selectors:
        geometry = projected_visual_features if selector == 'scorer' else visual_features
        name = 'asymvlm' if selector == 'scorer' else selector
        try:
            if selector == 'cdpruner' and cd_relevance is None:
                raise ValueError('independent reference CLIP relevance is missing')
            if not nonfull:
                order = torch.empty(0,dtype=torch.long,device=visual_features.device)
            elif selector == 'divprune':
                # Preserve exact old precision/tie behavior, not just the same
                # mathematical max-min rule. Repaired v11 order is a distinct
                # candidate and cannot silently consume old physical labels.
                order = divprune_order(visual_features,max(nonfull))
            else:
                order = stopping.selector_order(name, visual=visual_features, scores=scores,
                    relevance=cd_relevance, keep_count=max(nonfull))
            curve = stopping.stopping_curve(visual=geometry, order=order, counts=counts,
                scores=scores if selector == 'scorer' else None,
                weight_mode='learned_scores' if selector == 'scorer' else 'uniform',
                selector=name, geometry=CONTRACT[selector+'_geometry'])
            if selector == 'divprune' and nonfull:
                # The old physical selector is retained for reproducibility.
                # A repaired unique prefix is never silently paired with its labels.
                physical_order = order
            elif selector == 'cdpruner' and nonfull:
                physical_order = stable_cdpruner_order(visual_features, cd_relevance, max(nonfull))
            else:
                physical_order = order
            hashes = {}
            for count in counts:
                indices = stopping.selected_positions(order,count,input_visual_tokens=n)
                expected = torch.arange(n, device=order.device) if count == n else (
                    select_exact_count(scores, visual_features, count) if selector == 'scorer'
                    else physical_order[:count].sort().values)
                if not torch.equal(indices, expected) or len(indices.unique()) != count:
                    raise ValueError('unique-prefix selection differs from the frozen physical runtime')
                hashes[str(count)] = digest(indices.cpu().tolist())
            curve.update(selector=selector, valid=True, fallback=False, fallback_reason=None,
                physical_prefix_parity=True, selected_indices_sha256=hashes,
                ranking_sha256=digest(order.cpu().tolist()))
            result[selector] = curve
        except (ValueError, RuntimeError) as exc:
            # Invalid inputs/degenerate selector cannot justify pruning. Do not
            # hide process failures such as OOM as scientifically valid fallback.
            if isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower():
                raise
            result[selector] = _fallback(n, selector, f'{type(exc).__name__}: {exc}')
    return result


def summarize(prepared, scorer, *, selectors=('scorer', 'divprune'), cd_relevance=None):
    import torch
    from torch.nn import functional as F
    from learned_score.mlsys_scorer.models import Rank32Scorer
    from .adaptive_utility_v3 import calculate_statistic
    if not isinstance(scorer, Rank32Scorer):
        raise ValueError('features require the unchanged frozen rank32 checkpoint')
    with torch.no_grad():
        vision = prepared.inputs_embeds[0, prepared.visual_positions]
        question = prepared.inputs_embeds[0, prepared.semantic_text_positions]
        scores = scorer(vision, question).float()
        projected = F.linear(F.normalize(vision.float(), dim=-1), scorer.visual.weight.float())
        curves = compute_feature_curves(scores, vision, projected, selectors=selectors, cd_relevance=cd_relevance)
    return {'curves': curves, 'normalized_entropy': calculate_statistic(scores, 'normalized_entropy'),
        'input_visual_tokens': len(vision), 'input_text_tokens': int(prepared.inputs_embeds.shape[1]-len(vision))}


def _forbidden(value):
    if isinstance(value, dict):
        return bool(FORBIDDEN & set(value)) or any(_forbidden(v) for v in value.values())
    return isinstance(value, (tuple, list)) and any(_forbidden(v) for v in value)


def validate_feature(value, entry, metadata, saved):
    if _forbidden(value):
        raise ValueError('feature shard contains answer/quality fields')
    for key, expected in {'format': FORMAT, 'identity_sha256': digest(metadata),
            'request_id': entry['row_hash'], 'group_id': entry['group_id'], 'image_sha256': entry['image_sha256'],
            'dataset': entry['dataset'], 'model_tag': metadata['model_tag'], 'split': metadata['split'],
            'checkpoint_sha256': metadata['checkpoint_sha256'], 'answers_loaded': False,
            'answers_consumed_by_feature_extractor': False, 'contract_sha256': digest(CONTRACT)}.items():
        if value.get(key) != expected:
            raise ValueError('stopping feature identity differs: '+key)
    if any(value.get(k) != saved[k] for k in ('input_visual_tokens', 'input_text_tokens')):
        raise ValueError('native feature geometry changed')
    entropy = value.get('normalized_entropy')
    if type(entropy) not in (int, float) or not math.isfinite(entropy) or not 0 <= entropy <= 1:
        raise ValueError('invalid score entropy')
    if abs(entropy-saved['normalized_entropy']) > 1e-6:
        raise ValueError('scorer entropy differs from frozen features')
    if set(value.get('curves', {})) != set(metadata['selectors']):
        raise ValueError('missing or extra selector curves')
    n = value['input_visual_tokens']
    expected_counts = sorted({(n*p+99)//100 for p in RETENTIONS})
    for selector, curve in value['curves'].items():
        if curve.get('selector') != selector or curve.get('input_visual_tokens') != n:
            raise ValueError('curve geometry or selector mismatch')
        counts = curve.get('counts')
        if counts != ([n] if curve.get('fallback') else expected_counts):
            raise ValueError('curve count grid changed')
        if curve.get('fallback'):
            if curve.get('valid') is not False or not curve.get('fallback_reason') or curve.get('physical_prefix_parity') is not False:
                raise ValueError('invalid full-fallback provenance')
        elif curve.get('valid') is not True or curve.get('physical_prefix_parity') is not True:
            raise ValueError('missing physical prefix audit')
        for name in ('missing_mass', 'uncovered_mass', 'removed_fraction'):
            vals = curve.get(name, [])
            if (len(vals) != len(counts) or any(type(v) not in (int, float) or not math.isfinite(v) or not -1e-7 <= v <= 1+1e-7 for v in vals)
                    or abs(vals[-1]) > 1e-7 or any(b > a+1e-6 for a, b in zip(vals, vals[1:]))):
                raise ValueError('invalid monotone curve values')
        if any(u > m+1e-6 for u,m in zip(curve['uncovered_mass'],curve['missing_mass'])):
            raise ValueError('uncovered mass exceeds missing mass')
        if any(abs(r-(1-k/n)) > 1e-7 for r,k in zip(curve['removed_fraction'], counts)):
            raise ValueError('curve removal count mismatch')
        hashes = curve.get('selected_indices_sha256', {})
        if set(hashes) != {str(c) for c in counts} or any(not isinstance(v,str) or len(v)!=64 or set(v)-set('0123456789abcdef') for v in hashes.values()):
            raise ValueError('missing exact selected-index hashes')


def extract(study, model, split, rank, world_size, *, selectors=('scorer', 'divprune'), resume=False):
    import torch
    from learned_score.mlsys_scorer.backbone import frozen_versions, assert_frozen
    execution.verify(study)
    selectors = validate_selectors(selectors, model)
    if model not in execution.MODELS or world_size != study['world_size'] or not 0 <= rank < world_size:
        raise ValueError('feature shard differs from frozen study')
    if time.time() >= study['deadline_utc_epoch']:
        raise ValueError('feature extraction deadline expired')
    manifest, _, old_metadata, saved = execution.development_features(study, model, split)
    spec, root = study['models'][model], Path(study['output_root'])
    args = SimpleNamespace(adapter=spec['adapter'], model=spec['model_path'], model_revision=spec['revision'],
        checkpoint=Path(spec['checkpoint']), manifest=Path(study['manifest']), split=split,
        training_manifest=Path(study['cwd'])/'reproduction_results/mlsys_20261012/manifest.json',
        device='cuda:0', max_visual_patches=spec.get('max_visual_patches', 1280),
        attention_implementation='sdpa', backbone_dtype='bfloat16', dataset_splits=('scorer_dev',))
    adapter, _, backbone, scorer = runtime.load_runtime(args, manifest)
    dataset = AnswerFreeDataset(study['manifest'], split, project_root=study['cwd'])
    adapter.dataset = dataset
    versions, scorer_versions = frozen_versions(adapter.model), frozen_versions(scorer)
    clip = None
    if 'cdpruner' in selectors:
        from .cd_clip_reference import ReferenceClipRelevance, configuration
        clip = ReferenceClipRelevance(adapter, configuration()['clip_snapshot'])
    metadata = {'format': FORMAT+'_metadata', 'study_sha256': study['sha256'], 'model_tag': model,
        'split': split, 'world_size': world_size, 'manifest_sha256': study['manifest_sha256'],
        'checkpoint_sha256': file_digest(spec['checkpoint']), 'contract_sha256': digest(CONTRACT),
        'selectors': list(selectors), 'backbone': backbone, 'cache_contract': old_metadata['cache_contract'],
        'answers_loaded': False, 'source_hashes': source_pins(spec['adapter'])}
    directory = root/model/(split+'-curves')
    write_immutable(directory/f'rank{rank}.metadata.json', metadata)
    entries = sorted(manifest.splits[split], key=lambda e:(e['dataset'],e['row_hash']))[rank::world_size]
    pins, started = {}, time.monotonic()
    for entry in entries:
        path = directory/'features'/(entry['row_hash']+'.json')
        if path.exists():
            if not resume:
                raise ValueError('existing feature shard requires explicit resume')
            validate_feature(execution.read(path), entry, metadata, saved[entry['row_hash']])
        else:
            handle = None
            if clip is not None:
                clip.release()
                handle = adapter.model.vision_tower.register_forward_hook(clip.capture)
            try:
                with torch.no_grad():
                    prepared = adapter.prepare_sample(dataset.row_for_entry(entry), include_answer=False)
                    values = summarize(prepared, scorer, selectors=selectors,
                        cd_relevance=clip(prepared) if clip is not None else None)
                    del prepared
            finally:
                if handle is not None:
                    handle.remove()
                    clip.release()
            value = {'format': FORMAT, 'identity_sha256': digest(metadata), 'request_id': entry['row_hash'],
                'group_id': entry['group_id'], 'image_sha256': entry['image_sha256'], 'dataset': entry['dataset'],
                'model_tag': model, 'split': split, 'checkpoint_sha256': metadata['checkpoint_sha256'],
                'contract_sha256': digest(CONTRACT), 'answers_loaded': False,
                'answers_consumed_by_feature_extractor': False, **values}
            validate_feature(value, entry, metadata, saved[entry['row_hash']])
            write_immutable(path,value)
        assert_frozen(adapter.model,versions)
        assert_frozen(scorer,scorer_versions)
        pins[entry['row_hash']] = file_digest(path)
    write_immutable(directory/f'rank{rank}.complete.json', {'identity_sha256': digest(metadata),
        'requests': len(entries), 'row_sha256': pins})
    print(json.dumps({'event':'curves_complete','model':model,'split':split,'rank':rank,
        'requests':len(entries),'elapsed_seconds':time.monotonic()-started}), flush=True)


def collect(study, model):
    execution.verify(study)
    result = {}
    for split in execution.SPLITS:
        directory = Path(study['output_root'])/model/(split+'-curves')
        if not all((directory/f'rank{r}.complete.json').exists() for r in range(study['world_size'])):
            return None
        manifest, _, _, saved = execution.development_features(study,model,split)
        entries = sorted(manifest.splits[split], key=lambda e:(e['dataset'],e['row_hash']))
        identities = set()
        expected_sources = source_pins(study['models'][model]['adapter'])
        for rank in range(study['world_size']):
            metadata = execution.read(directory/f'rank{rank}.metadata.json')
            marker = execution.read(directory/f'rank{rank}.complete.json')
            for key, expected in {'format':FORMAT+'_metadata','study_sha256':study['sha256'],
                    'manifest_sha256':study['manifest_sha256'],'world_size':study['world_size'],
                    'model_tag':model,'split':split,'source_hashes':expected_sources,
                    'checkpoint_sha256':file_digest(study['models'][model]['checkpoint']),
                    'answers_loaded':False,'contract_sha256':digest(CONTRACT)}.items():
                if metadata.get(key) != expected:
                    raise ValueError('stopping metadata changed: '+key)
            validate_selectors(metadata['selectors'],model)
            identities.add(digest(metadata))
            assigned,pins = entries[rank::study['world_size']],{}
            for entry in assigned:
                path=directory/'features'/(entry['row_hash']+'.json')
                value=execution.read(path)
                validate_feature(value,entry,metadata,saved[entry['row_hash']])
                pins[entry['row_hash']]=file_digest(path)
                if entry['row_hash'] in result:
                    raise ValueError('duplicate request across feature splits')
                result[entry['row_hash']]=value
            if marker != {'identity_sha256':digest(metadata),'requests':len(assigned),'row_sha256':pins}:
                raise ValueError('completion marker does not bind exact feature shard bytes')
        if len(identities)!=1 or {p.stem for p in (directory/'features').glob('*.json')}!={e['row_hash'] for e in entries}:
            raise ValueError('mixed or extra feature shards')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--model-tag',choices=execution.MODELS,required=True)
    parser.add_argument('--adapter',choices=('qwen2_5_vl','llava_next'),required=True)
    parser.add_argument('--split',choices=execution.SPLITS,required=True)
    parser.add_argument('--rank',type=int,required=True)
    parser.add_argument('--world-size',type=int,required=True)
    parser.add_argument('--selectors',nargs='+',choices=SELECTORS,default=('scorer','divprune'))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    study=execution.checked(args.study)
    if study['models'][args.model_tag]['adapter']!=args.adapter:
        raise ValueError('task adapter/model mismatch')
    extract(study,args.model_tag,args.split,args.rank,args.world_size,selectors=args.selectors,resume=args.resume)


if __name__=='__main__':
    main()
