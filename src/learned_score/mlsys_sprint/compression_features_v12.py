"""One-pass, answer-free comparison of rank32 and native visual coverage.

The patch selector, relevance weights, and retained-count menu are frozen v11.
Only the similarity geometry changes. Historical source rows and source-study
deadlines are never modified. A new sealed v12 study supplies the output root
and deadline. This worker reads no answer or quality columns.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import re
import time
from types import SimpleNamespace

from . import compression_execution_v11 as execution
from . import compression_features_v11 as previous
from . import compression_stopping_v11 as stopping
from .followup_common import write_immutable
from .state import digest, file_digest

FORMAT = 'mlsys_dual_coverage_features_v12'
RETENTIONS = previous.RETENTIONS
GEOMETRIES = ('rank32', 'native')
CURVE_KEYS = ('scorer', 'native')
PARITY_ATOL = 1e-6
CONTRACT = {
    'retention_percentages': RETENTIONS,
    'selector': 'unchanged_frozen_rank32_stable_top_score',
    'weights': previous.CONTRACT['scorer_weights'],
    'rank32_geometry': previous.CONTRACT['scorer_geometry'],
    'native_geometry': 'normalized original projected visual embeddings before rank32 projection',
    'uncovered_mass': previous.CONTRACT['uncovered_mass'],
    'missing_mass': previous.CONTRACT['missing_mass'],
    'similarity_block_size': stopping.MAX_BLOCK_SIZE,
    'historical_float_parity_atol': PARITY_ATOL,
    'historical_selection_parity': 'exact_hash_all_physical_prefixes',
    'quality_monotonicity_assumed': False,
    'answers_consumed': False,
}


def compute_feature_curves(scores, visual_features, projected_visual_features, *, retentions=RETENTIONS):
    """Pure scorer-only curves; same selection/weights, different geometry.

    Returns ``{'scorer': rank32_curve, 'native': native_curve}``. Both curves
    carry selector='scorer'; 'native' is not a different patch selector. Invalid
    geometry falls back to full retention, while OOM/process errors propagate.
    The inherited kernel retains at most N-by-64 similarities and does not
    mutate tensors or scorer parameters.
    """
    import torch
    if not isinstance(visual_features, torch.Tensor) or visual_features.ndim != 2 or not all(visual_features.shape):
        raise ValueError('nonempty N-by-D native visual features required')
    if (not isinstance(projected_visual_features, torch.Tensor)
            or projected_visual_features.ndim != 2
            or len(projected_visual_features) != len(visual_features)
            or not projected_visual_features.shape[1]):
        raise ValueError('rank32 and native feature row counts must agree')
    # Reuse rather than reimplement the old sorting, weights, and FP32 kernel.
    current = previous.compute_feature_curves(scores, visual_features, projected_visual_features,
                                              selectors=('scorer',), retentions=retentions)['scorer']
    current = {**current, 'coverage_geometry': 'rank32'}
    n, counts = len(visual_features), current['counts']
    if current.get('fallback'):
        native = previous._fallback(n, 'scorer', 'rank32 reference invalid: '+current['fallback_reason'])
        native.update(geometry=CONTRACT['native_geometry'], coverage_geometry='native')
        return {'scorer': current, 'native': native}
    try:
        nonfull = [k for k in counts if k < n]
        order = (stopping.selector_order('asymvlm', scores=scores, keep_count=max(nonfull))
                 if nonfull else torch.empty(0, dtype=torch.long, device=visual_features.device))
        native = stopping.stopping_curve(visual=visual_features, order=order, counts=counts,
            scores=scores, weight_mode='learned_scores', selector='asymvlm',
            geometry=CONTRACT['native_geometry'])
        native.update(selector='scorer', coverage_geometry='native', valid=True,
            physical_prefix_parity=True, selected_indices_sha256=copy.deepcopy(current['selected_indices_sha256']),
            ranking_sha256=digest(order.cpu().tolist()))
        for key in ('counts', 'missing_mass', 'removed_fraction', 'weights_sha256',
                    'order_prefix_sha256', 'ranking_sha256'):
            if native[key] != current[key]:
                raise ValueError('coverage geometry changed frozen selection or weights: '+key)
    except (ValueError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower():
            raise
        native = previous._fallback(n, 'scorer', f'{type(exc).__name__}: {exc}')
        native.update(geometry=CONTRACT['native_geometry'], coverage_geometry='native')
    return {'scorer': current, 'native': native}


def assert_physical_prefix_parity(curves, physical_hashes):
    """Check historical physical selections without reading their quality.

    ``physical_hashes`` is count -> SHA256 of sorted selected indices. It must
    contain every nonfallback candidate count. Callers audit physical source,
    checkpoint, manifest, and prompt identities before providing these hashes.
    """
    expected = {str(k): v for k, v in physical_hashes.items()}
    for key in CURVE_KEYS:
        curve = curves[key]
        for count, fingerprint in curve['selected_indices_sha256'].items():
            if expected.get(count) != fingerprint:
                raise ValueError(f'physical selected-index parity failed: {key} count {count}')
    return True


def assert_reference_parity(curves, saved):
    """Exact prefixes and <=1e-6 numeric drift against saved v11 scorer curve."""
    current, reference = curves['scorer'], saved['curves']['scorer']
    for key in ('input_visual_tokens', 'counts', 'selector', 'geometry', 'weight_mode',
                'invalid', 'fallback', 'valid', 'physical_prefix_parity', 'ranking_sha256',
                'selected_indices_sha256'):
        if current.get(key) != reference.get(key):
            raise ValueError('frozen v11 scorer curve differs: '+key)
    drift = {}
    for key in ('missing_mass', 'uncovered_mass', 'removed_fraction'):
        left, right = current[key], reference[key]
        if len(left) != len(right):
            raise ValueError('frozen curve length differs: '+key)
        delta = max((abs(a-b) for a, b in zip(left, right)), default=0.)
        if not math.isfinite(delta) or delta > PARITY_ATOL:
            raise ValueError('frozen curve numerical parity failed: '+key)
        drift[key] = delta
    assert_physical_prefix_parity(curves, reference['selected_indices_sha256'])
    return {'selected_indices_exact': True, 'rank32_max_absolute_drift': drift,
            'absolute_tolerance': PARITY_ATOL}


def summarize(prepared, scorer):
    """Compute both geometries from one already encoded image/question."""
    import torch
    from torch.nn import functional as F
    from learned_score.mlsys_scorer.models import Rank32Scorer
    from .adaptive_utility_v3 import calculate_statistic
    if not isinstance(scorer, Rank32Scorer):
        raise ValueError('unchanged frozen Rank32Scorer required')
    with torch.no_grad():
        visual = prepared.inputs_embeds[0, prepared.visual_positions]
        question = prepared.inputs_embeds[0, prepared.semantic_text_positions]
        scores = scorer(visual, question).float()
        projected = F.linear(F.normalize(visual.float(), dim=-1), scorer.visual.weight.float())
        curves = compute_feature_curves(scores, visual, projected)
    return {'curves': curves, 'input_visual_tokens': len(visual),
            'input_text_tokens': int(prepared.inputs_embeds.shape[1]-len(visual)),
            'normalized_entropy': calculate_statistic(scores, 'normalized_entropy')}


def verify_study(study):
    """Verify frozen source identities without applying the expired v11 deadline."""
    from .adaptive_planning_v4 import _checked
    _checked(study)
    source_path = Path(study['source_study']).resolve()
    expected = study.get('source_study_sha256', study.get('input_hashes', {}).get(str(source_path)))
    if expected != file_digest(source_path):
        raise ValueError('v12 must pin the exact source-study file')
    source = execution.checked(source_path)
    execution.verify(source)
    cwd, root = Path(source['cwd']).resolve(), Path(study['output_root']).resolve()
    if (Path(study['cwd']).resolve() != cwd or root.parent != cwd/'reproduction_results'
            or not re.fullmatch(r'mlsys_compression_[0-9]{8}_v12', root.name)
            or root == Path(source['output_root']).resolve()):
        raise ValueError('v12 requires its own sibling output namespace')
    if type(study['world_size']) is not int or not 1 <= study['world_size'] <= 64:
        raise ValueError('invalid feature shard count')
    if (type(study['deadline_utc_epoch']) not in (int, float)
            or not math.isfinite(study['deadline_utc_epoch'])):
        raise ValueError('new finite study deadline required')
    for key in ('models', 'manifest', 'manifest_sha256'):
        if key in study and study[key] != source[key]:
            raise ValueError('v12 changed frozen source identity: '+key)
    if tuple(study.get('coverage_geometries', GEOMETRIES)) != GEOMETRIES:
        raise ValueError('v12 requires the declared rank32/native ablation')
    for path, expected_hash in study.get('input_hashes', {}).items():
        if file_digest(path) != expected_hash:
            raise ValueError('v12 input changed: '+path)
    return source


def source_pins(adapter):
    return {**previous.source_pins(adapter), str(Path(__file__).resolve()): file_digest(__file__)}


def _source_rows(source, model):
    """Return answer-free references and group/geometry identities only."""
    result, manifests, old_metadata = [], {}, {}
    for split in execution.SPLITS:
        manifest, _, metadata, native = execution.development_features(source, model, split)
        manifests[split], old_metadata[split] = manifest, metadata
        directory = Path(source['output_root'])/model/(split+'-curves')
        old = execution.read(directory/'rank0.metadata.json')
        for key, expected in {'study_sha256': source['sha256'], 'model_tag': model, 'split': split,
                'manifest_sha256': source['manifest_sha256'],
                'checkpoint_sha256': file_digest(source['models'][model]['checkpoint']),
                'source_hashes': previous.source_pins(source['models'][model]['adapter'])}.items():
            if old.get(key) != expected:
                raise ValueError('historical feature metadata differs: '+key)
        for entry in manifest.splits[split]:
            path = directory/'features'/(entry['row_hash']+'.json')
            saved = execution.read(path)
            if previous._forbidden(saved):
                raise ValueError('historical feature row contains answers or quality')
            # The old extractor metadata is independent of rank; audit one
            # completed shard and all feature identity fields used for reuse.
            previous.validate_feature(saved, entry, old, native[entry['row_hash']])
            result.append((split, entry, path, saved))
    ids = [e['row_hash'] for _, e, _, _ in result]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate request across development splits')
    if len({digest(meta['cache_contract']) for meta in old_metadata.values()}) != 1:
        raise ValueError('development splits have different cache contracts')
    return sorted(result, key=lambda r: (r[1]['dataset'], r[1]['row_hash'])), manifests, old_metadata


def validate_feature(value, entry, metadata, saved, source_path):
    if previous._forbidden(value):
        raise ValueError('v12 feature contains answers or quality')
    fields = {'format': FORMAT, 'identity_sha256': digest(metadata), 'request_id': entry['row_hash'],
        'group_id': entry['group_id'], 'image_sha256': entry['image_sha256'], 'dataset': entry['dataset'],
        'model_tag': metadata['model_tag'], 'checkpoint_sha256': metadata['checkpoint_sha256'],
        'contract_sha256': digest(CONTRACT), 'answers_loaded': False,
        'answers_consumed_by_feature_extractor': False, 'source_feature_sha256': file_digest(source_path),
        'input_visual_tokens': saved['input_visual_tokens'], 'input_text_tokens': saved['input_text_tokens'],
        'split': saved['split']}
    for key, expected in fields.items():
        if value.get(key) != expected:
            raise ValueError('v12 feature identity differs: '+key)
    if set(value.get('curves', {})) != set(CURVE_KEYS):
        raise ValueError('v12 requires exactly scorer and native curves')
    entropy = value.get('normalized_entropy')
    if (type(entropy) not in (int, float) or not math.isfinite(entropy) or not 0 <= entropy <= 1
            or abs(entropy-saved['normalized_entropy']) > PARITY_ATOL):
        raise ValueError('frozen scorer entropy differs')
    audit = assert_reference_parity(value['curves'], saved)
    if value.get('reference_parity') != audit:
        raise ValueError('missing exact reference-parity audit')
    for key, geometry in zip(CURVE_KEYS, GEOMETRIES):
        curve = value['curves'][key]
        if curve.get('coverage_geometry') != geometry or curve.get('selector') != 'scorer':
            raise ValueError('geometry ablation changed the selector')
        # Validate numeric/count invariants through the same answer-free policy
        # validation used by calibration, without exposing a single label.
        from .compression_policy_v11 import validate_rows
        validate_rows([{'row_hash': entry['row_hash'], 'group_id': entry['group_id'],
                        'input_visual_tokens': value['input_visual_tokens'], 'curves': {key: curve}}],
                      key, labels=False)
        if not curve.get('fallback'):
            for name in ('weights_sha256', 'missing_mass', 'removed_fraction', 'ranking_sha256',
                         'selected_indices_sha256'):
                if curve[name] != value['curves']['scorer'][name]:
                    raise ValueError('native geometry changed selector/weights: '+name)


def extract_shard(study, model, rank, world_size, *, resume=False):
    import torch
    from learned_score.mlsys_scorer.backbone import frozen_versions, assert_frozen
    from . import adaptive_experiment_v3 as runtime
    source = verify_study(study)
    if model not in execution.MODELS or world_size != study['world_size'] or not 0 <= rank < world_size:
        raise ValueError('feature shard differs from v12 study')
    if time.time() >= study['deadline_utc_epoch']:
        raise ValueError('v12 feature deadline expired')
    records, manifests, old_metadata = _source_rows(source, model)
    spec, directory = source['models'][model], Path(study['output_root'])/model/'features'
    args = SimpleNamespace(adapter=spec['adapter'], model=spec['model_path'], model_revision=spec['revision'],
        checkpoint=Path(spec['checkpoint']), manifest=Path(source['manifest']), split='fit',
        training_manifest=Path(source['cwd'])/'reproduction_results/mlsys_20261012/manifest.json',
        device='cuda:0', max_visual_patches=spec.get('max_visual_patches', 1280),
        attention_implementation='sdpa', backbone_dtype='bfloat16', dataset_splits=('scorer_dev',))
    adapter, _, backbone, scorer = runtime.load_runtime(args, manifests['fit'])
    versions, scorer_versions = frozen_versions(adapter.model), frozen_versions(scorer)
    datasets = {split: previous.AnswerFreeDataset(source['manifest'], split, project_root=source['cwd'])
                for split in execution.SPLITS}
    metadata = {'format': FORMAT+'_metadata', 'study_sha256': study['sha256'],
        'source_study_sha256': source['sha256'], 'model_tag': model, 'world_size': world_size,
        'manifest_sha256': source['manifest_sha256'], 'checkpoint_sha256': file_digest(spec['checkpoint']),
        'contract_sha256': digest(CONTRACT), 'coverage_geometries': list(GEOMETRIES),
        'backbone': backbone, 'cache_contract': old_metadata['fit']['cache_contract'],
        'answers_loaded': False, 'source_hashes': source_pins(spec['adapter'])}
    write_immutable(directory/f'rank{rank}.metadata.json', metadata)
    pins, started = {}, time.monotonic()
    for split, entry, source_path, saved in records[rank::world_size]:
        if time.time() >= study['deadline_utc_epoch']:
            raise ValueError('v12 deadline reached; completed atomic features remain resumable')
        path = directory/(entry['row_hash']+'.json')
        if path.exists():
            if not resume:
                raise ValueError('existing features require explicit resume')
            value = execution.read(path)
        else:
            dataset = datasets[split]
            adapter.dataset = dataset
            with torch.no_grad():
                prepared = adapter.prepare_sample(dataset.row_for_entry(entry), include_answer=False)
                values = summarize(prepared, scorer)
                del prepared
            value = {'format': FORMAT, 'identity_sha256': digest(metadata), 'request_id': entry['row_hash'],
                'group_id': entry['group_id'], 'image_sha256': entry['image_sha256'], 'dataset': entry['dataset'],
                'model_tag': model, 'split': split, 'checkpoint_sha256': metadata['checkpoint_sha256'],
                'contract_sha256': digest(CONTRACT), 'answers_loaded': False,
                'answers_consumed_by_feature_extractor': False, 'source_feature_sha256': file_digest(source_path),
                'reference_parity': assert_reference_parity(values['curves'], saved), **values}
        validate_feature(value, entry, metadata, saved, source_path)
        write_immutable(path, value)
        assert_frozen(adapter.model, versions)
        assert_frozen(scorer, scorer_versions)
        pins[entry['row_hash']] = file_digest(path)
        print(json.dumps({'event': 'feature_complete', 'model': model, 'rank': rank,
                          'request_id': entry['row_hash'], 'completed': len(pins)}), flush=True)
    marker = {'identity_sha256': digest(metadata), 'requests': len(pins), 'row_sha256': pins}
    write_immutable(directory/f'rank{rank}.complete.json', marker)
    print(json.dumps({'event': 'curves_complete', 'model': model, 'rank': rank,
                      'requests': len(pins), 'elapsed_seconds': time.monotonic()-started}), flush=True)
    return marker


def collect(study, model):
    """Return all feature rows only after exact completion-marker verification."""
    source = verify_study(study)
    if model not in execution.MODELS:
        raise ValueError('unsupported model')
    directory = Path(study['output_root'])/model/'features'
    if not all((directory/f'rank{r}.complete.json').exists() for r in range(study['world_size'])):
        return None
    records, _, _ = _source_rows(source, model)
    result, identities = {}, set()
    for rank in range(study['world_size']):
        metadata = execution.read(directory/f'rank{rank}.metadata.json')
        marker = execution.read(directory/f'rank{rank}.complete.json')
        for key, expected in {'format': FORMAT+'_metadata', 'study_sha256': study['sha256'],
                'source_study_sha256': source['sha256'], 'model_tag': model, 'world_size': study['world_size'],
                'manifest_sha256': source['manifest_sha256'], 'contract_sha256': digest(CONTRACT),
                'checkpoint_sha256': file_digest(source['models'][model]['checkpoint']),
                'coverage_geometries': list(GEOMETRIES), 'answers_loaded': False,
                'source_hashes': source_pins(source['models'][model]['adapter'])}.items():
            if metadata.get(key) != expected:
                raise ValueError('v12 feature metadata differs: '+key)
        identities.add(digest(metadata))
        pins = {}
        for _, entry, source_path, saved in records[rank::study['world_size']]:
            path = directory/(entry['row_hash']+'.json')
            value = execution.read(path)
            validate_feature(value, entry, metadata, saved, source_path)
            result[entry['row_hash']], pins[entry['row_hash']] = value, file_digest(path)
        if marker != {'identity_sha256': digest(metadata), 'requests': len(pins), 'row_sha256': pins}:
            raise ValueError('completion marker does not bind exact v12 feature bytes')
    if len(identities) != 1:
        raise ValueError('mixed v12 extraction identities')
    actual = {p.stem for p in directory.glob('*.json') if not p.name.startswith('rank')}
    if actual != set(result):
        raise ValueError('unexpected or missing v12 feature rows')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--model-tag', choices=execution.MODELS, required=True)
    parser.add_argument('--adapter', choices=('qwen2_5_vl', 'llava_next'))
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    study = execution.checked(args.study)
    if args.adapter and verify_study(study)['models'][args.model_tag]['adapter'] != args.adapter:
        raise ValueError('task adapter/model mismatch')
    extract_shard(study, args.model_tag, args.rank, args.world_size, resume=args.resume)


if __name__ == '__main__':
    main()
