"""Answer-free, query-weighted uncovered evidence at a fixed 30% anchor.

This module never generates answers or reads answer columns. The frozen rank32
scorer provides relevance and its normalized 32-dimensional visual projections.
Only one scalar and its provenance are saved, never full GPU embeddings.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

from . import compression_execution_v10 as execution
from . import adaptive_experiment_v3 as runtime
from .adaptive_data_v3 import load_manifest
from .followup_common import write_immutable
from .state import digest, file_digest

FORMAT = 'mlsys_query_weighted_coverage_v10'
STATISTIC = 'query_weighted_coverage'
FORMULA = 'sum_i softmax(zscore_population(raw_rank32_scores))_i * (1-max_j_in_top_ceil_30_percent max(0,cos(projected_rank32_visual_i,projected_rank32_visual_j)))'
ANCHOR_PERCENT = 30
CHUNK_SIZE = 256
FORBIDDEN = frozenset({'answer', 'answers', 'label', 'labels', 'references', 'prediction', 'quality_score', 'quality_by_count'})


def source_pins(adapter):
    root = Path(__file__).resolve().parents[1]
    return {**{str(root / name): sha for name, sha in runtime.source_hashes(adapter).items()},
            str(Path(__file__).resolve()): file_digest(__file__),
            str(Path(execution.__file__).resolve()): file_digest(execution.__file__)}


def weighted_uncovered(scores, projected_visual, *, chunk_size=CHUNK_SIZE):
    """FP32 dense/chunk-equivalent u30; exact stable top-ceiling anchor."""
    import torch
    from torch.nn import functional as F
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('positive integer chunk size required')
    scores, visual = scores.detach().float(), projected_visual.detach().float()
    if (scores.ndim != 1 or visual.ndim != 2 or not scores.numel()
            or len(scores) != len(visual) or visual.shape[1] != 32
            or not bool(torch.isfinite(scores).all()) or not bool(torch.isfinite(visual).all())):
        raise ValueError('finite nonempty rank32 scores and projected visual rows required')
    norms = visual.norm(dim=-1)
    if bool((norms <= 1e-12).any()):
        raise ValueError('zero projected visual norm has undefined coverage')
    visual = F.normalize(visual, dim=-1)
    n, count = len(scores), (len(scores)*ANCHOR_PERCENT+99)//100
    order = torch.argsort(scores, descending=True, stable=True)
    chosen = order[:count]
    standardized = (scores-scores.mean()) / scores.std(unbiased=False).clamp_min(1e-8)
    weights = torch.softmax(standardized, dim=0)
    selected = visual.index_select(0, chosen)
    weighted = []
    for start in range(0, n, chunk_size):
        stop = min(start+chunk_size, n)
        similarity = (visual[start:stop] @ selected.T).clamp(0., 1.)
        coverage = similarity.max(dim=-1).values
        # Exact self representation prevents numerical diagonal error.
        coverage[torch.isin(torch.arange(start, stop, device=scores.device), chosen)] = 1.
        weighted.append((weights[start:stop]*(1.-coverage)).double().sum())
    value = float(torch.stack(weighted).sum())
    if not math.isfinite(value) or not -1e-7 <= value <= 1.+1e-7:
        raise ValueError('coverage scalar outside its mathematical range')
    indices = chosen.sort().values.cpu().tolist()
    return {STATISTIC: min(1., max(0., value)), 'anchor_percent': ANCHOR_PERCENT,
            'anchor_count': count, 'anchor_selected_indices_sha256': digest(indices),
            'ranking_sha256': digest(order.cpu().tolist())}


def summarize(prepared, scorer):
    import torch
    from torch.nn import functional as F
    from learned_score.mlsys_scorer.models import Rank32Scorer
    from .adaptive_utility_v3 import calculate_statistic
    if not isinstance(scorer, Rank32Scorer):
        raise ValueError('u30 requires the frozen rank32 scorer')
    with torch.no_grad():
        vision = prepared.inputs_embeds[0, prepared.visual_positions]
        question = prepared.inputs_embeds[0, prepared.semantic_text_positions]
        scores = scorer(vision, question).float()
        projected = F.linear(F.normalize(vision.float(), dim=-1), scorer.visual.weight.float())
        result = weighted_uncovered(scores, projected)
    return {**result, 'normalized_entropy': calculate_statistic(scores, 'normalized_entropy'),
            'input_visual_tokens': len(vision),
            'input_text_tokens': int(prepared.inputs_embeds.shape[1]-len(vision))}


class AnswerFreeDataset:
    name = 'ASYMVLM_V10_ANSWER_FREE'

    def __init__(self, manifest_path, split, *, project_root):
        if split not in execution.SPLITS:
            raise ValueError('coverage extraction is development-only')
        self.manifest = load_manifest(manifest_path)
        self.root = Path(project_root).resolve()
        self.allowed = {row['row_hash']: row for row in self.manifest.splits[split]}
        self._verified, self._tables = set(), {}

    def row_for_entry(self, entry):
        if self.allowed.get(entry['row_hash']) != entry or FORBIDDEN & set(entry):
            raise ValueError('unknown or label-bearing feature identity')
        source = (self.root / entry['source_file']).resolve()
        approved = (self.root / '.data/LMUData' / (entry['dataset']+'.tsv')).resolve()
        if not source.is_relative_to(self.root) and source != approved:
            raise ValueError('feature source escapes the audited project')
        if source not in self._verified:
            if file_digest(source) != entry['source_shard_sha256']:
                raise ValueError('feature source hash mismatch')
            self._verified.add(source)
        if entry.get('source_format') == 'tsv':
            if source not in self._tables:
                import pandas as pd
                wanted = {r['source_row_index'] for r in self.allowed.values()
                          if Path(self.root / r['source_file']).resolve() == source}
                table = pd.read_csv(source, sep='\t', usecols=['index', 'question'], dtype=str,
                    keep_default_na=False, skiprows=lambda i: i > 0 and i-1 not in wanted)
                if len(table) != len(wanted):
                    raise ValueError('TSV offsets no longer cover the selected questions')
                self._tables[source] = dict(zip(sorted(wanted), table.to_dict('records')))
            raw = self._tables[source][entry['source_row_index']]
            if str(raw['index']) != str(entry['dataset_index']):
                raise ValueError('TSV question identity mismatch')
            image = Path(entry['image_path'])
            if file_digest(image) != entry['image_sha256']:
                raise ValueError('image bytes changed')
            image_value = str(image)
        else:
            if source not in self._tables:
                import pyarrow.parquet as pq
                self._tables.clear()
                self._tables[source] = pq.read_table(source, columns=['image', 'question_id', 'question'])
            raw = self._tables[source].slice(int(entry['source_row_index']), 1).to_pylist()[0]
            if str(raw['question_id']) != str(entry['question_id']):
                raise ValueError('parquet question identity mismatch')
            image_bytes = raw['image']['bytes']
            if hashlib.sha256(image_bytes).hexdigest() != entry['image_sha256']:
                raise ValueError('image bytes changed')
            image_value = io.BytesIO(image_bytes)
        return {'image': image_value, 'question': raw['question'], 'metric': entry['metric'],
                '_manifest_row_hash': entry['row_hash'], '_source_dataset': entry['dataset'],
                'group_id': entry['group_id'], 'category': entry.get('category', '')}

    @staticmethod
    def build_prompt(row):
        if FORBIDDEN & set(row):
            raise ValueError('answer-free prompt received labels')
        suffix = '\nAnswer yes or no.' if row['metric'] == 'mme' else '\nAnswer the question using a single word or phrase.'
        return [{'type': 'image', 'value': row['image']}, {'type': 'text', 'value': row['question']+suffix}]


def validate_feature(value, entry, metadata, saved):
    if FORBIDDEN & set(value):
        raise ValueError('coverage shard contains labels')
    for key, expected in {'format': FORMAT, 'identity_sha256': digest(metadata),
            'request_id': entry['row_hash'], 'group_id': entry['group_id'], 'image_sha256': entry['image_sha256'],
            'dataset': entry['dataset'], 'model_tag': metadata['model_tag'], 'split': metadata['split'],
            'checkpoint_sha256': metadata['checkpoint_sha256'], 'formula': FORMULA,
            'answers_loaded': False, 'answers_consumed_by_feature_extractor': False,
            'anchor_percent': ANCHOR_PERCENT,
            'anchor_count': (saved['input_visual_tokens']*ANCHOR_PERCENT+99)//100}.items():
        if value.get(key) != expected:
            raise ValueError('coverage feature identity or formula differs: '+key)
    if any(value.get(k) != saved[k] for k in ('input_visual_tokens', 'input_text_tokens')):
        raise ValueError('native coverage geometry differs')
    for name in (STATISTIC, 'normalized_entropy'):
        scalar = value.get(name)
        if isinstance(scalar, bool) or not isinstance(scalar, (float, int)) or not math.isfinite(scalar) or not 0 <= scalar <= 1:
            raise ValueError('invalid answer-free scalar')
    if abs(value['normalized_entropy']-saved['normalized_entropy']) > 1e-6:
        raise ValueError('live scorer entropy differs from the frozen checkpoint/features')
    for name in ('ranking_sha256', 'anchor_selected_indices_sha256'):
        if not isinstance(value.get(name), str) or len(value[name]) != 64 or any(c not in '0123456789abcdef' for c in value[name]):
            raise ValueError('missing exact ranking provenance')


def extract(study, model, split, rank, world_size, *, resume=False):
    import torch
    from learned_score.mlsys_scorer.backbone import frozen_versions, assert_frozen
    execution.verify(study)
    if model not in execution.MODELS or world_size != study['world_size'] or not 0 <= rank < world_size:
        raise ValueError('coverage shard differs from frozen study')
    if time.time() >= study['deadline_utc_epoch']:
        raise ValueError('coverage extraction deadline expired')
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
    metadata = {'format': FORMAT+'_metadata', 'study_sha256': study['sha256'], 'model_tag': model,
        'split': split, 'world_size': world_size, 'manifest_sha256': study['manifest_sha256'],
        'checkpoint_sha256': file_digest(spec['checkpoint']), 'formula': FORMULA, 'anchor_percent': ANCHOR_PERCENT,
        'backbone': backbone, 'cache_contract': old_metadata['cache_contract'], 'answers_loaded': False,
        'source_hashes': source_pins(spec['adapter'])}
    directory = root / model / (split+'-coverage')
    write_immutable(directory/f'rank{rank}.metadata.json', metadata)
    entries = sorted(manifest.splits[split], key=lambda e: (e['dataset'], e['row_hash']))[rank::world_size]
    pins, started = {}, time.monotonic()
    for entry in entries:
        path = directory / 'features' / (entry['row_hash']+'.json')
        if path.exists():
            if not resume:
                raise ValueError('existing coverage shard requires explicit resume')
            validate_feature(execution.read(path), entry, metadata, saved[entry['row_hash']])
        else:
            with torch.no_grad():
                prepared = adapter.prepare_sample(dataset.row_for_entry(entry), include_answer=False)
                values = summarize(prepared, scorer)
                del prepared
            value = {'format': FORMAT, 'identity_sha256': digest(metadata), 'request_id': entry['row_hash'],
                'group_id': entry['group_id'], 'image_sha256': entry['image_sha256'], 'dataset': entry['dataset'],
                'model_tag': model, 'split': split, 'checkpoint_sha256': metadata['checkpoint_sha256'],
                'formula': FORMULA, 'answers_loaded': False,
                'answers_consumed_by_feature_extractor': False, **values}
            validate_feature(value, entry, metadata, saved[entry['row_hash']])
            write_immutable(path, value)
        assert_frozen(adapter.model, versions)
        assert_frozen(scorer, scorer_versions)
        pins[entry['row_hash']] = file_digest(path)
    write_immutable(directory/f'rank{rank}.complete.json', {'identity_sha256': digest(metadata),
        'requests': len(entries), 'row_sha256': pins})
    print(json.dumps({'event': 'coverage_complete', 'model': model, 'split': split,
                      'rank': rank, 'requests': len(entries), 'elapsed_seconds': time.monotonic()-started}), flush=True)


def collect(study, model):
    execution.verify(study)
    result = {}
    for split in execution.SPLITS:
        directory = Path(study['output_root'])/model/(split+'-coverage')
        if not all((directory/f'rank{r}.complete.json').exists() for r in range(study['world_size'])):
            return None
        manifest, _, _, saved = execution.development_features(study, model, split)
        entries = sorted(manifest.splits[split], key=lambda e: (e['dataset'], e['row_hash']))
        expected_sources = source_pins(study['models'][model]['adapter'])
        identities = set()
        for rank in range(study['world_size']):
            metadata = execution.read(directory/f'rank{rank}.metadata.json')
            marker = execution.read(directory/f'rank{rank}.complete.json')
            if (metadata.get('study_sha256') != study['sha256'] or metadata.get('source_hashes') != expected_sources
                    or metadata.get('manifest_sha256') != study['manifest_sha256'] or metadata.get('world_size') != study['world_size']
                    or metadata.get('model_tag') != model or metadata.get('split') != split
                    or metadata.get('checkpoint_sha256') != file_digest(study['models'][model]['checkpoint'])
                    or metadata.get('answers_loaded') is not False or metadata.get('formula') != FORMULA):
                raise ValueError('coverage extraction metadata changed')
            identities.add(digest(metadata))
            assigned, pins = entries[rank::study['world_size']], {}
            for entry in assigned:
                path = directory/'features'/(entry['row_hash']+'.json')
                value = execution.read(path)
                validate_feature(value, entry, metadata, saved[entry['row_hash']])
                pins[entry['row_hash']] = file_digest(path)
                if entry['row_hash'] in result:
                    raise ValueError('duplicate request across coverage splits')
                result[entry['row_hash']] = value
            if marker != {'identity_sha256': digest(metadata), 'requests': len(assigned), 'row_sha256': pins}:
                raise ValueError('coverage completion does not bind exact shard bytes')
        if len(identities) != 1 or {p.stem for p in (directory/'features').glob('*.json')} != {e['row_hash'] for e in entries}:
            raise ValueError('mixed or extra coverage shards')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--model-tag', choices=execution.MODELS, required=True)
    p.add_argument('--adapter', choices=('qwen2_5_vl', 'llava_next'), required=True)
    p.add_argument('--split', choices=execution.SPLITS, required=True)
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--world-size', type=int, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    study = execution.checked(args.study)
    if study['models'][args.model_tag]['adapter'] != args.adapter:
        raise ValueError('task adapter/model mismatch')
    extract(study, args.model_tag, args.split, args.rank, args.world_size, resume=args.resume)


if __name__ == '__main__':
    main()
