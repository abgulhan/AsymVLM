"""Restartable v16 feature and exact-count physical workers.

This is a new execution namespace, not a modification of a pinned experiment.
Only image counts vary: prompts, original positions, 128-token greedy decoding,
and the protected-prefix cache with a 512-token history budget remain fixed.
"""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import socket
import time
from types import SimpleNamespace

from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest

FORMAT = 'mlsys_compression_execution_v16'
TASK_FORMAT = FORMAT + '_task'
RETENTIONS = tuple(range(10, 101, 10))
MAX_NEW_TOKENS = 128
GENERATED_BUDGET = 512
ADAPTERS = {'qwen25': 'qwen2_5_vl', 'llava_next': 'llava_next', 'phi35': 'phi3_vision'}


def read(path):
    return json.loads(Path(path).read_text())


def sealed(value):
    return {**value, 'sha256': digest(value)}


def build_task(study, phase, model, split, *, output_root=None, request_ids=None,
               schedule=None, feature_root=None, retentions=RETENTIONS,
               selectors=('scorer',), warmup=1, manifest=None):
    """Construct (do not publish) a fully pinned task for the campaign driver."""
    if study.get('sha256') != digest({k: v for k, v in study.items() if k != 'sha256'}):
        raise ValueError('campaign content hash differs')
    spec, cwd = study['models'][model], Path(study['cwd']).resolve()
    default_manifest = study['evaluation_manifest'] if split == 'evaluation' else study['development_manifests'][model]
    if manifest is not None and Path(manifest).resolve() != Path(default_manifest).resolve():
        if split != 'evaluation':
            raise ValueError('manifest override is restricted to approved evaluation expansion')
        source_key = 'phi35' if model == 'phi35' else 'visual'
        parent_manifest = read(study['source_studies'][source_key])['manifest']
        if (Path(manifest).resolve() != Path(parent_manifest).resolve() or
                study['input_hashes'].get(str(Path(manifest).resolve())) != file_digest(manifest)):
            raise ValueError('expanded manifest is not the pinned parent model panel')
    manifest = manifest or default_manifest
    training_manifest = cwd/'reproduction_results/mlsys_20261012/manifest.json'
    if request_ids is None and phase == 'canary':
        from .adaptive_data_v3 import dataset_family
        entries = read(manifest)['splits'][split]
        request_ids = []
        for domain in ('DocVQA', 'TextVQA'):
            candidates = [e for e in entries if dataset_family(e['dataset']) == domain]
            if candidates:
                request_ids.append(min(candidates, key=lambda e:e['row_hash'])['row_hash'])
        if not request_ids:
            raise ValueError('canary requires at least one DocVQA/TextVQA identity')
    pins = source_pins(spec['adapter'])
    paths = [Path(manifest), Path(spec['checkpoint']), Path(spec['checkpoint']).parent/'complete.json',
             Path(spec['model_path'])/'config.json', training_manifest,
             Path(study['output_root'])/'study.json']
    for name in ('model.safetensors.index.json', 'generation_config.json', 'preprocessor_config.json',
                 'processor_config.json', 'tokenizer_config.json'):
        path = Path(spec['model_path'])/name
        if path.exists():
            paths.append(path)
    paths.extend(Path(spec['model_path']).glob('*.py'))
    if schedule:
        paths.append(Path(schedule))
    if feature_root:
        selected = request_ids
        if selected is None and schedule:
            selected = list(read(schedule)['rows'])
        selected = selected or [e['row_hash'] for e in read(manifest)['splits'][split]]
        paths.extend(Path(feature_root)/'features'/(key+'.json') for key in selected)
    for path in paths:
        pins[str(path.resolve())] = file_digest(path)
    task = {'format': TASK_FORMAT, 'study_sha256': study['sha256'], 'cwd': str(cwd),
        'output_root': str(Path(output_root or Path(study['output_root'])/model/(phase+'-'+split)).resolve()),
        'model_tag': model, 'model': spec, 'manifest': str(Path(manifest).resolve()), 'split': split,
        'training_manifest': str(training_manifest), 'input_hashes': pins,
        'deadline_utc_epoch': study['deadline_utc_epoch'], 'retentions': list(retentions),
        'selectors': list(selectors), 'warmup': warmup}
    if request_ids is not None:
        task['request_ids'] = list(request_ids)
    if schedule:
        task['schedule'] = str(Path(schedule).resolve())
    if feature_root:
        task['feature_root'] = str(Path(feature_root).resolve())
    return validate_task(sealed(task), phase)


def source_pins(adapter):
    from . import adaptive_experiment_v4 as runtime
    root = Path(__file__).resolve().parents[1]
    pins = {str(root / name): sha for name, sha in runtime.source_hashes(adapter).items()}
    for name in ('compression_execution_v16.py', 'compression_features_v16.py',
                 'compression_features_v12.py', 'compression_features_v11.py',
                 'compression_features_v10.py', 'compression_stopping_v11.py',
                 'stable_divprune_v15c.py', 'selector_experiment_v5.py',
                 'importance_benchmark.py'):
        path = Path(__file__).with_name(name)
        pins[str(path)] = file_digest(path)
    return pins


def validate_task(task, phase, *, verify_files=True):
    if (task.get('format') != TASK_FORMAT or
            task.get('sha256') != digest({k: v for k, v in task.items() if k != 'sha256'})):
        raise ValueError('invalid sealed v16 execution task')
    if phase not in ('dev-features', 'eval-features', 'physical', 'canary'):
        raise ValueError('unknown execution phase')
    if task.get('model_tag') not in ADAPTERS or task['model'].get('adapter') != ADAPTERS[task['model_tag']]:
        raise ValueError('model tag and adapter disagree')
    if task.get('split') not in ('fit', 'screening', 'evaluation'):
        raise ValueError('unknown manifest split')
    if phase == 'dev-features' and task['split'] == 'evaluation':
        raise ValueError('development feature worker cannot consume evaluation split')
    if phase == 'eval-features' and task['split'] != 'evaluation':
        raise ValueError('evaluation feature worker requires evaluation split')
    cwd, output = Path(task['cwd']).resolve(), Path(task['output_root']).resolve()
    expected_root = cwd / 'reproduction_results/mlsys_compression_20261025_v16'
    if not output.is_relative_to(expected_root) or output == expected_root:
        raise ValueError('worker outputs must use a distinct v16 subdirectory')
    deadline = task.get('deadline_utc_epoch')
    if type(deadline) not in (int, float) or not math.isfinite(deadline):
        raise ValueError('finite execution deadline required')
    retentions = task.get('retentions', list(RETENTIONS))
    if (not isinstance(retentions, list) or not retentions or
            any(type(p) is not int or not 1 <= p <= 100 for p in retentions) or
            retentions != sorted(set(retentions))):
        raise ValueError('retentions must be distinct increasing integer percentages')
    if task.get('selectors', ['scorer']) not in (['scorer'], ['scorer', 'divprune'], ['divprune']):
        raise ValueError('unsupported physical selector menu')
    ids = task.get('request_ids')
    if ids is not None and (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)):
        raise ValueError('invalid explicit request panel')
    warmup = task.get('warmup', 1)
    if type(warmup) is not int or warmup < 0:
        raise ValueError('invalid warmup count')
    if phase != 'physical' and task.get('schedule'):
        raise ValueError('only physical execution accepts a schedule')
    if verify_files:
        pins = task.get('input_hashes', {})
        required = [task['manifest'], task['model']['checkpoint'],
                    str(Path(task['model']['checkpoint']).parent / 'complete.json'),
                    str(Path(task['model']['model_path']) / 'config.json'),
                    task.get('training_manifest', str(cwd / 'reproduction_results/mlsys_20261012/manifest.json'))]
        if task.get('schedule'):
            required.append(task['schedule'])
        if any(str(Path(p).resolve()) not in pins for p in required):
            raise ValueError('task omits a required manifest/checkpoint/config/schedule pin')
        for filename, sha in source_pins(task['model']['adapter']).items():
            if pins.get(filename) != sha:
                raise ValueError('missing or changed task source/input pin: ' + filename)
        for filename, sha in pins.items():
            if file_digest(filename) != sha:
                raise ValueError('changed task input pin: ' + filename)
    return task


def shard_entries(entries, task, rank, world_size):
    if type(world_size) is not int or not 1 <= world_size <= 64 or type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError('invalid execution shard')
    by_id = {e['row_hash']: e for e in entries}
    if len(by_id) != len(entries):
        raise ValueError('duplicate request in manifest split')
    chosen = task.get('request_ids', list(by_id))
    if set(chosen) - by_id.keys():
        raise ValueError('requested identity outside manifest split')
    return sorted((by_id[r] for r in chosen), key=lambda e: (e['dataset'], e['row_hash']))[rank::world_size]


def knot_cells(n, *, retentions=RETENTIONS, selectors=('scorer',)):
    if type(n) is not int or n < 1:
        raise ValueError('positive input visual count required')
    cells = [{'key': 'all_tokens', 'selector': 'full', 'selected_count': n}]
    for percent in retentions:
        if percent == 100:
            continue
        for selector in selectors:
            cells.append({'key': f'{selector}_uniform_{percent:03d}', 'selector': selector,
                          'selected_count': (n * percent + 99) // 100, 'retention_percent': percent})
    return cells


def answer_free_dataset(task, manifest):
    from .compression_features_v10 import AnswerFreeDataset
    # Reuse its audited column-limited reads and unchanged prompt builder.
    dataset = object.__new__(AnswerFreeDataset)
    dataset.manifest, dataset.root = manifest, Path(task['cwd']).resolve()
    dataset.allowed = {row['row_hash']: row for row in manifest.splits[task['split']]}
    dataset._verified, dataset._tables = set(), {}
    return dataset


def runtime_args(task):
    spec, cwd = task['model'], Path(task['cwd'])
    return SimpleNamespace(adapter=spec['adapter'], model=spec['model_path'],
        model_revision=spec['revision'], checkpoint=Path(spec['checkpoint']),
        manifest=Path(task['manifest']), split=task['split'],
        training_manifest=Path(task.get('training_manifest', cwd / 'reproduction_results/mlsys_20261012/manifest.json')),
        device='cuda:0', num_crops=16, max_visual_patches=spec.get('max_visual_patches', 1280),
        attention_implementation='eager' if spec['adapter'] == 'phi3_vision' else 'sdpa',
        backbone_dtype='bfloat16', dataset_splits=(task['split'],) if spec['adapter'] == 'phi3_vision' else ('scorer_dev',))


def make_selector(cell, scorer, feature, observed):
    import torch
    from . import adaptive_experiment_v3 as prior
    from .adaptive_experiment_v4 import validate_cell_count
    from .adaptive_benchmark_v1 import select_exact_count
    from .selector_experiment_v5 import validate_indices
    from .stable_divprune_v15c import stable_divprune_order
    count = validate_cell_count(cell, feature)
    internal = prior.Cell(cell['selector'], 'full' if cell['selector'] == 'full' else 'uniform', 0)

    @torch.no_grad()
    def select(prepared):
        visual = prepared.inputs_embeds[0, prepared.visual_positions]
        query = prepared.inputs_embeds[0, prepared.semantic_text_positions]
        n, p = len(visual), int(prepared.inputs_embeds.shape[1] - len(visual))
        if (n, p) != (feature['input_visual_tokens'], feature['input_text_tokens']):
            raise ValueError('live request geometry differs from frozen assignment')
        if cell['selector'] == 'full' or count == n:
            indices = torch.arange(n, dtype=torch.long, device=visual.device)
        elif cell['selector'] == 'scorer':
            indices = select_exact_count(scorer(visual, query), visual, count)
        elif cell['selector'] == 'divprune':
            indices = stable_divprune_order(visual, count).sort().values
        else:
            raise ValueError('unsupported v16 patch selector')
        validate_indices(indices, count, n)
        values = indices.detach().cpu().tolist()
        expected = feature.get('curves', {}).get('scorer', {}).get('selected_indices_sha256', {}).get(str(count))
        if cell['selector'] == 'scorer' and expected is not None and digest(values) != expected:
            raise ValueError('physical learned prefix differs from feature prefix')
        observed.update(selected_patch_indices=values, selected_patch_indices_sha256=digest(values),
            input_visual_tokens=n, input_text_tokens=p, retained_visual_tokens=count, visual_retention=count/n)
        return indices
    return select, internal


def _canary(adapter, scorer, prepared, feature):
    import torch
    from learned_score.mlsys_runtime import generate_bounded
    from learned_score.mlsys_runtime.dynamic import generate_dynamic
    from . import phi_adaptive_v4 as phi
    from .importance_benchmark import native_off_parity
    proofs = []
    for count in sorted({feature['input_visual_tokens'], (feature['input_visual_tokens'] + 1)//2}, reverse=True):
        observed = {}
        select, _ = make_selector({'selector': 'scorer', 'selected_count': count}, scorer, feature, observed)
        indices = select(prepared)
        pruned = adapter.physically_prune(prepared, indices, 'original')
        expected_sources = torch.cat((prepared.prompt_text_positions,
                                      prepared.visual_positions[indices])).sort().values
        if (not torch.equal(expected_sources, pruned.retained_source_positions)
                or not torch.equal(prepared.inputs_embeds[:, expected_sources], pruned.inputs_embeds)
                or len(expected_sources.unique()) != len(expected_sources)):
            raise ValueError('pruning altered protected tokens, embeddings, or original source positions')
        if adapter.name == 'phi3_vision':
            expected = phi.native_pruned_ids(adapter, pruned, 32)
        else:
            reference = generate_dynamic(adapter.model, pruned, max_new_tokens=32, slack=1, stop_on_eos=True)
            expected = reference.ids.detach().cpu()
            del reference
        actual = generate_bounded(adapter.model, pruned, max_new_tokens=32, generated_budget=512,
            slack=1, stop_on_eos=True, require_fused_attention=adapter.name != 'phi3_vision')
        proof = {**observed, 'native_dynamic_ids_equal': torch.equal(expected, actual.ids.detach().cpu()),
                 'cache': actual.cache.memory_stats(), 'original_position_policy': pruned.metadata['position_policy'],
                 'protected_prefix_indices_preserved': True, 'embedding_gather_exact': True,
                 'position_ids_shape': list(pruned.position_ids.shape),
                 'position_ids_sha256': digest(pruned.position_ids.detach().cpu().tolist())}
        del actual
        if count == feature['input_visual_tokens']:
            if adapter.name == 'phi3_vision':
                native = phi.native_full_generate_ids(adapter, prepared, 32)
                proof['native_multimodal_ids_equal'] = torch.equal(native, expected)
            else:
                proof['native_full'] = native_off_parity(adapter.model, pruned, prepared.native_inputs, steps=32)
                proof['native_multimodal_ids_equal'] = proof['native_full']['native_ids_equal']
        if (not proof['native_dynamic_ids_equal'] or proof.get('native_multimodal_ids_equal') is False or
                proof['cache']['compaction_events'] or proof['cache']['evicted_tokens']):
            raise RuntimeError('full/partial native parity canary failed: ' + json.dumps(proof))
        proofs.append(proof)
        del pruned
    return proofs


def run(task, phase, rank=0, world_size=1):
    import torch
    from torch.nn import functional as F
    from learned_score.mlsys_scorer.backbone import frozen_versions, assert_frozen
    from .adaptive_data_v3 import load_manifest
    from . import adaptive_experiment_v3 as prior
    from . import adaptive_experiment_v4 as physical
    from . import phi_adaptive_v4 as phi
    from . import compression_features_v16 as features
    validate_task(task, phase)
    manifest = load_manifest(task['manifest'])
    entries = shard_entries(manifest.splits[task['split']], task, rank, world_size)
    schedule = None
    if task.get('schedule'):
        schedule = physical.validate_schedule(read(task['schedule']), manifest_sha256=file_digest(task['manifest']),
            checkpoint_sha256=file_digest(task['model']['checkpoint']), split=task['split'],
            allowed_ids={e['row_hash'] for e in manifest.splits[task['split']]})
        entries = [e for e in entries if e['row_hash'] in schedule['rows']]
    output = Path(task['output_root'])
    if time.time() >= task['deadline_utc_epoch']:
        raise TimeoutError('v16 execution deadline reached before model load')
    args = runtime_args(task)
    with phi.installed() if args.adapter == 'phi3_vision' else nullcontext():
        adapter, label_dataset, backbone, scorer = prior.load_runtime(args, manifest)
    scorer.eval().requires_grad_(False)
    feature_dataset = answer_free_dataset(task, manifest)
    adapter.dataset = feature_dataset if phase != 'physical' else label_dataset
    contract = prior.contract_from_config(adapter.model.config)
    if schedule is not None and schedule['cache_contract'] != contract:
        raise ValueError('physical schedule cache geometry changed')
    versions, scorer_versions = frozen_versions(adapter.model), frozen_versions(scorer)
    metadata = {'format': FORMAT, 'task_sha256': task['sha256'], 'phase': phase, 'model_tag': task['model_tag'],
        'manifest_sha256': manifest.file_sha256, 'checkpoint_sha256': file_digest(args.checkpoint),
        'split': task['split'], 'world_size': world_size, 'backbone': backbone, 'cache_contract': contract,
        'source_hashes': source_pins(args.adapter), 'hardware': torch.cuda.get_device_name(adapter.device),
        'torch_version': torch.__version__, 'cuda_runtime_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'max_new_tokens': MAX_NEW_TOKENS, 'generated_budget': GENERATED_BUDGET,
        'schedule_sha256': file_digest(task['schedule']) if task.get('schedule') else None,
        'answers_loaded': phase == 'physical', 'timing_isolated': False,
        'allocation_count_supplied_offline': True, 'allocation_latency_included': False,
        'baseline_scope': 'corrected_projected_feature_DivPrune_selector_not_official_pipeline'}
    signature = digest(metadata)
    write_immutable(output / f'rank{rank}.metadata.json', metadata)
    complete_path = output / f'rank{rank}.complete.json'
    if complete_path.exists():
        marker = read(complete_path)
        if marker.get('identity_sha256') != signature:
            raise ValueError('completed shard identity changed')
        for path, sha in marker['input_hashes'].items():
            if file_digest(path) != sha:
                raise ValueError('completed result changed: ' + path)
        return marker
    pins, started, measurements = {}, time.monotonic(), 0
    for entry_index, entry in enumerate(entries):
        if time.time() >= task['deadline_utc_epoch']:
            raise TimeoutError('v16 deadline reached; atomic completed rows retained')
        key = entry['row_hash']
        destination = output / ('canaries' if phase == 'canary' else 'features') / (key + '.json')
        if phase != 'physical' and destination.exists():
            existing = read(destination)
            if existing.get('identity_sha256') != signature:
                raise ValueError('feature/canary resume identity changed')
            pins[str(destination)] = file_digest(destination)
            continue
        row = (label_dataset if phase == 'physical' else feature_dataset).row_for_entry(entry)
        if hasattr(row.get('image'), 'seek'):
            row['image'].seek(0)
        with torch.no_grad():
            prepared = adapter.prepare_sample(row, include_answer=False)
            visual = prepared.inputs_embeds[0, prepared.visual_positions]
            query = prepared.inputs_embeds[0, prepared.semantic_text_positions]
            feature = {'input_visual_tokens': len(visual), 'input_text_tokens': int(prepared.inputs_embeds.shape[1]-len(visual))}
            if phase in ('dev-features', 'eval-features'):
                scores = scorer(visual, query).float()
                projected = F.linear(F.normalize(visual.float(), dim=-1), scorer.visual.weight.float())
                curves = features.compute_feature_curves(scores, visual, projected, retentions=RETENTIONS)
                values = prior.feature_summary(prepared, scorer)
                feature.update(values, curves=curves)
                del scores, projected
            elif phase == 'physical' and task.get('feature_root'):
                feature_path = Path(task['feature_root'])/'features'/(key+'.json')
                if task['input_hashes'].get(str(feature_path.resolve())) != file_digest(feature_path):
                    raise ValueError('physical task must pin its feature input')
                saved = read(feature_path)
                for name, value in {**feature, 'request_id': key, 'image_sha256': entry['image_sha256'],
                                    'checkpoint_sha256': metadata['checkpoint_sha256']}.items():
                    if saved.get(name) != value:
                        raise ValueError('frozen feature geometry/checkpoint identity differs: ' + name)
                feature['curves'] = saved['curves']
                pins[str(feature_path)] = file_digest(feature_path)
            if phase == 'canary':
                feature['outcomes'] = _canary(adapter, scorer, prepared, feature)
                feature['passed'] = True
        del prepared, visual, query
        if phase != 'physical':
            value = {'format': FORMAT+'_feature' if phase != 'canary' else FORMAT+'_canary',
                'identity_sha256': signature, 'model_tag': task['model_tag'],
                'request_id': key, 'group_id': entry['group_id'], 'image_sha256': entry['image_sha256'],
                'dataset': entry['dataset'], 'category': entry.get('category', ''),
                'split': task['split'], 'checkpoint_sha256': metadata['checkpoint_sha256'],
                'cache_contract': contract, 'answers_loaded': False,
                'answers_consumed_by_feature_extractor': False, **feature}
            write_immutable(destination, value)
            pins[str(destination)] = file_digest(destination)
        else:
            row_schedule = schedule['rows'][key] if schedule is not None else {
                **feature, 'cells': knot_cells(feature['input_visual_tokens'],
                    retentions=task.get('retentions', RETENTIONS), selectors=task.get('selectors', ['scorer']))}
            references = prior.benchmark.parse_references(row['answer'])
            metric = entry.get('quality_metric', entry.get('metric'))
            cells = row_schedule['cells']
            offset = int(key[:8], 16) % len(cells)
            reused = {}
            for cell in cells[offset:] + cells[:offset]:
                if time.time() >= task['deadline_utc_epoch']:
                    raise TimeoutError('v16 deadline reached between physical cells')
                count = physical.validate_cell_count(cell, feature, row_schedule)
                path = output/'rows'/f"{key}.{cell['key']}.json"
                physical_key = (cell['selector'], count)
                if path.exists():
                    value = read(path)
                    if value.get('identity_sha256') != signature or value.get('status') != 'ok':
                        raise ValueError('physical resume identity/status differs')
                    if value.get('physical_measurement_performed'):
                        reused[physical_key] = value
                    pins[str(path)] = file_digest(path)
                    continue
                record = {'format': FORMAT, 'identity_sha256': signature, 'request_id': key,
                    'group_id': entry['group_id'], 'image_sha256': entry['image_sha256'],
                    'dataset': entry['dataset'], 'category': entry.get('category'), 'model_tag': task['model_tag'],
                    'model': backbone['model'], 'split': task['split'], 'quality_metric': metric,
                    'method': cell['key'], 'selector': cell['selector'], 'selected_count': count,
                    'schedule_cell': cell, 'schedule_content_sha256': schedule['sha256'] if schedule else task['sha256'],
                    'checkpoint_sha256': metadata['checkpoint_sha256'], 'manifest_sha256': manifest.file_sha256,
                    'host': socket.gethostname(), 'job_id': os.getenv('PBS_JOBID'), 'hardware': metadata['hardware'],
                    'timing_isolated': False, 'max_new_tokens': MAX_NEW_TOKENS, 'generated_budget': GENERATED_BUDGET}
                record.update(allocation_count_supplied_offline=True, allocation_latency_included=False)
                if physical_key in reused:
                    record = physical.reuse_physical_measurement(reused[physical_key], record)
                else:
                    observed = {}
                    select, internal = make_selector(cell, scorer, feature, observed)
                    try:
                        if entry_index == 0:
                            for _ in range(task.get('warmup', 1)):
                                _, ids, _, metrics = prior.measured(adapter, row, select, internal, 8)
                                physical.generation_trace(adapter, ids, metrics, 8)
                        prediction, ids, timing, metrics = prior.measured(adapter, row, select, internal, MAX_NEW_TOKENS)
                        metrics = physical.generation_trace(adapter, ids, metrics)
                        expected = physical.expected_cache_bytes(feature, count, contract)
                        prior.check_cache(metrics, expected)
                        record.update(status='ok', prediction=prediction,
                            quality_score=prior.score_prediction(prediction, references, metric),
                            generated_tokens=int(ids.shape[-1]), generated_token_ids=metrics['generated_token_ids'],
                            stop_reason=metrics['stop_reason'], expected_cache_bytes=expected,
                            cache_contract=contract, metrics=metrics, physical_measurement_performed=True,
                            physical_reuse_of_cell=None, timing_reused_not_an_independent_observation=False,
                            **timing, **observed)
                        del ids
                    except Exception as exc:
                        atomic_json(output/'errors'/f"{key}.{cell['key']}.{time.time_ns()}.json",
                            {**record, 'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
                        raise
                    reused[physical_key] = record
                    measurements += 1
                assert_frozen(adapter.model, versions)
                assert_frozen(scorer, scorer_versions)
                write_immutable(path, record)
                pins[str(path)] = file_digest(path)
        assert_frozen(adapter.model, versions)
        assert_frozen(scorer, scorer_versions)
        print(json.dumps({'event': 'v16_progress', 'phase': phase, 'rank': rank, 'requests': entry_index+1,
                          'physical_measurements': measurements, 'elapsed_seconds': time.monotonic()-started}), flush=True)
    marker = sealed({'format': FORMAT+'_complete', 'identity_sha256': signature, 'task_sha256': task['sha256'],
        'requests': len(entries), 'physical_measurements': measurements, 'input_hashes': pins,
        'passed': True, 'elapsed_seconds': time.monotonic()-started})
    write_immutable(complete_path, marker)
    return marker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('dev-features', 'eval-features', 'physical', 'canary'))
    parser.add_argument('--task', required=True, type=Path)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    args = parser.parse_args(argv)
    run(read(args.task), args.phase, args.rank, args.world_size)


if __name__ == '__main__':
    main()
