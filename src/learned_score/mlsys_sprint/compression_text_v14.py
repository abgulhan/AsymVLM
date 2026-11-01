"""Frozen-v13 vision budgeting plus long-generation cache eviction pilot.

Twenty-four existing olmOCR development documents, two models, no retuning or
reserved-page execution. All bounded arms use M=original prefix+B+1 per page;
saved image slots become generated-history slots, not a reduction in that cap.
The unrestricted all-token reference has a different measured capacity. R-KV
is the existing unprotected-prefix adaptation and a separate cache-policy
comparison, not an allocation-only ablation. Quality timings are not isolated.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from statistics import mean
import time

from . import compression_text_v12 as previous
from . import compression_policy_v12 as policy
from .cache_protocol_v4 import _read_sealed, seal, select_development_entries, load_protocol
from .followup_common import (load_followup_protocol, load_model, model_spec,
    require_cuda, verify_native_gate, write_immutable)
from .state import digest, file_digest, atomic_json

FORMAT = 'mlsys_compression_text_v14'
ASSIGNMENTS_FORMAT = 'mlsys_compression_text_assignments_v14'
MODELS, PROFILES = previous.MODELS, previous.PROFILES
METHODS, MAX_NEW_TOKENS = previous.METHODS, previous.MAX_NEW_TOKENS
MAX_SECONDS = previous.MAX_SECONDS
budget_for = previous.budget_for
_BASE_SELECTED = previous.selected_indices


def validate_decision(value):
    methods = {'scorer_static': ('scorer', 'static'), 'scorer_adaptive': ('scorer', 'rank32_coverage'),
        'divprune_static': ('divprune', 'static'), 'divprune_adaptive': ('divprune', 'rank32_coverage')}
    method = value.get('selected_method')
    if (method not in methods or tuple((value.get('selector'), value.get('rule_kind'))) != methods[method]
            or set(value.get('model_rules', {})) != set(MODELS)
            or value.get('independent_confirmation') is not False):
        raise ValueError('pilot requires an existing frozen shared v13 method and both model rules')
    for model, rule in value['model_rules'].items():
        if (rule.get('format') != policy.FORMAT or rule.get('model') != model
                or rule.get('kind') != value['rule_kind']
                or rule.get('dataset_is_policy_input') is not False
                or rule.get('sha256') != digest({k: v for k, v in rule.items() if k != 'sha256'})):
            raise ValueError('frozen v13 rule identity changed')
    return value


def build_config(parent_protocol, decision_path, output_root, *, started_epoch=None, deadline_epoch=None):
    parent_path, decision_path = Path(parent_protocol).resolve(), Path(decision_path).resolve()
    parent = load_protocol(parent_path)
    decision = validate_decision(_read_sealed(decision_path))
    source_path = decision_path.parent / 'study.json'
    from .compression_finalize_v13 import verify
    source = _read_sealed(source_path)
    verify(source)
    if decision.get('study_sha256') != source['sha256']:
        raise ValueError('decision and frozen vision study differ')
    base_path = Path(parent['base_protocol'])
    base = load_followup_protocol(base_path)
    manifest_path = Path(parent['manifest'])
    manifest = json.loads(manifest_path.read_text())
    entries = select_development_entries(manifest['splits']['scorer_dev'])
    if entries != parent['development_entries']:
        raise ValueError('deterministic development page identities changed')
    # Reserved identities may be inspected for disjointness, never their pages
    # or answer tests. The benchmark loader is restricted to scorer_dev.
    reserved = {e['group_id'] for e in manifest['splits'].get('final_test', [])}
    if reserved & {e['group_id'] for e in entries}:
        raise ValueError('development pages overlap reserved source documents')
    gates = []
    for model in MODELS:
        spec = model_spec(base, model)
        expected = source['models'][model]
        if (any(expected.get(k) != spec.get(k) for k in
                ('adapter', 'model_path', 'revision', 'max_visual_patches'))
                or file_digest(expected['checkpoint']) != spec['checkpoint_sha256']):
            raise ValueError('vision and text model/scorer identities must match exactly')
        gates.append(Path(verify_native_gate(base, model)['path']))
    start = time.time() if started_epoch is None else float(started_epoch)
    deadline = start + MAX_SECONDS if deadline_epoch is None else float(deadline_epoch)
    if not 0 < deadline - start <= MAX_SECONDS:
        raise ValueError('joint development pilot has a hard four-hour execution bound')
    from learned_score.mlsys_runtime import matched_generation_v3, matched_cache_v3, dynamic
    from . import compression_features_v12, cache_methods_v3, followup_common, cache_protocol_v4
    modules = [Path(m.__file__) for m in (previous, policy, compression_features_v12,
        matched_generation_v3, matched_cache_v3, dynamic, cache_methods_v3, followup_common, cache_protocol_v4)]
    pins = {**parent['input_hashes'], **base['input_hashes'], **source['input_hashes'],
        **{str(p.resolve()): file_digest(p) for p in
            [parent_path, decision_path, source_path, base_path, manifest_path, Path(__file__), *modules, *gates]}}
    return seal({'format': FORMAT, 'parent_v4_protocol': str(parent_path),
        'frozen_policy': str(decision_path), 'frozen_policy_sha256': decision['sha256'],
        'source_study': str(source_path), 'selected_method': decision['selected_method'],
        'patch_selector': decision['selector'], 'rule_kind': decision['rule_kind'],
        'base_protocol': str(base_path), 'manifest': str(manifest_path),
        'entries': entries, 'active_models': list(MODELS), 'profiles': list(PROFILES),
        'started_epoch': start, 'deadline_epoch': deadline, 'output_root': str(Path(output_root).resolve()),
        'input_hashes': pins, 'split': 'scorer_dev', 'final_pages_opened': False,
        'quality_stage': 'previously_examined_development_pilot', 'independent_confirmation': False,
        'max_new_tokens': MAX_NEW_TOKENS, 'natural_stop_enabled': True,
        'evaluator_root': parent['evaluator_root'], 'cache_patch': parent['cache_patch'],
        'runtime_revision': previous.RUNTIME_REVISION,
        'budget_definition': 'M=original_full_prefix+B+1; staging inside M; identical per-page bounded capacity',
        'matched_control': 'same protected-recency policy and exact aggregate retained image count',
        'rkv_contrast': 'existing unprotected-prefix adaptation; changes cache selection AND protection',
        'assignment_uses_labels': False, 'assignment_timed_in_quality_runs': False,
        'all_full_vision_fallback_policy': 'execute text-cache pilot; report absence of vision-pruning signal',
        'four_hour_cap_includes_assignment_preparation': True})


@contextmanager
def bindings(config, model_bundle=None):
    """Reuse audited physical code without changing any historical file."""
    saved = {key: getattr(previous, key) for key in ('FORMAT', 'ASSIGNMENTS_FORMAT', 'selected_indices', 'load_model')}

    def select(prepared, scorer, keep):
        if config['patch_selector'] == 'scorer':
            return _BASE_SELECTED(prepared, scorer, keep)
        import torch
        from learned_score.llava_selectors import divprune_order
        from .compression_stopping_v11 import selected_positions
        vision = prepared.inputs_embeds[0, prepared.visual_positions]
        order = (torch.empty(0, device=vision.device, dtype=torch.long) if keep == len(vision)
            else divprune_order(vision, keep))
        return selected_positions(order, keep, input_visual_tokens=len(vision))

    def cached_load(base, model, device, *, with_scorer=False):
        expected_model, expected_device, bundle = model_bundle
        if model != expected_model or str(device) != str(expected_device) or with_scorer is not True:
            raise ValueError('loaded text model reuse requested with a different identity')
        return bundle

    previous.FORMAT, previous.ASSIGNMENTS_FORMAT = FORMAT, ASSIGNMENTS_FORMAT
    previous.selected_indices = select
    if model_bundle is not None:
        previous.load_model = cached_load
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(previous, key, value)


def load_config(path):
    raw = _read_sealed(path)
    with bindings(raw):
        value = previous.load_config(path)
    decision = validate_decision(_read_sealed(value['frozen_policy']))
    if (decision['sha256'] != value['frozen_policy_sha256']
            or decision['selector'] != value['patch_selector']
            or decision['selected_method'] != value['selected_method']
            or value.get('assignment_uses_labels') is not False
            or value.get('independent_confirmation') is not False):
        raise ValueError('pilot decision or scientific scope changed')
    return value


def uniform_exact_counts(rows, total):
    """Integer-only waterfilling; never round a fraction back into patch counts."""
    from .compression_policy_v8 import uniform_at_visual_total
    geometry = [{'row_hash': r['request_id'], 'group_id': r['group_id'], 'dataset': 'olmOCR-Bench',
        'input_visual_tokens': r['input_visual_tokens'], 'input_text_tokens': r['input_text_tokens']} for r in rows]
    result = uniform_at_visual_total(geometry, total)
    if not result['feasible']:
        raise ValueError('infeasible exact aggregate image count')
    return result['counts']


def prepare_assignments(config, model, bundle):
    import torch
    from torch.nn import functional as F
    from learned_score.mlsys_scorer.backbone import read_entry, frozen_versions, assert_frozen
    from learned_score.llava_selectors import divprune_order
    from .compression_features_v12 import compute_feature_curves
    path = previous.assignment_path(config, model)
    if path.exists():
        with bindings(config):
            return previous.validate_assignments(_read_sealed(path), config, model)
    adapter, dataset, backbone, scorer, _ = bundle
    scorer.requires_grad_(False)
    versions, scorer_versions = frozen_versions(adapter.model), frozen_versions(scorer)
    decision = validate_decision(_read_sealed(config['frozen_policy']))
    rows = []
    for entry in config['entries']:
        if time.time() >= config['deadline_epoch']:
            raise TimeoutError('text pilot expired during answer-free assignment preparation')
        sample = read_entry(dataset, entry)
        with torch.no_grad():
            prepared = adapter.prepare_sample(sample)
            vision = prepared.inputs_embeds[0, prepared.visual_positions]
            question = prepared.inputs_embeds[0, prepared.semantic_text_positions]
            scores = scorer(vision, question).float()
            projected = F.linear(F.normalize(vision.float(), dim=-1), scorer.visual.weight.float())
            curves = compute_feature_curves(scores, vision, projected)
            order = (torch.argsort(scores, descending=True, stable=True) if config['patch_selector'] == 'scorer'
                else divprune_order(vision, max(0, len(vision) - 1)))
        rows.append({'request_id': entry['row_hash'], 'row_hash': entry['row_hash'],
            'group_id': entry['group_id'], 'image_sha256': entry['image_sha256'],
            'input_visual_tokens': len(vision),
            'input_text_tokens': int(prepared.inputs_embeds.shape[1] - len(vision)),
            'curves': curves, 'full_ranking': order.cpu().tolist()})
        del prepared, vision, question, scores, projected, order
    predictions = policy.infer(decision['model_rules'][model], policy.answer_free(rows))
    controls = uniform_exact_counts(rows, sum(predictions['counts'].values()))
    for row in rows:
        request, n = row['request_id'], row['input_visual_tokens']
        row.update(adaptive_count=predictions['counts'][request], static_count=controls[request],
            fallback_reason=predictions['fallbacks'].get(request))
        order = row.pop('full_ranking')
        row['selected_indices_sha256'] = {kind: digest(list(range(n)) if row[kind + '_count'] == n
            else sorted(order[:row[kind + '_count']])) for kind in ('static', 'adaptive')}
    assert_frozen(adapter.model, versions)
    assert_frozen(scorer, scorer_versions)
    active = mean(r['adaptive_count'] < r['input_visual_tokens'] for r in rows)
    base = load_followup_protocol(config['base_protocol'])
    value = seal({'format': ASSIGNMENTS_FORMAT, 'model': model, 'config_sha256': config['sha256'],
        'frozen_vision_policy_sha256': config['frozen_policy_sha256'], 'rows': rows,
        'answers_consumed': False, 'backbone': backbone, 'checkpoint_sha256': model_spec(base, model)['checkpoint_sha256'],
        'prompt': 'unchanged_olmocr_prompt_no_VQA_suffix', 'labels_used_for_tuning': False,
        'eligible_for_joint_pilot': True, 'frozen_rule_prunes_any': bool(active),
        'eligibility_definition': 'supported frozen method; all-full assignments do not cancel text-cache evaluation',
        'visual_pruning_active_fraction': active, 'fallback_fraction': len(predictions['fallbacks']) / len(rows),
        'patch_selector': config['patch_selector'], 'selected_method': config['selected_method'],
        'no_rule_retuning_on_text_benchmark': True})
    with bindings(config):
        previous.validate_assignments(value, config, model)
    write_immutable(path, value)
    return value


def run_model(config, model, device='cuda:0'):
    """One GPU load per model; atomic per-page markers allow bounded retries."""
    root = Path(config['output_root'])
    complete = root / f'{model}.complete.json'
    if complete.exists():
        return collect_model(config, model)
    base = load_followup_protocol(config['base_protocol'])
    with previous.pilot_window(config):
        bundle = load_model(base, model, device, with_scorer=True)
        assigned = prepare_assignments(config, model, bundle)
        with bindings(config, (model, device, bundle)):
            gate_path = root / 'canary' / f'{model}.json'
            if not gate_path.exists():
                previous.canary(config, model, device)
            for profile in PROFILES:
                for rank in range(24):
                    previous.run_worker(config, model, rank, profile, device)
                    print(json.dumps({'event': 'text_page_complete', 'model': model, 'profile': profile,
                        'rank': rank, 'development_only': True}), flush=True)
        paths = [previous.assignment_path(config, model), gate_path,
            *[root / model / f'b{b}' / f'rank{r}' / 'complete.json' for b in PROFILES for r in range(24)]]
        value = seal({'format': FORMAT + '_complete', 'config_sha256': config['sha256'], 'model': model,
            'profiles': list(PROFILES), 'pages_per_profile': 24, 'input_hashes': {str(p): file_digest(p) for p in paths},
            'physical_generation_complete': True, 'official_quality_scoring_complete': False,
            'visual_pruning_active_fraction': assigned['visual_pruning_active_fraction'],
            'independent_confirmation': False})
        write_immutable(complete, value)
    return collect_model(config, model)


def collect_model(config, model):
    root = Path(config['output_root'])
    value = _read_sealed(root / f'{model}.complete.json')
    if value.get('config_sha256') != config['sha256'] or value.get('model') != model:
        raise ValueError('completed text model identity differs')
    for path, sha in value['input_hashes'].items():
        if file_digest(path) != sha:
            raise ValueError('completed text artifact changed')
    with bindings(config):
        for profile in PROFILES:
            for rank in range(24):
                previous.collect_page(config, model, rank, profile)
    return value


def report(config, model, profile, output):
    from learned_score.mlsys_next.cache_method_report_v3 import paired_difference
    with bindings(config):
        value = previous.report(config, model, profile, output)
        rows = [row for rank in range(24) for row in previous.collect_page(config, model, rank, profile)]
    context = digest({'config': config['sha256'], 'model': model, 'profile': profile, 'rows': rows})
    scores = {method: json.loads((Path(output) / context / method / 'official_scores.json').read_text())
        for method in ('fullref', *METHODS)}
    for summary in value['quality']:
        physical = [r for r in rows if r['method'] == summary['method']]
        summary.update(versus_static_same_cache=paired_difference(scores[summary['method']],
            scores['static_recency'], repetitions=10000),
            mean_request_seconds=mean(r['metrics']['request_seconds'] for r in physical),
            mean_prepared_prefill_ttft_seconds=(mean(r['metrics']['prefill_ttft_seconds'] for r in physical)
                if all(isinstance(r['metrics'].get('prefill_ttft_seconds'), (float, int)) for r in physical) else None),
            generated_tokens_per_request_second=sum(r['generated_tokens'] for r in physical) /
                sum(r['metrics']['request_seconds'] for r in physical),
            mean_eviction_events=mean(r['metrics']['cache'].get('compaction_events', 0) for r in physical))
    value.update(selected_vision_method=config['selected_method'], allocation_uses_cached_frozen_counts=True,
        allocation_cost_excluded_from_quality_timing=True, timing_isolated=False,
        timing_claim='descriptive quality-run timings, not controlled latency or speedup evidence',
        static_label='Uniform image retention at exactly matched aggregate image count',
        adaptive_label='Frozen v13 per-input rule (not retuned on olmOCR)',
        full_reference_label='All image tokens and no cache eviction; not a matched-capacity baseline',
        rkv_comparison_label='Existing R-KV adaptation; unprotected prefix differs from protected-recency arms',
        text_eviction_demonstrated=any(r['method'] == 'adaptive_recency' and r['eviction_active_fraction'] > 0
            for r in value['quality']), exact_per_page_bounded_capacity=True)
    write_immutable(Path(output) / context / 'summary-v14.json', value)
    return value


def advance(study):
    """Publish one restartable GPU task/model and CPU-score complete cohorts."""
    from .compression_campaign_v14 import verify, python, make_task, publish_plan
    verify(study)
    root, now = Path(study['output_root']), time.time()
    config_path = root / 'text/config.json'
    if not config_path.exists():
        parent = Path(study['cwd']) / 'reproduction_results/mlsys_cache_methods_20261014_v4/protocol.json'
        config = build_config(parent, study['decision_path'], root / 'text', started_epoch=now,
            deadline_epoch=min(now + MAX_SECONDS, study['deadline_utc_epoch']))
        write_immutable(config_path, config)
    config = load_config(config_path)
    pins = {str(config_path): file_digest(config_path), **config['input_hashes']}
    status = {'format': FORMAT + '_status', 'config_sha256': config['sha256'], 'config_path': str(config_path),
        'plans': [], 'models': {}, 'independent_confirmation': False, 'deadline_epoch': config['deadline_epoch']}
    for model in MODELS:
        model_status = {'profiles': {}}
        status['models'][model] = model_status
        for profile in PROFILES:
            complete_pages = sum((root / 'text' / model / f'b{profile}' / f'rank{r}' / 'complete.json').exists()
                for r in range(24))
            record = {'complete_paired_pages': complete_pages, 'required_pages': 24}
            model_status['profiles'][str(profile)] = record
            if complete_pages == 24:
                output = root / 'text/reports' / model / f'b{profile}'
                index = output.with_suffix('.json')
                if index.exists():
                    measured = json.loads(index.read_text())
                    if measured.get('config_sha256') != config['sha256']:
                        raise ValueError('official text report configuration changed')
                else:
                    measured = report(config, model, profile, output)
                    write_immutable(index, measured)
                record.update(status='complete', report=str(index),
                    text_eviction_demonstrated=measured['text_eviction_demonstrated'])
        marker = root / 'text' / f'{model}.complete.json'
        if marker.exists():
            collect_model(config, model)
            model_status['physical_status'] = 'complete'
        elif now >= config['deadline_epoch']:
            model_status['physical_status'] = 'deadline_reached_incomplete'
        else:
            argv = python(study, 'compression_text_v14') + ['worker', '--config', str(config_path), '--model', model]
            task = make_task(study, 'text-' + model, argv, [marker], model=model, resource='gpu', minimum_start_seconds=900)
            status['plans'].append(str(publish_plan(study, 'text-' + model, [task], pins)))
            model_status['physical_status'] = 'queued_or_running'
    status['complete'] = all(m['physical_status'] == 'complete' and all(
        p.get('status') == 'complete' for p in m['profiles'].values()) for m in status['models'].values())
    status['status'] = 'complete' if status['complete'] else 'running_or_incomplete'
    atomic_json(root / 'text/status.json', status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    create = commands.add_parser('config')
    create.add_argument('--parent', type=Path, required=True)
    create.add_argument('--decision', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    for name in ('worker', 'report'):
        sub = commands.add_parser(name)
        sub.add_argument('--config', type=Path, required=True)
        sub.add_argument('--model', choices=MODELS, required=True)
        if name == 'worker':
            sub.add_argument('--device', default='cuda:0')
        else:
            sub.add_argument('--profile', type=int, choices=PROFILES, required=True)
            sub.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == 'config':
        value = build_config(args.parent, args.decision, args.output)
        write_immutable(args.output / 'config.json', value)
    else:
        config = load_config(args.config)
        if args.action == 'worker':
            require_cuda(args.device)
            value = run_model(config, args.model, args.device)
        else:
            value = report(config, args.model, args.profile, args.output)
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
