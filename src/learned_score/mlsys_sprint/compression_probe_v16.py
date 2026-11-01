"""Development-only teacher, query-dependence, and spatial-selection probes.

The teacher probe loads only cached training tensors and a frozen small scorer.
Physical controls keep the generation prompt unchanged and change only the
query supplied to the patch scorer, or Qwen's existing 20% regional reserve.
These are diagnosis experiments, not additional test-selected policy families.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
import math
from pathlib import Path
import statistics
import time

from . import compression_execution_v16 as execution
from .followup_common import write_immutable
from .state import digest, file_digest

FORMAT = 'mlsys_compression_probe_v16'
MODES = ('correct', 'shuffled', 'spatial020')
PERCENTAGES = (50, 70)
TRAINING_CACHE = {
    'qwen25': 'reproduction_results/mlsys_20261012/training_v1/qwen25/saliency',
    'llava_next': 'reproduction_results/mlsys_20261012/llava_memory_v2/llava_next/saliency',
    'phi35': 'reproduction_results/mlsys_adaptive_20261016_v3/phi35/saliency',
}


def average_ranks(values):
    """Average zero-based ranks for ties; no arbitrary tie-breaking signal."""
    import torch
    if values.ndim != 1 or not len(values) or not bool(torch.isfinite(values).all()):
        raise ValueError('finite nonempty vector required')
    order = torch.argsort(values, stable=True)
    _, inverse, lengths = torch.unique_consecutive(values[order], return_inverse=True, return_counts=True)
    ends = lengths.cumsum(0)
    averages = (ends - lengths + ends - 1).double()/2
    result = torch.empty(len(values), dtype=torch.float64, device=values.device)
    result[order] = averages[inverse]
    return result


def spearman(left, right):
    a, b = average_ranks(left), average_ranks(right)
    if a.shape != b.shape:
        raise ValueError('rank-correlation sizes differ')
    a, b = a-a.mean(), b-b.mean()
    denom = a.norm()*b.norm()
    return None if float(denom) == 0 else float((a@b)/denom)


def distribution_stats(probability):
    import torch
    p = probability.double()
    if p.ndim != 1 or not len(p) or not bool(torch.isfinite(p).all()) or bool((p < 0).any()) or float(p.sum()) <= 0:
        raise ValueError('finite nonnegative nonzero mass required')
    p = p/p.sum()
    n = len(p)
    return {'maximum_mass':float(p.max()), 'top10_mass':float(p.topk(max(1, math.ceil(n/10))).values.sum()),
        'effective_support_fraction':float(1/p.square().sum()/n),
        'normalized_entropy':float(-(p*p.clamp_min(1e-300).log()).sum()/math.log(n)) if n > 1 else 1.}


def teacher_diagnostics(saliency, correct_scores, shuffled_scores):
    import torch
    s, c, h = (t.detach().double() for t in (saliency, correct_scores, shuffled_scores))
    if (s.ndim != 1 or not len(s) or s.shape != c.shape or s.shape != h.shape or
            not all(bool(torch.isfinite(t).all()) for t in (s,c,h))):
        raise ValueError('finite matching scores and cached normalized teacher target required')
    # Match training's 1e-6 normalization floor, not inference's 1e-8 floor.
    target = torch.softmax((s-s.mean())/s.std(unbiased=False).clamp_min(1e-6), 0)
    student = torch.softmax((c-c.mean())/c.std(unbiased=False).clamp_min(1e-6), 0)
    n = len(s)
    result = {'visual_tokens':n, 'cached_saliency_mean':float(s.mean()), 'cached_saliency_std':float(s.std(unbiased=False)),
        'cached_saliency_minimum_fraction':float((s == s.min()).double().mean()),
        'cached_saliency_all_zero':bool((s == 0).all()),
        'cached_saliency_constant':bool((s == s[0]).all()), 'training_target':distribution_stats(target),
        'scorer_training_distribution':distribution_stats(student),
        'teacher_scorer_spearman':spearman(s,c), 'correct_shuffled_spearman':spearman(c,h),
        'teacher_to_student_kl':float((target*(target.clamp_min(1e-300).log()-student.clamp_min(1e-300).log())).sum()),
        'query_changes_raw_scores':not torch.equal(c,h), 'retentions':{}}
    for percent in (30,50,70):
        k = (n*percent+99)//100
        chosen = torch.argsort(c, descending=True, stable=True)[:k]
        shuffled = torch.argsort(h, descending=True, stable=True)[:k]
        oracle = torch.argsort(s, descending=True, stable=True)[:k]
        result['retentions'][str(percent)] = {'selected_count':k,
            'correct_vs_shuffled_overlap':len(set(chosen.tolist()) & set(shuffled.tolist()))/k,
            'learned_retained_teacher_target_mass':float(target[chosen].sum()),
            'shuffled_retained_teacher_target_mass':float(target[shuffled].sum()),
            'teacher_rank_retained_target_mass':float(target[oracle].sum()),
            'uniform_expected_target_mass':k/n}
    return result


def deterministic_donors(entries):
    groups = defaultdict(list)
    for entry in entries:
        groups[entry['dataset']].append(entry)
    donors = {}
    for rows in groups.values():
        rows.sort(key=lambda r:digest(['v16-query-donor', r['row_hash']]))
        for i, row in enumerate(rows):
            donor = next((rows[(i+j)%len(rows)] for j in range(1,len(rows))
                          if rows[(i+j)%len(rows)]['image_sha256'] != row['image_sha256']), None)
            if donor is not None:
                donors[row['row_hash']] = donor['row_hash']
    return donors


def teacher_probe(study, model, output, limit=64):
    import torch
    from learned_score.mlsys_scorer.checkpoints import load_scorer
    from learned_score.mlsys_scorer.backbone import frozen_versions, assert_frozen
    from .benchmark import require_completed_checkpoint
    if type(limit) is not int or not 2 <= limit <= 64:
        raise ValueError('teacher probe is bounded to 2..64 training-cache rows')
    if not Path(output).resolve().is_relative_to(Path(study['output_root']).resolve()):
        raise ValueError('teacher report must remain in the v16 campaign namespace')
    torch.set_num_threads(2)
    checkpoint = Path(study['models'][model]['checkpoint'])
    scorer, trained = load_scorer(checkpoint, device='cpu')
    require_completed_checkpoint(checkpoint, trained)
    scorer.eval().requires_grad_(False)
    version = frozen_versions(scorer)
    directory = Path(study['cwd'])/TRAINING_CACHE[model]
    paths = sorted(directory.glob('*.pt'), key=lambda p:p.stem)[:limit]
    if len(paths) != limit:
        raise ValueError('training cache does not cover diagnostic panel')
    records, pins = {}, {str(checkpoint):file_digest(checkpoint), str(Path(__file__)):file_digest(__file__)}
    for path in (checkpoint.parent/'complete.json', Path(__file__).parents[1]/'mlsys_scorer/models.py',
                 Path(__file__).parents[1]/'mlsys_scorer/checkpoints.py',
                 Path(__file__).parents[1]/'multimodel/objectives.py'):
        pins[str(path.resolve())] = file_digest(path)
    for path in paths:
        record = torch.load(path, map_location='cpu', weights_only=True)
        if (record.get('format') != 'asymvlm_mlsys_saliency_v1' or record.get('row_hash') != path.stem or
                record.get('backbone') != trained['identity']['backbone'] or
                record.get('manifest_sha256') != trained['identity']['manifest_sha256']):
            raise ValueError('teacher cache differs from frozen scorer training provenance')
        records[path.stem] = record
        pins[str(path)] = file_digest(path)
    donors = deterministic_donors(list(records.values()))
    results = []
    for key, record in records.items():
        if key not in donors:
            # Do not cross benchmark boundaries just to manufacture a donor.
            results.append({'row_hash':key, 'dataset':record['dataset'], 'status':'no_same_dataset_different_image_donor'})
            continue
        with torch.no_grad():
            correct = scorer(record['vision'], record['question'])
            donor_question = records[donors[key]]['question']
            shuffled = scorer(record['vision'], donor_question)
            from torch.nn import functional as F
            # Before-decoder embeddings are identical for identical token IDs.
            # This identifies shared tokens without recovering any text/answer.
            question = record['question'].float()
            common = (question[:,None,:] == donor_question.float()[None,:,:]).all(-1).any(-1)
            projected_visual = F.normalize(F.linear(F.normalize(record['vision'].float(),dim=-1),
                scorer.visual.weight.float()),dim=-1)
            projected_query = F.normalize(F.linear(F.normalize(question,dim=-1),
                scorer.question.weight.float()),dim=-1)
            winners = (projected_visual@projected_query.T).argmax(-1)
        results.append({'row_hash':key, 'dataset':record['dataset'], 'status':'ok', 'donor_row_hash':donors[key],
                        'shared_question_token_fraction':float(common.float().mean()),
                        'visual_argmax_shared_question_fraction':float(common[winners].float().mean()),
                        **teacher_diagnostics(record['saliency'], correct, shuffled)})
        assert_frozen(scorer, version)
    valid = [r for r in results if r['status'] == 'ok']
    summary = {'cached_rows':len(paths), 'measured_rows':len(valid),
        'cached_zero_target_fraction':statistics.mean(r['cached_saliency_all_zero'] for r in valid) if valid else None,
        'cached_constant_target_fraction':statistics.mean(r['cached_saliency_constant'] for r in valid) if valid else None}
    for key in ('teacher_scorer_spearman','correct_shuffled_spearman','teacher_to_student_kl','cached_saliency_std',
                'shared_question_token_fraction','visual_argmax_shared_question_fraction'):
        values = [r[key] for r in valid if r[key] is not None]
        summary[key] = {'n':len(values), 'mean':statistics.mean(values) if values else None}
    summary['retentions'] = {str(p):{key:statistics.mean(r['retentions'][str(p)][key] for r in valid)
        for key in ('correct_vs_shuffled_overlap','learned_retained_teacher_target_mass',
                    'shuffled_retained_teacher_target_mass','teacher_rank_retained_target_mass')}
        for p in (30,50,70)} if valid else {}
    payload = execution.sealed({'format':FORMAT+'_teacher', 'study_sha256':study['sha256'], 'model_tag':model,
        'training_data_diagnostic_not_evaluation':True, 'selection':'first64_lexical_row_sha256',
        'scorer_frozen':True, 'backbone_loaded':False, 'training_normalization_floor':1e-6,
        'answers_read_from_cache':False, 'teacher_uses_training_answer_gradients':True,
        'cached_saliency_definition':'z_normalize(ReLU(-answer_CE_key_bias_gradient))',
        'absolute_raw_gradient_magnitude_not_available':True,
        'input_hashes':pins, 'summary':summary, 'rows':results})
    write_immutable(Path(output)/'report.json', payload)
    return payload


def probe_panel(entries, maximum_requests=32):
    """Equal-dataset SHA prefix; keep MME image pairs intact."""
    from .compression_policy_v10 import canonical_dataset
    if maximum_requests != 32:
        raise ValueError('predeclared probe panel is at most 32 requests')
    grouped = defaultdict(lambda:defaultdict(list))
    for row in entries:
        family = canonical_dataset(row['dataset'])
        grouped['ChartQA' if family == 'ChartQA_TEST' else family][row['group_id']].append(row)
    chosen = []
    for dataset in ('DocVQA','TextVQA','ChartQA','MME'):
        used = 0
        for group, rows in sorted(grouped[dataset].items(), key=lambda kv:digest(['v16-probe',kv[0]])):
            rows = sorted(rows,key=lambda r:r['row_hash'])
            if dataset == 'MME' and len(rows) != 2:
                raise ValueError('MME probe must preserve full pairs')
            take = rows if dataset == 'MME' else rows[:1]
            if used+len(take) > 8:
                break
            chosen.extend(take)
            used += len(take)
    donors = deterministic_donors(chosen)
    if set(donors) != {r['row_hash'] for r in chosen}:
        raise ValueError('probe has no same-dataset different-image query donor')
    return sorted(chosen,key=lambda r:(r['dataset'],r['row_hash']))


def build_probe_task(study, model, split='screening', maximum_requests=32):
    if split not in ('fit','screening'):
        raise ValueError('query/spatial diagnostics are development-only')
    entries = execution.read(study['development_manifests'][model])['splits'][split]
    panel = probe_panel(entries, maximum_requests)
    task = execution.build_task(study, 'physical', model, split,
        output_root=Path(study['output_root'])/model/'dev-controls',
        request_ids=[r['row_hash'] for r in panel], retentions=PERCENTAGES, warmup=0)
    task.pop('sha256')
    task.update(probe_format=FORMAT, probe_modes=list(MODES if model == 'qwen25' else MODES[:2]),
        probe_donors=deterministic_donors(panel), probe_not_candidate_selection=True)
    from .compression_policy_v10 import canonical_dataset
    task['probe_dataset_counts'] = {domain:sum(
        canonical_dataset(r['dataset']) == ('ChartQA_TEST' if domain == 'ChartQA' else domain) for r in panel)
        for domain in ('DocVQA','TextVQA','ChartQA','MME')}
    for path in (Path(__file__),Path(__file__).parents[1]/'mlsys_runtime/spatial.py'):
        task['input_hashes'][str(path.resolve())] = file_digest(path)
    return execution.sealed(task)


@contextmanager
def installed_probe(task, mode, shared):
    """Scoped in-process hooks; historical files and model parameters untouched."""
    import torch
    from . import adaptive_experiment_v3 as prior
    from .adaptive_benchmark_v1 import select_exact_count
    from .selector_experiment_v5 import validate_indices
    old_load, old_select, old_knots = prior.load_runtime, execution.make_selector, execution.knot_cells
    def load(args, manifest):
        if 'runtime' not in shared:
            shared['runtime'] = old_load(args, manifest)
            shared['entries'] = {r['row_hash']:r for r in manifest.splits[task['split']]}
            shared['donor_dataset'] = execution.answer_free_dataset(task, manifest)
            shared['donor_questions'] = {}
        return shared['runtime']
    def selector(cell, scorer, feature, observed):
        # Full-token baseline remains the same and does not need a query.
        if mode == 'correct' or cell['selector'] == 'full':
            return old_select(cell, scorer, feature, observed)
        adapter = shared['runtime'][0]
        count = cell['selected_count']
        internal = prior.Cell('scorer','uniform',0)
        @torch.no_grad()
        def select(prepared):
            vision = prepared.inputs_embeds[0,prepared.visual_positions]
            question = prepared.inputs_embeds[0,prepared.semantic_text_positions]
            if (len(vision),prepared.inputs_embeds.shape[1]-len(vision)) != (
                    feature['input_visual_tokens'],feature['input_text_tokens']):
                raise ValueError('probe selection geometry changed')
            spatial_metadata = None
            if mode == 'shuffled':
                request = shared['active_request']
                donor = task['probe_donors'][request]
                if donor not in shared['donor_questions']:
                    row = shared['donor_dataset'].row_for_entry(shared['entries'][donor])
                    # Existing label-aware prompt builder accepts this answer-free row.
                    donor_prepared = adapter.prepare_sample(row, include_answer=False)
                    shared['donor_questions'][donor] = donor_prepared.inputs_embeds[
                        0,donor_prepared.semantic_text_positions].detach().clone()
                    del donor_prepared
                question = shared['donor_questions'][donor]
                observed['scorer_query_donor'] = donor
                original = prepared.inputs_embeds[0,prepared.semantic_text_positions]
                observed['scorer_query_embeddings_identical'] = torch.equal(original,question)
                indices = select_exact_count(scorer(vision,question),vision,count)
            else:
                from learned_score.mlsys_runtime.spatial import qwen_patch_coordinates, select_spatial
                coordinates, shape = qwen_patch_coordinates(prepared)
                # The old helper floors n*r. A half-token interior value maps
                # unambiguously to this exact integer; verify rather than assume.
                retention = 1. if count == len(vision) else (count+.5)/len(vision)
                indices, spatial_metadata = select_spatial(scorer(vision,question),coordinates,shape,
                    retention=retention,fraction=.2,regions=4)
            validate_indices(indices,count,len(vision))
            values = indices.cpu().tolist()
            observed.update(selected_patch_indices=values,selected_patch_indices_sha256=digest(values),
                input_visual_tokens=len(vision),input_text_tokens=prepared.inputs_embeds.shape[1]-len(vision),
                retained_visual_tokens=count,visual_retention=count/len(vision),
                diagnostic_mode=mode,generation_query_unchanged=True,spatial=spatial_metadata)
            return indices
        return select,internal
    # Capture request identity before source loading; donor construction does not
    # call this label-dataset method and cannot overwrite the active request.
    original_row = None
    def bound_load(args,manifest):
        nonlocal original_row
        loaded = load(args,manifest)
        dataset = loaded[1]
        original_row = dataset.row_for_entry
        def row_for_entry(entry):
            shared['active_request'] = entry['row_hash']
            return original_row(entry)
        dataset.row_for_entry = row_for_entry
        return loaded
    def knots(*args,**kwargs):
        return [c for c in old_knots(*args,**kwargs) if c['selector'] != 'full']
    prior.load_runtime,execution.make_selector,execution.knot_cells = bound_load,selector,knots
    try:
        yield
    finally:
        if original_row is not None:
            shared['runtime'][1].row_for_entry = original_row
        prior.load_runtime,execution.make_selector,execution.knot_cells = old_load,old_select,old_knots


def physical_probe(task, rank=0, world_size=1):
    if task.get('probe_format') != FORMAT or task.get('split') not in ('fit','screening'):
        raise ValueError('sealed development diagnostic task required')
    execution.validate_task(task,'physical')
    expected = list(MODES if task['model_tag'] == 'qwen25' else MODES[:2])
    if task.get('probe_modes') != expected:
        raise ValueError('diagnostic mode menu changed')
    shared, markers = {}, {}
    for mode in expected:
        child = {k:v for k,v in task.items() if k != 'sha256'}
        child['output_root'] = str(Path(task['output_root'])/mode)
        child['diagnostic_mode'] = mode
        with installed_probe(task,mode,shared):
            markers[mode] = execution.run(execution.sealed(child),'physical',rank,world_size)
    marker = execution.sealed({'format':FORMAT+'_complete','task_sha256':task['sha256'],
        'passed':True,'modes':{m:v['sha256'] for m,v in markers.items()},
        'counts':list(PERCENTAGES),'generation_query_unchanged':True,
        'timing_is_diagnostic_and_includes_donor_preparation':True})
    write_immutable(Path(task['output_root'])/f'rank{rank}.complete.json',marker)
    return marker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('teacher','physical'))
    parser.add_argument('--study',type=Path)
    parser.add_argument('--model',choices=tuple(TRAINING_CACHE))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--task',type=Path)
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--world-size',type=int,default=1)
    args = parser.parse_args(argv)
    if args.phase == 'teacher':
        if not all((args.study,args.model,args.output)):
            parser.error('teacher requires study, model, and output')
        report = teacher_probe(execution.read(args.study),args.model,args.output)
        print(json.dumps(report['summary']),flush=True)
    else:
        if args.task is None:
            parser.error('physical requires task')
        physical_probe(execution.read(args.task),args.rank,args.world_size)


if __name__ == '__main__':
    main()
