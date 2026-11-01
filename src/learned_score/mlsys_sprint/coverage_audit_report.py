"""Paired, source-grouped reports; incomplete cells never become zero scores."""
from collections import defaultdict
import statistics

import numpy as np


def group_ci(pairs, *, draws=5000):
    groups=defaultdict(list)
    for candidate,reference in pairs:
        groups[candidate['group_id']].append(100*(candidate['quality_score']-reference['quality_score']))
    values=np.array([statistics.mean(groups[g]) for g in sorted(groups)])
    if len(values)<2:return None
    rng=np.random.default_rng(20261025)
    samples=values[rng.integers(0,len(values),size=(draws,len(values)))].mean(1)
    return [float(v) for v in np.quantile(samples,[.025,.975])]


def summarize(pairs, model, dataset, method, reference):
    candidates=[r for r,_ in pairs];controls=[r for _,r in pairs]
    cache=lambda row,key:row['metrics']['cache'][key]
    return {'model':model,'dataset':dataset,'method':method,'reference':reference,
        'requests':len(pairs),'source_groups':len({r['group_id'] for r in candidates}),
        'quality_metric':('MME_question_accuracy_diagnostic_not_official_total' if dataset=='MME' else candidates[0]['quality_metric']),
        'accuracy_percent':100*statistics.mean(r['quality_score'] for r in candidates),
        'reference_accuracy_percent':100*statistics.mean(r['quality_score'] for r in controls),
        'paired_delta_pp':100*statistics.mean(a['quality_score']-b['quality_score'] for a,b in pairs),
        'paired_group_bootstrap_ci95_pp':group_ci(pairs),
        'mean_retained_image_tokens':statistics.mean(r['selected_count'] for r in candidates),
        'aggregate_image_retention_percent':100*sum(r['selected_count'] for r in candidates)/sum(r['input_visual_tokens'] for r in candidates),
        'mean_per_request_image_retention_percent':100*statistics.mean(r['selected_count']/r['input_visual_tokens'] for r in candidates),
        'mean_kv_capacity_mib':statistics.mean(cache(r,'kv_capacity_bytes') for r in candidates)/1024**2,
        'mean_total_owned_cache_mib':statistics.mean(cache(r,'total_owned_bytes') for r in candidates)/1024**2,
        'aggregate_kv_reduction_vs_reference_percent':100*(1-sum(cache(r,'kv_capacity_bytes') for r in candidates)/sum(cache(r,'kv_capacity_bytes') for r in controls)),
        'mean_request_kv_reduction_vs_reference_percent':100*statistics.mean(1-cache(a,'kv_capacity_bytes')/cache(b,'kv_capacity_bytes') for a,b in pairs),
        'requests_with_image_compression_percent':100*statistics.mean(r['selected_count']<r['input_visual_tokens'] for r in candidates),
        'requests_with_kv_reduction_vs_reference_percent':100*statistics.mean(cache(a,'kv_capacity_bytes')<cache(b,'kv_capacity_bytes') for a,b in pairs),
        'mean_end_to_end_seconds':statistics.mean(r['end_to_end_seconds'] for r in candidates),
        'mean_selector_seconds':statistics.mean(r['selector_seconds'] for r in candidates),
        'mean_ttft_seconds':statistics.mean(r['ttft_seconds'] for r in candidates),
        'latency_note':'includes live scoring/allocation; shared-node diagnostic, not isolated speedup claim',
        'fallback_percent':100*statistics.mean(r.get('fallback_reason') is not None for r in candidates)}


def comparison_tables(bykey):
    fixed,adaptive=[],[]
    cohorts=sorted({(r['model'],r['dataset']) for r in bykey.values()})
    for model,dataset in cohorts:
        request_ids=sorted({r['request_id'] for r in bykey.values() if (r['model'],r['dataset'])==(model,dataset)})
        for percent in (50,70):
            reference=f'importance_{percent:03d}';method=f'coverage_greedy_{percent:03d}'
            pairs=[]
            for key in request_ids:
                a=bykey.get((model,key,method));b=bykey.get((model,key,reference))
                if a is None or b is None:continue
                if a['selected_count']!=b['selected_count'] or a['metrics']['cache']['kv_capacity_bytes']!=b['metrics']['cache']['kv_capacity_bytes']:
                    raise ValueError('fixed-count comparison has unequal measured budgets')
                pairs.append((a,b))
            if dataset=='MME':pairs=complete_mme_pairs(pairs)
            if pairs:
                fixed.append(summarize(pairs,model,dataset,method,reference))
        methods=('importance_legacy_tau','importance_fine_tau','coverage_greedy_tau')
        # All adaptive rows use the SAME complete request cohort within domain.
        complete=[key for key in request_ids if all((model,key,m) in bykey for m in (*methods,'all_tokens'))]
        for method in methods:
            pairs=[(bykey[(model,key,method)],bykey[(model,key,'all_tokens')]) for key in complete]
            if dataset=='MME':pairs=complete_mme_pairs(pairs)
            if pairs:adaptive.append(summarize(pairs,model,dataset,method,'all_tokens'))
    return fixed,adaptive


def complete_mme_pairs(pairs):
    grouped=defaultdict(list)
    for pair in pairs:grouped[pair[0]['group_id']].append(pair)
    return [pair for group in sorted(grouped) if len(grouped[group])==2 for pair in grouped[group]]
