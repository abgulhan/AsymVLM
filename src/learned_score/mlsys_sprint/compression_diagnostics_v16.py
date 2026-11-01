"""Physical-only, development-only failure decomposition for coverage stopping.

Net gains and positive harms are separate.  Correlations are stratified by
retention so that compression itself cannot manufacture apparent risk signal.
Oracle results use answers and are never deployable or promoted as methods.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from . import compression_policy_v16 as policy
from .compression_matrix_v8 import seal
from .followup_common import write_immutable
from .state import digest

FORMAT = 'mlsys_compression_diagnostics_v16'
SHUFFLE_SEEDS = (20261025, 20261026, 20261027)


def _ranks(values):
    values = np.asarray(values,dtype=float)
    order = np.argsort(values,kind='stable')
    result = np.empty(len(values),dtype=float)
    start = 0
    while start < len(values):
        end = start+1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start+end-1)/2
        start = end
    return result


def _correlation(x,y,*,rank=False):
    if len(x) < 3:
        return None
    x,y = np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if rank:
        x,y = _ranks(x),_ranks(y)
    if np.std(x)<1e-12 or np.std(y)<1e-12:
        return None
    return float(np.corrcoef(x,y)[0,1])


def _auc(x,harm):
    positive = np.asarray(harm,dtype=bool)
    npos,nneg = int(positive.sum()),int((~positive).sum())
    if not npos or not nneg:
        return None
    ranks = _ranks(x)+1
    return float((ranks[positive].sum()-npos*(npos+1)/2)/(npos*nneg))


def conditional_signal(rows,kind):
    """Within fixed declared retention knots; native MME pair effects separate.

    Correlation targets here are per-question physical loss. MME quality gates
    elsewhere use the native paired/category metric, not these correlations.
    """
    data = policy.transformed_rows(rows,kind)
    observations = []
    for row in data:
        curve = row['curves']['scorer']
        if policy.base.previous._invalid_curve(curve):
            continue
        n = row['input_visual_tokens']
        full = row['quality_by_count'][str(n)]
        risks = curve['missing_mass' if kind=='missing_mass' else 'uncovered_mass']
        for k,risk in zip(curve['counts'],risks):
            if k==n:
                continue
            # The physical menus are ceil(N*p/100); recover the declared knot
            # where unique, otherwise use exact (N,k), never blur retention.
            knots = [p for p in range(10,100,10) if (n*p+99)//100==k]
            knot = f'{knots[0]:03d}%' if len(knots)==1 else f'count={k}/N={n}'
            loss = full-row['quality_by_count'][str(k)]
            observations.append({'group_id':row['group_id'],'row_hash':row['row_hash'],
                'stratum':(row['dataset'],knot),'risk':risk,'loss':loss,'harm':max(0.,loss)})
    strata = defaultdict(list)
    for value in observations:
        strata[value['stratum']].append(value)
    summaries = []
    residual_x,residual_y = [],[]
    for (dataset,knot),values in sorted(strata.items()):
        x = np.array([v['risk'] for v in values]); y = np.array([v['loss'] for v in values]); h=np.maximum(y,0)
        residual_x.extend(x-x.mean()); residual_y.extend(y-y.mean())
        summaries.append({'dataset':dataset,'retention_knot':knot,'requests':len(values),
            'independent_groups':len({v['group_id'] for v in values}),
            'mean_loss_pp':float(y.mean()*100),'positive_harm_pp':float(h.mean()*100),
            'positive_benefit_pp':float(np.maximum(-y,0).mean()*100),
            'harmed_requests':int((y>1e-12).sum()),'helped_requests':int((y < -1e-12).sum()),
            'risk_loss_spearman':_correlation(x,y,rank=True),
            'risk_positive_harm_spearman':_correlation(x,h,rank=True),
            'harm_detection_auc':_auc(x,y>1e-12)})
    folds = policy.base.grouped_folds(rows,5,20261025)
    errors,base_errors = [],[]
    fold_records=[]
    for fold in range(5):
        train=[v for v in observations if folds[v['group_id']]!=fold]
        valid=[v for v in observations if folds[v['group_id']]==fold]
        groups=defaultdict(list)
        for v in train:
            groups[v['stratum']].append(v)
        means={key:(float(np.mean([v['risk'] for v in values])),float(np.mean([v['harm'] for v in values])))
               for key,values in groups.items()}
        x=np.array([v['risk']-means[v['stratum']][0] for v in train])
        y=np.array([v['harm']-means[v['stratum']][1] for v in train])
        denominator=float(x@x)
        slope=max(0.,float(x@y)/denominator) if denominator>1e-24 else 0.
        fold_errors,fold_base=[],[]
        skipped=0
        for v in valid:
            if v['stratum'] not in means:
                skipped+=1
                continue
            mean_x,mean_y=means[v['stratum']]
            prediction=np.clip(mean_y+slope*(v['risk']-mean_x),0,1)
            fold_errors.append(float((prediction-v['harm'])**2))
            fold_base.append(float((mean_y-v['harm'])**2))
        errors.extend(fold_errors);base_errors.extend(fold_base)
        fold_records.append({'fold':fold,'nonnegative_risk_slope':slope,'validation_observations':len(fold_errors),
                             'missing_training_stratum_observations':skipped})
    return {'kind':kind,'within_retention':summaries,
            'dataset_and_retention_demeaned_pearson':_correlation(residual_x,residual_y),
            'oof_positive_harm_mse':float(np.mean(errors)) if errors else None,
            'oof_retention_only_mse':float(np.mean(base_errors)) if base_errors else None,
            'oof_relative_mse_reduction':1-sum(errors)/sum(base_errors) if sum(base_errors)>1e-24 else None,
            'folds':fold_records,'scope':'per_question_physical_loss_not_MME_paired_native_score',
            'no_quality_interpolation':True,'training_and_validation_groups_disjoint':True}


def leave_one_domain_out(rows,kind,*,model,bytes_per_token,generation_reserve=127,profile='primary'):
    records=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        train=[r for r in rows if r['dataset']!=dataset]
        valid=[r for r in rows if r['dataset']==dataset]
        rule=policy.fit_rule(train,kind,model=model,bytes_per_token=bytes_per_token,
                             generation_reserve=generation_reserve,profile=profile)
        output=policy.infer(rule,policy.answer_free(valid))
        records.append({'omitted_dataset':dataset,'threshold':rule['threshold'],
                        'training_datasets':sorted({r['dataset'] for r in train}),
                        'fallbacks':output['fallbacks'],
                        'summary':policy.base.summary(valid,output['counts'],bytes_per_token=bytes_per_token,
                                                      generation_reserve=generation_reserve)})
    return {'kind':kind,'profile':profile,'domains':records,
            'evidence':'previously_examined_development_domain_transfer_not_confirmation'}


def shuffle_assignments(rows,counts,*,seed):
    if set(counts)!={r['row_hash'] for r in rows}:
        raise ValueError('shuffle cohort differs from assignment cohort')
    strata=defaultdict(list)
    for row in rows:
        strata[(row['dataset'],row['input_visual_tokens'])].append(row['row_hash'])
    result={}
    for stratum,keys in sorted(strata.items()):
        keys=sorted(keys)
        donors=sorted(keys,key=lambda key:digest(['v16-shuffle',seed,stratum,key]))
        result.update({key:counts[donor] for key,donor in zip(keys,donors)})
    lookup={r['row_hash']:r for r in rows}
    if any(str(k) not in lookup[key]['quality_by_count'] for key,k in result.items()):
        raise ValueError('shuffle requires missing physical outcome; no interpolation allowed')
    for dataset in {r['dataset'] for r in rows}:
        keys=[r['row_hash'] for r in rows if r['dataset']==dataset]
        if sum(result[k] for k in keys)!=sum(counts[k] for k in keys):
            raise AssertionError('shuffle lost exact within-domain token conservation')
    return result


def shuffled_controls(rows,counts,*,bytes_per_token,generation_reserve=127):
    values=[]
    for seed in SHUFFLE_SEEDS:
        shuffled=shuffle_assignments(rows,counts,seed=seed)
        changed=sum(shuffled[k]!=counts[k] for k in counts)
        values.append({'seed':seed,'changed_assignments':changed,'total_assignments':len(counts),
                       'percent_assignments_changed':100*changed/len(counts),'counts':shuffled,
                       'summary':policy.base.summary(rows,shuffled,bytes_per_token=bytes_per_token,
                                                     generation_reserve=generation_reserve)})
    return {'seeds':list(SHUFFLE_SEEDS),'strata':['dataset','input_visual_tokens'],
            'exact_aggregate_KV_preserved_within_each_dataset':True,
            'adaptive_summary':policy.base.summary(rows,counts,bytes_per_token=bytes_per_token,
                                                   generation_reserve=generation_reserve),'controls':values}


def oracle_headroom(rows,*,retentions=(50,70,90)):
    from . import compression_diagnostics_v11 as oracle
    results=[]
    for dataset in sorted({r['dataset'] for r in rows}):
        group=[r for r in rows if r['dataset']==dataset]
        actions={objective:oracle.make_actions(group,objective) for objective in ('max_quality','min_positive_harm')}
        for retention in retentions:
            # A physical uniform knot defines the budget. Sparse Phi menus may
            # use a lower available count, explicitly recorded in counts.
            counts={r['row_hash']:max((int(k) for k in r['quality_by_count']
                        if int(k)<=(r['input_visual_tokens']*retention+99)//100),
                        default=min(map(int,r['quality_by_count']))) for r in group}
            budget=sum(counts.values())
            results.append({'dataset':dataset,'target_retention_percent':retention,
                'uniform_physical_counts':counts,'uniform_summary':policy.base.previous.summary(group,counts),
                'answer_informed':{objective:oracle.solve_oracle(values,budget,exact_group_limit=24,
                    max_states=10000,iterations=40) for objective,values in actions.items()}})
    return {'results':results,'answer_informed_not_deployable':True,
            'quality_monotonicity_assumed':False,'interpolated_quality':False}


def fixed_count_selector_comparison(rows,physical,*,selector='divprune'):
    """Compare only unique, physically evaluated replacement selections.

    Legacy duplicate-index DivPrune results are rejected. New worker artifacts
    must expose selected_patch_indices, allowing validity to be checked rather
    than trusting a method name or unsupported 'corrected' flag.
    """
    if selector not in ('divprune','cdpruner'):
        raise ValueError('undeclared alternative selector')
    index={}
    for value in physical:
        if value.get('selector')!=selector:
            continue
        key=value.get('request_id',value.get('row_hash'))
        count=value['selected_count']
        indices=value.get('selected_patch_indices')
        if (not isinstance(indices,list) or len(indices)!=count or len(set(indices))!=count
                or indices!=sorted(indices) or any(type(i)is not int or not 0<=i<value['input_visual_tokens'] for i in indices)):
            raise ValueError('alternative selector physical indices are absent, duplicated, or invalid')
        if (key,count) in index:
            raise ValueError('duplicate alternative physical cell')
        index[key,count]=value
    groups=defaultdict(list)
    for row in rows:
        n=row['input_visual_tokens']
        for k,q in row['quality_by_count'].items():
            count=int(k)
            if count==n or (row['row_hash'],count) not in index:
                continue
            other=index[row['row_hash'],count]
            if other['input_visual_tokens']!=n or other['input_text_tokens']!=row['input_text_tokens']:
                raise ValueError('alternative selection geometry differs')
            knot=tuple(p for p in range(10,100,10) if (n*p+99)//100==count)
            groups[row['dataset'],knot].append((row,count,float(other['quality_score'])))
    results=[]
    for (dataset,knot),observations in sorted(groups.items()):
        selected=[copy_row_with_quality(row,count,quality) for row,count,quality in observations]
        counts={row['row_hash']:count for row,count,_ in observations}
        original=[row for row,_,_ in observations]
        if dataset=='MME':
            sizes=defaultdict(int)
            for row in original:
                sizes[row['group_id']]+=1
            complete={g for g,n in sizes.items() if n==2}
            original=[r for r in original if r['group_id'] in complete]
            selected=[r for r in selected if r['group_id'] in complete]
            counts={r['row_hash']:counts[r['row_hash']] for r in original}
        if not original:
            continue
        learned=policy.base.previous.summary(original,counts)
        alternative=policy.base.previous.summary(selected,counts)
        results.append({'dataset':dataset,'retention_knots':list(knot),'requests':len(original),
                        'learned_summary':learned,'alternative_summary':alternative,
                        'alternative_minus_learned_pp':100*(alternative['macro_accuracy']-learned['macro_accuracy'])})
    return {'selector':selector,'comparisons':results,'unique_indices_verified':True,
            'scope':'projected_feature_selector_adaptation_not_official_end_to_end_baseline'}


def copy_row_with_quality(row,count,quality):
    if not np.isfinite(quality) or not 0<=quality<=1:
        raise ValueError('finite physical quality in [0,1] required')
    return {**row,'quality_by_count':{**row['quality_by_count'],str(count):quality}}


def report(rows,*,model,bytes_per_token,generation_reserve=127,include_oracle=False):
    config={'model':model,'bytes_per_token':bytes_per_token,'generation_reserve':generation_reserve}
    kinds=policy.available_kinds(rows)
    output={'format':FORMAT,'model':model,'input_rows_sha256':digest(rows),
            'requests':len(rows),'source_groups':len({r['group_id'] for r in rows}),
            'domains':sorted({r['dataset'] for r in rows}),'available_kinds':kinds,
            'conditional_signal':{k:conditional_signal(rows,k) for k in kinds},
            'domain_transfer':{k:leave_one_domain_out(rows,k,**config) for k in kinds},
            'independent_confirmation':False,'evaluation_labels_opened':False,
            'evidence':'previously_examined_development_only'}
    if include_oracle:
        output['oracle']=oracle_headroom(rows)
    return seal(output)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--models',nargs='+',choices=policy.MODELS,default=list(policy.MODELS))
    parser.add_argument('--oracle',action='store_true')
    args=parser.parse_args(argv)
    for model in args.models:
        rows,provenance,config=policy.load_development(args.project,model)
        value=report(rows,model=model,include_oracle=args.oracle,**config)
        write_immutable(args.output/(model+'.json'),seal({'diagnostics':value,'provenance':provenance}))
        print(json.dumps({'model':model,'requests':len(rows),'output':str(args.output/(model+'.json'))}),flush=True)


if __name__=='__main__':
    main()
