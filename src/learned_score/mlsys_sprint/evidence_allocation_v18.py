"""One fixed removed-weight limit; physical calibration and exact controls.

The scalar is not a probability of a wrong answer. It is the largest fraction
of the scorer's total importance weight that calibration permits discarding.
"""
from collections import defaultdict
import math


def _quality(row):
    value=row.get('quality_score')
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError('physical native quality must be finite and normalized to[0,1]')
    return value


def _same_request(feature,row):
    for key in ('request_id','dataset','group_id','image_sha256','input_visual_tokens',
                'input_text_tokens','cache_contract'):
        if key not in feature or key not in row or feature[key]!=row[key]:
            raise ValueError('physical feature/request binding differs: '+key)
    n,p=feature['input_visual_tokens'],feature['input_text_tokens']
    k=row.get('selected_count')
    contract=feature['cache_contract']
    cost,reserve=contract.get('total_owned_bytes_per_slot'),contract.get('reserved_generated_slots')
    if (any(type(v) is not int for v in (n,p,k,cost,reserve)) or
            n<1 or p<0 or not 1<=k<=n or cost<1 or reserve<0):
        raise ValueError('invalid native token/cache geometry')
    if row.get('expected_cache_bytes')!=(k+p+reserve)*cost:
        raise ValueError('physical KV cost differs from assigned native count')
    _quality(row)


def choose_count(curve, tau):
    if not math.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError('removed-weight limit must be in [0,1]')
    risk, candidates = curve['removed_weight_by_count'], curve['candidate_counts']
    n = len(risk)-1
    if (n<1 or candidates!=sorted(set(candidates)) or not candidates or candidates[-1]!=n
            or any(type(k) is not int or not 1<=k<=n for k in candidates)
            or any(not math.isfinite(r) or not 0<=r<=1 for r in risk)
            or any(a+1e-10<b for a,b in zip(risk,risk[1:])) or risk[-1]!=0):
        raise ValueError('finite monotone risk and full-token endpoint required')
    return next((k for k in candidates if risk[k]<=tau),n)


def exact_uniform(features, total):
    """Common retained fraction plus deterministic integer reconciliation.

    Fractional rounding uses integer arithmetic and identity-only tie breaking;
    no labels or per-request risk influence the matched uniform comparator.
    """
    rows=sorted(features,key=lambda f:f['request_id'])
    if not rows or len({f['request_id'] for f in rows})!=len(rows):
        raise ValueError('nonempty unique request panel required')
    sizes=[f['input_visual_tokens'] for f in rows]
    if any(type(n) is not int or n<1 for n in sizes) or type(total) is not int:
        raise ValueError('integer native and aggregate counts required')
    denominator=sum(sizes)
    if not len(rows)<=total<=denominator:
        raise ValueError('aggregate outside feasible count range')
    # Lower bound one matters only for synthetic/extreme cases, not50-100%pilot.
    counts=[max(1,total*n//denominator) for n in sizes]
    remainders=[total*n%denominator for n in sizes]
    difference=total-sum(counts)
    order=sorted(range(len(rows)),key=lambda i:(-remainders[i],rows[i]['request_id']))
    while difference>0:
        changed=False
        for i in order:
            if counts[i]<sizes[i]:
                counts[i]+=1; difference-=1; changed=True
                if not difference: break
        if not changed: raise ValueError('cannot reconcile uniform upward')
    while difference<0:
        changed=False
        for i in reversed(order):
            if counts[i]>1:
                counts[i]-=1; difference+=1; changed=True
                if not difference: break
        if not changed: raise ValueError('cannot reconcile uniform downward')
    return {f['request_id']:k for f,k in zip(rows,counts)}


def matched_assignments(features, checkpoint_sha, tau):
    adaptive={f['request_id']:choose_count(f['curves'][checkpoint_sha],tau) for f in features}
    by_domain=defaultdict(list)
    for f in features: by_domain[f['dataset']].append(f)
    uniform={}
    for rows in by_domain.values():
        # Native Qwen contract is identical within each domain; check rather than
        # equating token counts across different KV architectures.
        contracts={f['cache_contract']['total_owned_bytes_per_slot'] for f in rows}
        if len(contracts)!=1: raise ValueError('cannot match tokens across cache contracts')
        uniform.update(exact_uniform(rows,sum(adaptive[f['request_id']] for f in rows)))
    return adaptive,uniform


def calibrate(features, results, checkpoint_sha):
    """Enumerate101limits using new-scorer physical labels, never interpolation."""
    if not features or len({f['request_id'] for f in features})!=len(features):
        raise ValueError('unique nonempty calibration features required')
    by_id={f['request_id']:f for f in features}
    physical={}; full={}; expected=set()
    for feature in features:
        curve=feature['curves'][checkpoint_sha]
        choose_count(curve,0.)
        if len(curve['removed_weight_by_count'])-1!=feature['input_visual_tokens']:
            raise ValueError('risk curve length differs from native image-token count')
        expected.update((feature['request_id'],k) for k in curve['candidate_counts']
                        if k<feature['input_visual_tokens'])
    for r in results:
        if r.get('status')!='ok' or r.get('physical_measurement_performed') is not True:
            raise ValueError('physical successful calibration measurements required')
        if r.get('request_id') not in by_id:
            raise ValueError('physical calibration request outside frozen feature panel')
        feature=by_id[r['request_id']]
        _same_request(feature,r)
        key=(r['request_id'],r['selected_count'])
        if r['selector']=='full':
            if r['request_id'] in full:
                raise ValueError('duplicate full-token calibration measurement')
            if r['selected_count']!=feature['input_visual_tokens']:
                raise ValueError('full-token reference is not actually full')
            full[r['request_id']]=r
        elif r.get('selector')=='scorer' and r.get('checkpoint_sha256')==checkpoint_sha:
            if key in physical: raise ValueError('duplicate scorer calibration measurement')
            if key not in expected and key[1]!=feature['input_visual_tokens']:
                raise ValueError('calibration measurement outside declared physical candidate menu')
            physical[key]=r
        else:
            raise ValueError('calibration labels must come from the new scorer checkpoint')
    if set(full)!=set(by_id) or expected-set(physical):
        raise ValueError('calibration requires the complete frozen request/candidate matrix')
    for key,r in physical.items():
        if key[1]==by_id[key[0]]['input_visual_tokens'] and r['quality_score']!=full[key[0]]['quality_score']:
            raise ValueError('new scorer full-token quality differs from native full baseline')
    trials=[]
    for integer in range(101):
        tau=integer/100
        losses=defaultdict(list); bytes_used=0; counts={}
        for f in features:
            rid=f['request_id']; k=choose_count(f['curves'][checkpoint_sha],tau)
            baseline=full[rid]
            measured=baseline if k==f['input_visual_tokens'] else physical[(rid,k)]
            if measured['selected_count']!=k: raise ValueError('wrong physical calibration count')
            loss=100*(baseline['quality_score']-measured['quality_score'])
            if not math.isfinite(loss): raise ValueError('nonfinite native quality')
            losses[f['dataset']].append(loss)
            bytes_used+=measured['expected_cache_bytes']; counts[rid]=k
        domain_losses={d:sum(v)/len(v) for d,v in losses.items()}
        mean_loss=sum(domain_losses.values())/len(domain_losses)
        feasible=mean_loss<=1.+1e-10 and max(domain_losses.values())<=2.+1e-10
        trials.append(dict(tau=tau,mean_loss_pp=mean_loss,domain_loss_pp=domain_losses,
            aggregate_kv_bytes=bytes_used,counts=counts,feasible=feasible))
    feasible=[t for t in trials if t['feasible']]
    winner=min(feasible,key=lambda t:(t['aggregate_kv_bytes'],t['mean_loss_pp'],t['tau']))
    compressed=any(winner['counts'][f['request_id']]<f['input_visual_tokens'] for f in features)
    return dict(selected=winner,trials=trials,compression_feasible=compressed,
                rule='minimum_physical_KV_subject_to_full_relative_quality_limits',
                note='empirical_calibration_not_a_statistical_accuracy_guarantee')


def screening_gate(results, *, expected_request_ids=None):
    by_method=defaultdict(dict)
    methods=('current_070','control_070','evidence_070')
    for r in results:
        if r.get('method') not in methods:
            continue
        if r.get('status')!='ok':
            raise ValueError('screening gate cannot discard failed comparisons')
        if r['request_id'] in by_method[r['method']]:
            raise ValueError('duplicate screening request/method; audit repeated controls before gating')
        _quality(r)
        by_method[r['method']][r['request_id']]=r
    if not all(m in by_method for m in methods): raise ValueError('missing screening method')
    ids=set(by_method[methods[0]])
    if any(set(by_method[m])!=ids for m in methods): raise ValueError('unpaired screening panel')
    if expected_request_ids is not None and (len(expected_request_ids)!=len(set(expected_request_ids)) or
            ids!=set(expected_request_ids)):
        raise ValueError('screening gate requires all predeclared request identities')
    scores={m:defaultdict(list) for m in methods}
    for rid in ids:
        reference=by_method[methods[0]][rid]
        for m in methods:
            r=by_method[m][rid]
            for key in ('dataset','group_id','image_sha256','input_visual_tokens','input_text_tokens','cache_contract'):
                if key not in reference or key not in r or r[key]!=reference[key]:
                    raise ValueError('screening request geometry differs: '+key)
            if r['selected_count']!=reference['selected_count']:
                raise ValueError('screening scorers must retain identical integer counts')
            scores[m][r['dataset']].append(100*r['quality_score'])
    for method in methods:
        checkpoints={r.get('checkpoint_sha256') for r in by_method[method].values()}
        if len(checkpoints)!=1 or None in checkpoints:
            raise ValueError('each screening method requires exactly one frozen scorer checkpoint')
    means={m:{d:sum(v)/len(v) for d,v in groups.items()} for m,groups in scores.items()}
    # Stronger control is chosen once by overall balanced-domain mean, not
    # independently per request. Domain checks compare against that same arm.
    stronger=max(methods[:2],key=lambda m:(sum(means[m].values())/len(means[m]),m))
    deltas={d:means['evidence_070'][d]-means[stronger][d] for d in means[stronger]}
    mean_delta=sum(deltas.values())/len(deltas)
    return dict(passed=mean_delta>=1.-1e-10 and min(deltas.values())>=-1.-1e-10,
        mean_delta_pp=mean_delta,domain_delta_pp=deltas,stronger_control=stronger,means=means)
