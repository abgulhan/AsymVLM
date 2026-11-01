"""Bounded coverage repairs and source-grouped development selection.

This module only consumes physical, previously declared development outcomes.
It never reads evaluation labels or interpolates quality.  A shared definition
is selected across models, while each model receives its own scalar threshold.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from . import compression_policy_v12 as base
from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import digest, file_digest

FORMAT = 'mlsys_compression_policy_v16'
PROFILES = {'conservative': (.5, 1.), 'primary': (1., 2.), 'aggressive': (2., 3.)}
# Operation counts are a predeclared simplicity ordering, not measured latency.
CANDIDATES = {
    'missing_mass': {'curve': 'scorer', 'statistic': 'missing_mass', 'beta': 0., 'lambda': 0., 'operations': 0},
    'rank32_coverage': {'curve': 'scorer', 'statistic': 'uncovered_mass', 'beta': 0., 'lambda': 0., 'operations': 1},
    'native_coverage': {'curve': 'native', 'statistic': 'uncovered_mass', 'beta': 0., 'lambda': 0., 'operations': 2},
    'rank32_importance_floor': {'curve': 'scorer_floor010', 'statistic': 'uncovered_mass', 'beta': .1, 'lambda': 0., 'operations': 2},
    'rank32_representation_floor': {'curve': 'scorer', 'statistic': 'uncovered_mass', 'beta': 0., 'lambda': .1, 'operations': 2},
    'rank32_both_floors': {'curve': 'scorer_floor010', 'statistic': 'uncovered_mass', 'beta': .1, 'lambda': .1, 'operations': 3},
}
KINDS = tuple(CANDIDATES)
MODELS = ('qwen25', 'llava_next', 'phi35')


def _definition(kind):
    if kind not in CANDIDATES:
        raise ValueError('undeclared v16 rule family')
    return CANDIDATES[kind]


def transformed_rows(rows, kind, *, labels=True):
    """Expose one answer-free statistic through the exact, audited v12 sweep.

    Ranking and physical prefixes never change.  A weight-floor curve must be
    measured from the same frozen geometry; it cannot be reconstructed from
    missing mass alone because its uniform-weight coverage is also necessary.
    """
    definition = _definition(kind)
    if kind == 'native_coverage':
        rows = base.validate_rows(rows, kind, labels=labels)
    else:
        rows = base.validate_rows(rows, 'rank32_coverage', labels=labels)
    result = []
    for row in rows:
        original = row['curves']['scorer']
        curve = row['curves'].get(definition['curve'])
        if curve is None:
            raise ValueError('missing required feature curve: ' + definition['curve'])
        invalid = base.previous._invalid_curve(curve) or base.previous._invalid_curve(original)
        if not invalid and curve['counts'] != original['counts']:
            raise ValueError('repair changed the physical count menu')
        for field in ('selected_indices_sha256', 'ranking_sha256'):
            if not invalid and (field in curve or field in original):
                if curve.get(field) != original.get(field):
                    raise ValueError('repair changed the frozen patch ranking: ' + field)
        if definition['beta'] and not invalid:
            expected = [(1-definition['beta'])*mass + definition['beta']*(1-k/row['input_visual_tokens'])
                        for k, mass in zip(original['counts'], original['missing_mass'])]
            if any(abs(a-b) > 1e-6 for a, b in zip(expected, curve['missing_mass'])):
                raise ValueError('importance-floor mass does not match declared beta')
        adapted = copy.deepcopy(row)
        selected = copy.deepcopy(curve)
        if definition['lambda']:
            lam = definition['lambda']
            selected['uncovered_mass'] = [(1-lam)*u+lam*m for u,m in
                                          zip(curve['uncovered_mass'], curve['missing_mass'])]
        adapted['curves'] = {'scorer': selected}
        # Native geometry or importance-floor failures must not bypass a bad
        # learned ranker.  Propagate the frozen ranker's invalidity to inference.
        if base.previous._invalid_curve(original):
            selected.update(invalid=True, fallback=True, fallback_reason='invalid_frozen_selector')
        result.append(adapted)
    return base.validate_rows(result, 'rank32_coverage', labels=labels)


def answer_free(rows):
    return base.answer_free(rows)


def fit_rule(rows, kind, *, model, bytes_per_token, profile='primary', generation_reserve=127):
    if profile not in PROFILES:
        raise ValueError('undeclared quality-loss profile')
    data = transformed_rows(rows, kind)
    mean, worst = PROFILES[profile]
    primitive = 'missing_mass' if kind == 'missing_mass' else 'rank32_coverage'
    rule = base.fit_rule(data, primitive, model=model, bytes_per_token=bytes_per_token,
                         generation_reserve=generation_reserve,
                         max_mean_loss_pp=mean, max_domain_loss_pp=worst)
    value = {'format': FORMAT, 'kind': kind, 'definition': copy.deepcopy(_definition(kind)),
             'model': model, 'profile': profile, 'base_rule': rule,
             'threshold': rule['threshold'], 'bytes_per_token': bytes_per_token,
             'generation_reserve': generation_reserve, 'fit_summary': rule['fit_summary'],
             'physical_labels_only': True, 'scorer_changed': False,
             'source_rows_sha256': digest(rows), 'independent_confirmation': False}
    return seal(value)


def infer(rule, rows, *, allow_development_overlap=False):
    if (rule.get('format') != FORMAT or rule.get('sha256') != digest({k:v for k,v in rule.items() if k != 'sha256'})
            or rule.get('definition') != _definition(rule.get('kind'))):
        raise ValueError('v16 policy identity mismatch')
    if rule.get('legacy_retention_percentages') is not None:
        rows = restrict_menu(rows,rule['legacy_retention_percentages'])
    data = transformed_rows(rows, rule['kind'], labels=False)
    output = base.infer(rule['base_rule'], data, allow_development_overlap=allow_development_overlap)
    return {**output, 'policy_sha256': rule['sha256'], 'base_policy_sha256': rule['base_rule']['sha256'],
            'rule_family': rule['kind']}


def restrict_menu(rows,retentions):
    """Preserve the actual historical count menu when wrapping an incumbent."""
    if (not retentions or any(type(p)is not int or not 1<=p<=100 for p in retentions)
            or 100 not in retentions):
        raise ValueError('valid physical retention menu including full required')
    result=copy.deepcopy(rows)
    for row in result:
        n=row['input_visual_tokens']
        requested=sorted({(n*p+99)//100 for p in retentions})
        for curve in row['curves'].values():
            if base.previous._invalid_curve(curve):
                continue
            if not set(requested)<=set(curve['counts']):
                raise ValueError('legacy physical count menu absent from inference features')
            positions=[curve['counts'].index(count) for count in requested]
            for name in ('counts','missing_mass','uncovered_mass','removed_fraction'):
                curve[name]=[curve[name][i] for i in positions]
            if 'selected_indices_sha256' in curve:
                curve['selected_indices_sha256']={str(k):curve['selected_indices_sha256'][str(k)] for k in requested}
    return result


def available_kinds(rows):
    return [kind for kind, spec in CANDIDATES.items()
            if all(spec['curve'] in r['curves'] for r in rows)]


def development_report(rows, *, model, bytes_per_token, generation_reserve=127,
                       profile='primary', kinds=None, folds=5, seed=20261025,
                       required_domains=None):
    """OOF development comparison; the resulting selection is not confirmation."""
    if folds != 5:
        raise ValueError('v16 prespecifies five source-grouped folds')
    if profile not in PROFILES:
        raise ValueError('undeclared profile')
    kinds = tuple(kinds or available_kinds(rows))
    if not kinds or len(set(kinds)) != len(kinds) or not set(kinds) <= set(KINDS):
        raise ValueError('nonempty unique declared candidates required')
    required_domains = tuple(base.DATASETS if required_domains is None else required_domains)
    if not required_domains or not set(required_domains) <= set(base.DATASETS):
        raise ValueError('explicit supported development domains required')
    if set(r['dataset'] for r in rows) != set(required_domains):
        raise ValueError('all four development domains required for shared selection')
    partition = base.grouped_folds(rows, folds, seed)
    config = {'model': model, 'bytes_per_token': bytes_per_token,
              'generation_reserve': generation_reserve, 'profile': profile}
    candidates, final_rules = {}, {}
    for kind in kinds:
        transformed_rows(rows, kind)
        counts, records = {}, []
        for fold in range(folds):
            train = [r for r in rows if partition[r['group_id']] != fold]
            valid = [r for r in rows if partition[r['group_id']] == fold]
            if not train or not valid:
                raise ValueError('empty source-grouped fold')
            rule = fit_rule(train, kind, **config)
            assigned = infer(rule, answer_free(valid))
            counts.update(assigned['counts'])
            records.append({'fold': fold, 'fit_group_ids': rule['base_rule']['fit_group_ids'],
                            'validation_group_ids': sorted({r['group_id'] for r in valid}),
                            'rule_sha256': rule['sha256'], 'threshold': rule['threshold'],
                            'fallbacks': assigned['fallbacks'],
                            'summary': base.summary(valid, assigned['counts'], bytes_per_token=bytes_per_token,
                                                    generation_reserve=generation_reserve)})
        summary = base.summary(rows, counts, bytes_per_token=bytes_per_token, generation_reserve=generation_reserve)
        candidates[kind] = {'kind': kind, 'definition': CANDIDATES[kind], 'counts': counts,
                            'summary': summary, 'folds': records,
                            'eligible': base.meets_constraints(summary, *PROFILES[profile]),
                            'normalized_kv_capacity': summary['allocated_kv_bytes']/summary['full_allocated_kv_bytes']}
        final_rules[kind] = fit_rule(rows, kind, **config)
    return seal({'format': FORMAT+'_development', 'configuration': config,
                 'candidates': candidates, 'final_rules': final_rules,
                 'group_folds': partition, 'seed': seed, 'folds': folds,
                 'input_rows_sha256': digest(rows), 'unavailable_candidates': sorted(set(KINDS)-set(kinds)),
                 'observed_domains': sorted(required_domains),
                 'all_four_domains_supported': set(required_domains)==set(base.DATASETS),
                 'independent_confirmation': False, 'evidence': 'previously_examined_development_selection',
                 'selection_is_not_unbiased_performance_estimate': True})


def select_shared(reports_by_model, *, required_models=MODELS, cost_tie_fraction=.01):
    """Equal-model NORMALIZED KV objective; all required models must qualify."""
    if not 0 <= cost_tie_fraction < 1:
        raise ValueError('relative cost tie must be in [0,1)')
    if set(reports_by_model) != set(required_models):
        raise ValueError('all required model reports are necessary before freezing')
    if len({r['configuration']['profile'] for r in reports_by_model.values()}) != 1:
        raise ValueError('profiles must match across models')
    if any(r.get('all_four_domains_supported') is False for r in reports_by_model.values()):
        raise ValueError('incomplete domain coverage cannot establish the shared four-domain gate')
    profile = next(iter(reports_by_model.values()))['configuration']['profile']
    comparisons = {}
    for kind in KINDS:
        if not all(kind in r['candidates'] for r in reports_by_model.values()):
            comparisons[kind] = {'eligible': False, 'reason': 'features_or_report_incomplete'}
            continue
        values = {model:r['candidates'][kind] for model,r in reports_by_model.items()}
        comparisons[kind] = {'eligible': all(v['eligible'] for v in values.values()),
            'equal_model_normalized_kv': sum(v['normalized_kv_capacity'] for v in values.values())/len(values),
            'worst_domain_loss_pp': max(v['summary']['worst_domain_quality_loss_pp'] for v in values.values()),
            'model_summaries': {m:v['summary'] for m,v in values.items()},
            'operations': CANDIDATES[kind]['operations']}
    eligible = [k for k,c in comparisons.items() if c['eligible']]
    winner = None
    if eligible:
        minimum = min(comparisons[k]['equal_model_normalized_kv'] for k in eligible)
        near = [k for k in eligible if comparisons[k]['equal_model_normalized_kv'] <= minimum*(1+cost_tie_fraction)+1e-12]
        winner = min(near, key=lambda k:(CANDIDATES[k]['operations'], comparisons[k]['worst_domain_loss_pp'], k))
    return seal({'format': FORMAT+'_shared_selection', 'profile': profile, 'rule_kind': winner,
        'comparisons': comparisons, 'models': {m:{'rule':r['final_rules'][winner]} for m,r in reports_by_model.items()} if winner else {},
        'cost_tie_fraction': cost_tie_fraction, 'objective': 'equal_model_mean_normalized_allocated_kv',
        'selection_order': ['feasible_all_models', 'minimum_normalized_KV', 'within_1_percent_fewer_operations',
                            'worst_domain_loss', 'lexical_family'],
        'independent_confirmation': False, 'physical_matched_benchmarks_required': True,
        'if_no_candidate': 'retain_incumbent_and_report_no_qualifying_improvement'})


def select_operating_points(reports_by_profile, *, target_profile='primary', cost_tie_fraction=.01):
    """Use only the same or stricter PREDECLARED calibration profiles.

    Calibration at the target mean/worst loss can overfit development folds.
    A stricter profile is a legitimate numeric threshold, not relaxed quality
    gates. This further development selection is explicitly not independent
    confirmation; final physical benchmarks must evaluate the frozen choice.
    """
    if target_profile not in PROFILES or target_profile not in reports_by_profile:
        raise ValueError('target profile report required')
    target=reports_by_profile[target_profile]
    allowed={name:report for name,report in reports_by_profile.items()
             if name in PROFILES and all(a<=b for a,b in zip(PROFILES[name],PROFILES[target_profile]))}
    models={r['configuration']['model'] for r in allowed.values()}
    hashes={r['input_rows_sha256'] for r in allowed.values()}
    if len(models)!=1 or len(hashes)!=1:
        raise ValueError('operating points must use one identical model and development cohort')
    candidates,final_rules,choices={},{},{}
    for kind in target['candidates']:
        available={name:r['candidates'][kind] for name,r in allowed.items() if kind in r['candidates']}
        eligible={name:c for name,c in available.items()
                  if base.meets_constraints(c['summary'],*PROFILES[target_profile])}
        chosen=target_profile
        if eligible:
            minimum=min(c['normalized_kv_capacity'] for c in eligible.values())
            near=[name for name,c in eligible.items() if c['normalized_kv_capacity']<=minimum*(1+cost_tie_fraction)+1e-12]
            chosen=min(near,key=lambda name:(eligible[name]['summary']['worst_domain_quality_loss_pp'],
                                             PROFILES[name],name))
        candidate=copy.deepcopy(available[chosen])
        candidate.update(eligible=bool(eligible),calibration_profile=chosen,
                         evaluation_quality_profile=target_profile)
        candidates[kind]=candidate
        final_rules[kind]=allowed[chosen]['final_rules'][kind]
        choices[kind]={'calibration_profile':chosen,'eligible_profiles':sorted(eligible),
                       'quality_limits_unchanged':True}
    return seal({'format':FORMAT+'_development_operating_points','configuration':target['configuration'],
        'candidates':candidates,'final_rules':final_rules,'choices':choices,
        'input_rows_sha256':target['input_rows_sha256'],
        'observed_domains':target.get('observed_domains',sorted(next(iter(target['candidates'].values()))['summary']['domains'])),
        'all_four_domains_supported':target.get('all_four_domains_supported',True),
        'source_reports':{p:r['sha256'] for p,r in allowed.items()},
        'same_or_stricter_predeclared_profiles_only':True,'target_profile':target_profile,
        'independent_confirmation':False,'selection_is_not_unbiased_performance_estimate':True})


def merge_features(rows, features):
    """Require exact cohort, geometry and old physical prefix parity."""
    if set(features) != {r['row_hash'] for r in rows}:
        raise ValueError('new features must cover the complete development cohort')
    result = []
    for row in rows:
        feature = features[row['row_hash']]
        for key in ('input_visual_tokens', 'input_text_tokens'):
            if feature.get(key) != row[key]:
                raise ValueError('new feature geometry mismatch: '+key)
        if feature.get('group_id', row['group_id']) != row['group_id']:
            raise ValueError('new feature group mismatch')
        old = row['curves']['scorer']
        curves = copy.deepcopy(row['curves'])
        for name,new in feature['curves'].items():
            if base.previous._invalid_curve(new):
                if new['counts'] != [row['input_visual_tokens']]:
                    raise ValueError('invalid newly extracted geometry must expose full fallback only')
                curves[name] = copy.deepcopy(new)
                continue
            # Allow a wider freshly extracted menu, but never fabricate labels.
            positions = []
            for count in old['counts']:
                if count not in new['counts']:
                    raise ValueError('new curve omits measured physical count')
                positions.append(new['counts'].index(count))
            adapted = copy.deepcopy(new)
            for field in ('counts', 'missing_mass', 'uncovered_mass', 'removed_fraction'):
                adapted[field] = [new[field][i] for i in positions]
            if 'selected_indices_sha256' in new:
                adapted['selected_indices_sha256'] = {str(k):new['selected_indices_sha256'][str(k)] for k in old['counts']}
            if old.get('selected_indices_sha256') is not None:
                if adapted.get('selected_indices_sha256') != old['selected_indices_sha256']:
                    raise ValueError('new feature prefixes differ from cached physical inference')
            if name == 'scorer':
                for field in ('missing_mass', 'uncovered_mass'):
                    if any(abs(a-b)>1e-6 for a,b in zip(adapted[field],old[field])):
                        raise ValueError('new extraction changed the incumbent curve: '+field)
            curves[name] = adapted
        result.append({**copy.deepcopy(row), 'curves': curves})
    return result


def extend_physical_development(rows, features, physical_rows, *, additional_entries=()):
    """Admit new development domains/knots only with every physical outcome.

    ``features`` is the complete old+new cohort. ``physical_rows`` comes from
    completed, provenance-checked v16 worker shards. Its selected-index hashes
    are checked again here. Callers must audit additional source identities
    against scorer training and evaluation BEFORE running the worker; this
    routine does not invent a new split or reclassify inspected evaluation.
    """
    from .adaptive_data_v3 import dataset_family
    original = {r['row_hash']:r for r in rows}
    if len(original)!=len(rows):
        raise ValueError('duplicate existing development request')
    extra = {e['row_hash']:e for e in additional_entries}
    if len(extra)!=len(additional_entries) or set(original)&set(extra):
        raise ValueError('additional development requests overlap existing rows')
    if set(features)!=set(original)|set(extra):
        raise ValueError('expanded features must cover exactly old plus admitted new rows')
    if {r['group_id'] for r in rows}&{r['group_id'] for r in additional_entries}:
        raise ValueError('additional source group overlaps existing development')
    # Verify unchanged physical-prefix identities before using old labels at
    # any retained count. New knots never inherit an interpolated outcome.
    merge_features(rows,{k:features[k] for k in original})
    physical = {}
    for value in physical_rows:
        if value.get('selector') not in ('scorer','full'):
            continue
        key = value.get('request_id',value.get('row_hash'))
        if key not in features:
            raise ValueError('new physical outcome is outside admitted development')
        feature=features[key]
        count=value['selected_count']
        n=feature['input_visual_tokens']
        if (value.get('status')!='ok' or value.get('input_visual_tokens')!=n
                or value.get('input_text_tokens')!=feature['input_text_tokens']
                or type(count)is not int or not 1<=count<=n):
            raise ValueError('new physical status or geometry mismatch')
        indices=value.get('selected_patch_indices')
        expected=feature['curves']['scorer']['selected_indices_sha256'].get(str(count))
        if (not isinstance(indices,list) or len(indices)!=count or len(set(indices))!=count
                or indices!=sorted(indices) or digest(indices)!=expected):
            raise ValueError('new physical selection differs from frozen learned prefix')
        score=value.get('quality_score')
        if type(score) not in (int,float) or not 0<=score<=1:
            raise ValueError('finite native physical quality in [0,1] required')
        cell=(key,count)
        if cell in physical and physical[cell]!=score:
            raise ValueError('conflicting physical labels for one exact prefix')
        physical[cell]=float(score)
    merged=[]
    for key,feature in sorted(features.items()):
        if key in original:
            row=copy.deepcopy(original[key])
        else:
            entry=extra[key]
            row={'row_hash':key,'group_id':entry['group_id'],'dataset':dataset_family(entry['dataset']),
                 'category':entry.get('category',''),'quality_metric':entry.get('quality_metric',entry.get('metric')),
                 'input_visual_tokens':feature['input_visual_tokens'],
                 'input_text_tokens':feature['input_text_tokens'],'quality_by_count':{}}
        row['curves']=copy.deepcopy(feature['curves'])
        for count in row['curves']['scorer']['counts']:
            if (key,count) in physical:
                if str(count) in row['quality_by_count'] and abs(row['quality_by_count'][str(count)]-physical[key,count])>1e-12:
                    raise ValueError('new physical result disagrees with cached same-prefix quality')
                row['quality_by_count'][str(count)]=physical[key,count]
            if str(count) not in row['quality_by_count']:
                raise ValueError(f'missing exact physical development label: {key} count={count}')
        merged.append(row)
    for kind in available_kinds(merged):
        transformed_rows(merged,kind)
    return merged


def load_development(project, model):
    """Load only pinned fit/screening; no evaluation path is inspected."""
    project = Path(project).resolve()
    if model in ('qwen25','llava_next'):
        root = project/'reproduction_results/mlsys_compression_20261022_v12'
        path = root/model/'development/rows.json'
        payload, study = checked(path), checked(root/'study.json')
        provenance = payload['provenance']
        if provenance.get('confirmation_opened') is not False or provenance.get('evaluation_opened') is not False:
            raise ValueError('development provenance must exclude evaluation/confirmation labels')
        config = {'bytes_per_token': study['bytes_per_token'][model], 'generation_reserve': study['generation_reserve']}
        return payload['rows'], {**provenance, 'v16_source_path':str(path), 'v16_source_sha256':file_digest(path)}, config
    if model != 'phi35':
        raise ValueError('unsupported development model')
    from . import compression_third_model_v15 as phi
    path = project/phi.ROOT/'study.json'
    study = checked(path)
    rows, source_hashes = [], {str(path):file_digest(path)}
    for split in ('fit','screening'):
        features, physical = phi.collect_features(study,split), phi.collect_physical(study,split)
        if features is None or physical is None:
            raise ValueError('Phi development features/physical labels incomplete: '+split)
        rows.extend(phi._policy_rows(study,split,features,physical))
        source_hashes[str(Path(study['output_root'])/'schedules'/(split+'.json'))] = file_digest(Path(study['output_root'])/'schedules'/(split+'.json'))
    return rows, {'input_hashes':source_hashes,'evaluation_opened':False,'confirmation_opened':False,
                  'physical_row_digest':digest(rows),'source_splits':['fit','screening']}, {
        'bytes_per_token':study['bytes_per_token'],'generation_reserve':study['generation_reserve']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--features', action='append', default=[], help='model=JSON mapping request IDs to extracted feature rows')
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(PROFILES))
    args = parser.parse_args(argv)
    extras = dict(value.split('=',1) for value in args.features)
    reports = {profile:{} for profile in args.profiles}
    for model in args.models:
        rows, provenance, config = load_development(args.project,model)
        if model in extras:
            feature_path = Path(extras[model])
            payload = json.loads(feature_path.read_text())
            features = payload.get('features',payload)
            rows = merge_features(rows,features)
            provenance['v16_feature_sha256'] = file_digest(feature_path)
        write_immutable(args.output/model/'rows.json',seal({'rows':rows,'provenance':provenance}))
        for profile in args.profiles:
            report = development_report(rows,model=model,profile=profile,**config)
            reports[profile][model] = report
            write_immutable(args.output/model/(profile+'.json'),report)
    selections = {}
    for profile,values in reports.items():
        operating={model:select_operating_points({p:r[model] for p,r in reports.items()},target_profile=profile)
                   for model in args.models}
        for model,report in operating.items():
            write_immutable(args.output/model/('operating-point-'+profile+'.json'),report)
        selection = select_shared(operating,required_models=tuple(args.models))
        selections[profile] = selection['rule_kind']
        write_immutable(args.output/('shared-'+profile+'.json'),selection)
    print(json.dumps({'output':str(args.output),'models':args.models,
                      'selected':selections},sort_keys=True))


if __name__ == '__main__':
    main()
