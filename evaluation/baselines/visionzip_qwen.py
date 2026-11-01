"""Native-feature VisionZip adaptation for Qwen2.5-VL.

Adapted from the code used to produce the paper results. Cluster scheduling
and campaign bookkeeping were removed; the numerical code is unchanged.
"""
import argparse

from contextlib import contextmanager

import gc

import os

from pathlib import Path

import time

import traceback

from learned_score.mlsys_sprint import manuscript_campaign_v35 as campaign

from learned_score.mlsys_sprint.manuscript_execute_v35 import Runtime,selector,score,cache_key

from learned_score.mlsys_sprint.manuscript_data_v35 import family

from learned_score.mlsys_sprint.state import atomic_json,digest,file_digest

@contextmanager
def visionzip_statistics(runtime):
    """Expose the existing Qwen native-tower statistics without replacing weights."""
    if runtime.model!='qwen25':raise ValueError('VisionZip port is admitted only for Qwen2.5')
    original=runtime.adapter.extract_visual_features;stats={}
    def extract(pixels,grid,with_stats=False):
        features,received,keys=original(pixels,grid,with_stats=True)
        stats.update(features=features,received=received,keys=keys)
        return features,received,keys
    runtime.adapter.extract_visual_features=extract
    try:yield stats
    finally:runtime.adapter.extract_visual_features=original


def visionzip_choose(count,stats,observed):
    import torch
    from learned_score.qwen2_pruning import visionzip_selection
    from learned_score.mlsys_sprint.selector_experiment_v5 import validate_indices
    @torch.no_grad()
    def choose(prepared):
        if prepared.answer_positions.numel():raise ValueError('answer leakage')
        positions=prepared.visual_positions;visual=prepared.inputs_embeds[0,positions]
        if not torch.equal(visual,stats['features']):raise ValueError('native visual-feature ordering mismatch')
        if count==len(visual):
            indices=torch.arange(len(visual),device=visual.device)
            metadata=dict(dominant_tokens=count,contextual_tokens=0)
        else:
            indices,merged,metadata=visionzip_selection(visual,stats['received'],stats['keys'],count)
            # Only this request's contextual patch embeddings are replaced.
            prepared.inputs_embeds[0,positions[indices]]=merged
        validate_indices(indices,count,len(visual))
        observed.update(input_visual_tokens=len(visual),retained_visual_tokens=count,
            selected_indices_sha256=digest(indices.cpu().tolist()),**metadata)
        return indices
    return choose


def physical(runtime,entry,method,count,*,use_cache=True):
    import torch
    from learned_score.mlsys_sprint import adaptive_experiment_v3 as old
    from learned_score.mlsys_sprint import adaptive_experiment_v4 as measured
    from learned_score.mlsys_scorer.backbone import assert_frozen
    config=runtime.config;root=Path(config['output_root'])/'addons'
    raw,prepared,feature=runtime.geometry(entry)
    n=feature['input_visual_tokens'];del prepared
    underlying='visionzip_native' if method=='visionzip_native' else 'asymvlm_uniform'
    key=cache_key(config,runtime.model,entry,underlying,count)
    if underlying=='visionzip_native':key=digest(dict(base=key,port_sha256=file_digest(__file__)))
    path=root/'raw'/key[:2]/(key+'.json')
    expected=measured.expected_cache_bytes(feature,count,runtime.contract)
    if path.exists() and use_cache:saved=campaign.read(path)
    else:
        observed={};cell=old.Cell('scorer','uniform',0)
        torch.cuda.reset_peak_memory_stats()
        if underlying=='visionzip_native':
            with visionzip_statistics(runtime) as stats:
                prediction,ids,timing,metrics=old.measured(runtime.adapter,raw,visionzip_choose(count,stats,observed),cell,128)
        else:
            prediction,ids,timing,metrics=old.measured(runtime.adapter,raw,selector('asymvlm_uniform',count,runtime.scorer,observed),cell,128)
        metrics=measured.generation_trace(runtime.adapter,ids,metrics);old.check_cache(metrics,expected)
        if observed['retained_visual_tokens']!=count:raise ValueError('wrong physical count')
        assert_frozen(runtime.adapter.model,runtime.versions);assert_frozen(runtime.scorer,runtime.scorer_versions)
        saved=dict(cache_key=key,selected_count=count,prediction=prediction,generated_token_ids=ids[0].tolist(),
            timing=timing,metrics=metrics,physical_measurement_performed=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            **observed,source_sha256=file_digest(__file__),bundle_hash=config['bundle_hash'])
        if use_cache:atomic_json(path,saved)
    if saved['cache_key']!=key or saved['selected_count']!=count:raise ValueError('invalid physical cache')
    labeled=runtime.labels.row_for_entry(entry)
    references=old.benchmark.parse_references(labeled['answer'])
    record=dict(request_id=entry['row_hash'],group_id=entry['group_id'],dataset=family(entry['dataset']),
        model=runtime.model,method=method,input_visual_tokens=n,retained_visual_tokens=count,
        kv_bytes=expected,full_kv_bytes=measured.expected_cache_bytes(feature,n,runtime.contract),
        quality_score=score(saved['prediction'],references,entry['metric']),raw_path=str(path),
        cache_key=key,source_sha256=file_digest(__file__),bundle_hash=config['bundle_hash'],
        evidence='training-disjoint public rerun; not untouched confirmation',timing_isolated=False,status='ok')
    if use_cache:atomic_json(root/'records'/runtime.model/record['dataset']/method/(entry['row_hash']+'.json'),record)
    return record,saved


def visionzip_canary(runtime,entry):
    import torch
    raw,prepared,feature=runtime.geometry(entry)
    baseline=prepared.inputs_embeds.clone();native=runtime.adapter.extract_visual_features
    features,_,_=native(prepared.native_inputs['pixel_values'],prepared.native_inputs['image_grid_thw'],with_stats=False)
    with visionzip_statistics(runtime) as stats:
        extracted,_,_=runtime.adapter.extract_visual_features(prepared.native_inputs['pixel_values'],prepared.native_inputs['image_grid_thw'])
        if not torch.equal(features,extracted):raise ValueError('statistics path changes native Qwen features')
        observed={};visionzip_choose(feature['input_visual_tokens'],stats,observed)(prepared)
        if not torch.equal(baseline,prepared.inputs_embeds):raise ValueError('full-token VisionZip changes input embeddings')
    del prepared,baseline,features,extracted
    full,native_full=physical(runtime,entry,'asymvlm_uniform',feature['input_visual_tokens'],use_cache=False)
    _,ported_full=physical(runtime,entry,'visionzip_native',feature['input_visual_tokens'],use_cache=False)
    if native_full['generated_token_ids']!=ported_full['generated_token_ids']:
        raise ValueError('VisionZip full-retention generation parity failed')
    record,_=physical(runtime,entry,'visionzip_native',(feature['input_visual_tokens']*70+99)//100,use_cache=False)
    return dict(passed=True,native_features_exact=True,full_generation_exact=True,physical_pruned_count=record['retained_visual_tokens'],
                method='VisionZip native Qwen2.5 adaptation; dominant/contextual merging, original mRoPE positions')


