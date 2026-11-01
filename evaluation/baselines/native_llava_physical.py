"""Physical (measured) execution wrapper for the LLaVA-NeXT native baselines.

Adapted from the code used to produce the paper results. Cluster scheduling
and campaign bookkeeping were removed; the numerical code is unchanged.
"""
import argparse

import fcntl

import importlib.util

import json

import os

from pathlib import Path

import signal

import socket

import subprocess

import threading

import time

import traceback

from learned_score.mlsys_sprint import manuscript_campaign_v35 as base

from learned_score.mlsys_sprint.manuscript_execute_v35 import Runtime, score

from learned_score.mlsys_sprint.manuscript_data_v35 import family

from learned_score.mlsys_sprint.state import atomic_json, digest, file_digest

METHODS = ('fastv_native', 'vlcache_native', 'visionzip_llava_native')

def physical(runtime, native, entry, method, percent, tokens=128):
    import torch
    from learned_score.mlsys_runtime.dynamic import generate_dynamic
    from learned_score.mlsys_sprint.adaptive_experiment_v3 import benchmark
    from learned_score.mlsys_scorer.backbone import assert_frozen
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    captured = None
    if method == 'visionzip_llava_native':
        with native.visionzip_capture(runtime.adapter.model) as captured:
            raw, prepared, feature = runtime.geometry(entry)
    else:
        raw, prepared, feature = runtime.geometry(entry)
    n = len(prepared.visual_positions)
    count = (n*percent+99)//100
    if method == 'visionzip_llava_native':
        pruned, selection = native.visionzip_prune(runtime.adapter, prepared, captured, count)
        result = generate_dynamic(runtime.adapter.model, pruned, max_new_tokens=tokens, stop_on_eos=True)
        ids, metrics = result.ids, dict(result.metrics)
        layers = runtime.contract['layers'] if 'layers' in runtime.contract else runtime.adapter.model.config.text_config.num_hidden_layers
        kv_heads = runtime.adapter.model.config.text_config.num_key_value_heads
        head_dim = runtime.adapter.model.config.text_config.hidden_size//runtime.adapter.model.config.text_config.num_attention_heads
        prefix = pruned.inputs_embeds.shape[1]
        per_token_bytes = layers*2*kv_heads*head_dim*prepared.inputs_embeds.element_size()
        measured_final = result.cache.memory_stats()['kv_active_bytes']
        measured_prefix = measured_final-(ids.shape[1]-1)*per_token_bytes
        if measured_prefix != prefix*per_token_bytes:
            raise ValueError('VisionZip physical cache byte count mismatch')
        metrics.update(prefix_kv_bytes=measured_prefix, final_kv_bytes=measured_final,
            prefix_tokens_per_layer=[prefix]*layers, actual_prefix_slots=layers*prefix,
            target_prefix_slots=layers*prefix, selection=selection)
    else:
        ids, metrics = native.generate_native(runtime.adapter.model, prepared, method, percent, tokens)
    prediction = runtime.adapter.processor.decode(ids[0], skip_special_tokens=True).strip()
    elapsed = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated()
    assert_frozen(runtime.adapter.model, runtime.versions)
    assert_frozen(runtime.scorer, runtime.scorer_versions)
    # No ground-truth answers enter preparation, allocation, or generation.
    labeled = runtime.labels.row_for_entry(entry)
    if labeled['question'] != raw['question']:
        raise ValueError('label/input identity mismatch')
    refs = benchmark.parse_references(labeled['answer'])
    full_prefix = prepared.inputs_embeds.shape[1]
    layers = runtime.adapter.model.config.text_config.num_hidden_layers
    return dict(status='ok', model=runtime.model, dataset=family(entry['dataset']),
        request_id=entry['row_hash'], group_id=entry['group_id'], panel_order=entry['panel_order'],
        method=method, percent=percent, prediction=prediction, generated_ids=ids[0].cpu().tolist(),
        generated_tokens=ids.shape[1], quality_score=score(prediction, refs, entry['metric']), metric=entry['metric'],
        input_visual_tokens=n, nominal_retained_visual_tokens=count, input_prefix_tokens=full_prefix,
        metrics=metrics, request_seconds=elapsed, peak_gpu_allocated_bytes=peak,
        realized_prefix_retention=metrics['actual_prefix_slots']/(layers*full_prefix),
        prefix_budget_error_slots=metrics['actual_prefix_slots']-metrics['target_prefix_slots'],
        physical_measurement_performed=True, timing_isolated=False,
        systems_evidence='concurrent_quality_run_not_headline_latency',
        baseline_fidelity='native implementation/adaptation; not upstream end-to-end reproduction',
        bundle_hash=runtime.config['bundle_hash'], source_sha256=file_digest(native.__file__),
        evaluator_sha256=file_digest(__file__), max_new_tokens=tokens, time=time.time(),
        host=None, job_id=None)


def canary(runtime, native, entry):
    import torch
    from learned_score.mlsys_runtime.dynamic import generate_dynamic
    _, prepared, _ = runtime.geometry(entry)
    full = runtime.adapter.physically_prune(prepared, torch.arange(len(prepared.visual_positions), device=prepared.inputs_embeds.device), 'original')
    expected = generate_dynamic(runtime.adapter.model, full, max_new_tokens=32, stop_on_eos=True).ids[0].cpu().tolist()
    results = {}
    for method in METHODS:
        full_result = physical(runtime, native, entry, method, 100, 32)
        if full_result['generated_ids'] != expected:
            raise ValueError('full-token parity failed: '+method)
        reduced = physical(runtime, native, entry, method, 70, 32)
        if reduced['metrics']['prefix_kv_bytes'] >= full_result['metrics']['prefix_kv_bytes']:
            raise ValueError('physical cache did not shrink: '+method)
        if abs(reduced['prefix_budget_error_slots']) > runtime.adapter.model.config.text_config.num_hidden_layers:
            raise ValueError('cache budget exceeded rounding tolerance: '+method)
        results[method] = dict(full_parity=True, reduced=reduced, full_prefix_kv_bytes=full_result['metrics']['prefix_kv_bytes'])
        print('physical canary passed '+method, flush=True)
    return dict(passed=True, methods=results, request_id=entry['row_hash'], time=time.time())


