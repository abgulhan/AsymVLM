"""Verify actual SDPA dispatch; configuration names alone are not proof."""

import torch

from .generation import generate_bounded


def classify_attention_events(operator_names, cuda_event_count):
    names = sorted(set(operator_names))
    fused = [name for name in names if any(marker in name for marker in (
        "aten::_scaled_dot_product_flash_attention",
        "aten::_scaled_dot_product_efficient_attention",
        "aten::_scaled_dot_product_cudnn_attention",
    )) and "for_cpu" not in name]
    math = [name for name in names if "_scaled_dot_product_attention_math" in name]
    return {
        "fused_sdpa_verified": bool(fused and not math and cuda_event_count > 0),
        "fused_dispatch_operators": fused,
        "math_attention_operators": math,
        "cuda_profiler_events": int(cuda_event_count),
        "performance_measurement": False,
    }


def verify_fused_attention(model, pruned, max_new_tokens=48, generated_budget=8, slack=4):
    """Profile an actual bounded-cache decode, after an untimed warmup.

    Timing from this run must not be used as a speed result: the profiler and
    dispatch validation intentionally perturb execution.
    """
    if pruned.inputs_embeds.device.type != "cuda":
        raise ValueError("GPU fused-attention verification requires CUDA")
    warmup = generate_bounded(
        model, pruned, max_new_tokens=min(8, max_new_tokens),
        generated_budget=generated_budget, slack=slack, stop_on_eos=False,
    )
    del warmup
    torch.cuda.synchronize(pruned.inputs_embeds.device)
    with torch.profiler.profile(activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]) as profiler:
        result = generate_bounded(
            model, pruned, max_new_tokens=max_new_tokens,
            generated_budget=generated_budget, slack=slack, stop_on_eos=False,
        )
    operator_names = [event.key for event in profiler.key_averages()]
    cuda_events = sum("CUDA" in str(event.device_type) for event in profiler.events())
    report = classify_attention_events(operator_names, cuda_events)
    report.update({
        "generated_tokens": result.metrics["generated_tokens"],
        "cache": result.metrics["cache"],
        "attention_backend_requested": result.metrics["attention_backend_requested"],
        "native_sliding_window": result.metrics["native_sliding_window"],
    })
    return report
