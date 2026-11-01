"""Versioned numerical repair of greedy conditional-DPP selection.

The query-conditioned DPP objective is unchanged. Kernel construction and
Cholesky-style residual updates use FP64; previously selected indices remain
explicitly unavailable. Residuals below a dimension-scaled roundoff threshold
are zero-gain ties, resolved by original patch order. No learned fallback,
answer-dependent choice, jitter, or retention change is introduced.

Standalone CLI accepts the frozen benchmark's arguments. Existing output
directories are never reused. ``stable_cdpruner_context`` composes with the
separate text-policy launcher without changing its timing/generation function.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import fcntl
import json
from pathlib import Path
import threading

from learned_score.mlsys_scorer.checkpoints import atomic_json, canonical_digest
from learned_score.mlsys_scorer.protocol import digest_file


_LOCK = threading.Lock()
_DIAGNOSTICS = ContextVar("stable_cdpruner_diagnostics", default=None)


def repair_configuration():
    return {"format": "mlsys_cdpruner_numerical_repair_v1",
            "algorithm": "conditional_DPP_greedy_MAP",
            "kernel_dtype": "float64", "residual_dtype": "float64",
            "relevance_normalization": "(r-min(r)+1e-6)/max(range(r),1e-6)",
            "roundoff_tolerance": "8*eps64*max(number_of_patches,hidden_size)*max_kernel_diagonal",
            "zero_gain_tie_break": "original_patch_index",
            "already_selected_indices": "explicitly_excluded",
            "retention_unchanged": True, "objective_unchanged": True,
            "learned_or_answer_dependent_fallback": False,
            "source_sha256": digest_file(Path(__file__))}


def stable_cdpruner_order(vision, relevance, keep_count):
    import torch
    with torch.no_grad():
        return _stable_cdpruner_order(vision, relevance, keep_count)


def _stable_cdpruner_order(vision, relevance, keep_count):
    import torch
    from torch.nn import functional as F
    if (vision.ndim != 2 or not len(vision) or not vision.shape[1]
            or relevance.shape != (len(vision),)
            or type(keep_count) is not int or not 1 <= keep_count <= len(vision)):
        raise ValueError("DPP needs nonempty patches, matching relevance, and an exact valid count")
    if not torch.isfinite(vision).all() or not torch.isfinite(relevance).all():
        raise FloatingPointError("nonfinite CDPruner inputs")
    normalized = F.normalize(vision.double(), dim=-1)
    weights = relevance.double()
    weights = (weights - weights.min() + 1e-6) / (weights.max() - weights.min()).clamp_min(1e-6)
    weighted = normalized * weights[:, None]
    kernel = weighted @ weighted.T
    if not torch.isfinite(kernel).all():
        raise FloatingPointError("nonfinite FP64 DPP kernel")
    residual = kernel.diagonal().clone().clamp_min_(0)
    scale = residual.max().clamp_min(torch.finfo(torch.float64).tiny)
    tolerance = 8 * torch.finfo(torch.float64).eps * max(vision.shape) * scale
    # Avoid division by zero even for the all-zero-feature PSD kernel.
    denominator_floor = tolerance.clamp_min(torch.finfo(torch.float64).tiny)
    cis = torch.zeros((keep_count, len(vision)), dtype=torch.float64, device=vision.device)
    selected = torch.empty(keep_count, dtype=torch.long, device=vision.device)
    available = torch.ones(len(vision), dtype=torch.bool, device=vision.device)
    zero_gains = torch.zeros((), dtype=torch.long, device=vision.device)
    smallest_raw = torch.zeros((), dtype=torch.float64, device=vision.device)
    for index in range(keep_count):
        scores = torch.where(residual > tolerance, residual, 0.)
        scores.masked_fill_(~available, -float("inf"))
        chosen = scores.argmax()
        selected[index] = chosen
        positive = residual[chosen] > tolerance
        zero_gains += ~positive
        correction = (cis[:index, chosen, None] * cis[:index]).sum(dim=0) if index else 0.
        update = (kernel[chosen] - correction) / residual[chosen].clamp_min(denominator_floor).sqrt()
        # Once all remaining gains are below numerical precision, the objective
        # cannot distinguish them. No unstable divide-and-update is performed.
        update = torch.where(positive, update, 0.)
        cis[index] = update
        residual -= update.square()
        smallest_raw = torch.minimum(smallest_raw, residual.min())
        residual.clamp_min_(0)
        available[chosen] = False
    if not torch.isfinite(cis).all() or not torch.isfinite(residual).all():
        raise FloatingPointError("nonfinite FP64 DPP residual update")
    if len(selected.unique()) != keep_count:
        raise RuntimeError("stable DPP violated exact unique retained count")
    diagnostics = _DIAGNOSTICS.get()
    if diagnostics is not None:
        diagnostics.update({"algorithm": "conditional_DPP_greedy_MAP", "kernel_dtype": "float64",
                            "zero_gain_tie_selections": int(zero_gains.item()),
                            "roundoff_tolerance": float(tolerance.item()),
                            "minimum_preclamp_residual": float(smallest_raw.item()),
                            "requested_patches": keep_count, "unique_selected_patches": keep_count})
    return selected


@contextmanager
def stable_cdpruner_context(module=None):
    """Patch selection only; preserves all cell fields and timing/run callbacks."""
    if module is None:
        from . import benchmark as module
    from learned_score import llava_selectors
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("stable selector contexts cannot nest or run in concurrent threads")
    old_cd, old_make = llava_selectors.cdpruner_order, module.make_selector

    def make_selector(cell, scorer=None, scorer_variant=None, observed=None):
        select = old_make(cell, scorer, scorer_variant, observed)
        def with_diagnostics(prepared):
            diagnostics = {}
            token = _DIAGNOSTICS.set(diagnostics)
            try:
                result = select(prepared)
                if diagnostics and observed is not None:
                    observed["selector_numerics"] = diagnostics
                return result
            finally:
                _DIAGNOSTICS.reset(token)
        return with_diagnostics

    llava_selectors.cdpruner_order = stable_cdpruner_order
    module.make_selector = make_selector
    try:
        yield module
    finally:
        llava_selectors.cdpruner_order, module.make_selector = old_cd, old_make
        _LOCK.release()


def guard_output_directory(output, configuration):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".baseline_revision.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = output / "baseline_revision.metadata.json"
        binding = {"configuration": configuration, "configuration_sha256": canonical_digest(configuration)}
        if path.exists():
            if json.loads(path.read_text()) != binding:
                raise ValueError("baseline numerical implementation differs; use a new output directory")
        else:
            if any(item.name != ".baseline_revision.lock" for item in output.iterdir()):
                raise ValueError("do not mix repaired baselines with old outputs")
            atomic_json(path, binding)


@dataclass(frozen=True)
class RepairedCell:
    cell: object
    configuration_sha256: str

    @property
    def selector(self): return self.cell.selector
    @property
    def visual_retention(self): return self.cell.visual_retention
    @property
    def generated_budget(self): return self.cell.generated_budget
    @property
    def key(self):
        return self.cell.key if self.cell.method == "all_tokens" else f"{self.cell.key}_stablecd_{self.configuration_sha256[:12]}"
    @property
    def method(self): return "all_tokens" if self.cell.method == "all_tokens" else self.key


@contextmanager
def scoped_repaired_benchmark(configuration, module=None):
    if module is None:
        from . import benchmark as module
    original_grid, original_atomic = module.make_grid, module.atomic_json
    binding = {"configuration": configuration, "configuration_sha256": canonical_digest(configuration)}
    def grid(*args, **kwargs):
        return [RepairedCell(cell, binding["configuration_sha256"]) for cell in original_grid(*args, **kwargs)]
    def write(path, payload):
        if isinstance(payload, dict) and payload.get("format") == "mlsys_request_result_v1":
            payload = {**payload, "baseline_revision_binding": binding}
        return original_atomic(path, payload)
    with stable_cdpruner_context(module):
        module.make_grid, module.atomic_json = grid, write
        try:
            yield module
        finally:
            module.make_grid, module.atomic_json = original_grid, original_atomic


def main(argv=None):
    from . import benchmark
    args = benchmark.parse_args(argv)
    configuration = repair_configuration()
    guard_output_directory(args.output, configuration)
    with scoped_repaired_benchmark(configuration, benchmark):
        benchmark.main(argv)


if __name__ == "__main__":
    main()
