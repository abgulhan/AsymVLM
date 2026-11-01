"""CPU comparison against executable, unmodified vendored DivPrune methods.

This is selector-rule evidence only. It neither imports nor validates the
upstream model pipeline, crop handling, positional encoding, or separators.
Known degenerate-input failures are reported, never silently repaired.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import torch

from learned_score.llava_selectors import divprune_order
from .adaptive_allocation_v4 import _seal


FORMAT = "mlsys_divprune_selector_audit_v5"
DEFAULT_UPSTREAM = Path(__file__).resolve().parents[2] / "external_baselines/DivPrune/LLaVA/llava/model/llava_arch.py"


def load_upstream_rule(path=DEFAULT_UPSTREAM):
    """Execute the two exact upstream method ASTs, without model dependencies."""
    path = Path(path)
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LlavaMetaForCausalLM"]
    if len(classes) != 1:
        raise ValueError("missing or ambiguous upstream class")
    names = ("pairwise_cosine_similarity", "DivPrune")
    methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(methods) != 2 or {node.name for node in methods} != set(names):
        raise ValueError("upstream method contract changed")
    if any(node.decorator_list for node in methods):
        raise ValueError("unexpected upstream decorators")
    body = ast.Module(body=methods, type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(body, str(path), "exec"), namespace)
    rule = type("ExactVendoredDivPrune", (), {name: namespace[name] for name in names})()
    return rule, {"path": str(path), "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                  "method_ast_sha256": hashlib.sha256(ast.dump(body).encode()).hexdigest()}


def _capture(call, n, k):
    try:
        output = call()
        matrix = None
        if isinstance(output, tuple):
            output, matrix = output
        indices = output.tolist()
        valid = (output.ndim == 1 and output.dtype == torch.long and len(indices) == k
                 and len(set(indices)) == k and all(0 <= value < n for value in indices))
        return {"status": "ok" if valid else "invalid_selection", "indices": indices,
                "unique_count": len(set(indices)), "exact_unique_count_valid": valid,
                "distance_matrix_finite": None if matrix is None else bool(torch.isfinite(matrix).all()),
                "error": None}
    except Exception as error:
        return {"status": "error", "indices": None, "unique_count": None,
                "exact_unique_count_valid": False, "distance_matrix_finite": None,
                "error": f"{type(error).__name__}: {error}"}


def compare_features(features, keep_count, *, name="external_features", upstream=None):
    """Audit identical CPU features. Failures remain inspectable, not repaired."""
    if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.device.type != "cpu":
        raise ValueError("audit features must be a rank-two CPU tensor")
    if not features.is_floating_point() or not bool(torch.isfinite(features).all()) or not features.numel():
        raise ValueError("audit features must be finite and nonempty floating point")
    n = len(features)
    if type(keep_count) is not int or not 0 <= keep_count <= n:
        raise ValueError("keep_count must be an integer in [0, N]")
    if upstream is None:
        upstream, _ = load_upstream_rule()
    reference = _capture(lambda: upstream.DivPrune(features.clone(), n, threshold_ratio=keep_count / n), n, keep_count)
    adaptation = _capture(lambda: divprune_order(features.clone(), keep_count), n, keep_count)
    equal = reference["indices"] is not None and reference["indices"] == adaptation["indices"]
    valid = (reference["exact_unique_count_valid"] and adaptation["exact_unique_count_valid"]
             and reference["distance_matrix_finite"] is True)
    normalized = torch.nn.functional.normalize(features.float(), dim=-1)
    normalized_same_precision = features / features.norm(dim=1, keepdim=True)
    return {"case": name, "dtype": str(features.dtype), "shape": list(features.shape), "keep_count": keep_count,
            "features_sha256": hashlib.sha256(features.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
            "zero_norm_rows": int((features.float().norm(dim=1) == 0).sum()),
            "normalization_same_precision_max_error": (float((normalized - normalized_same_precision.float()).abs().max())
                    if bool(torch.isfinite(normalized_same_precision).all()) else None),
            "upstream": reference, "adaptation": adaptation, "ordered_indices_equal": equal,
            "sorted_indices_equal": (reference["indices"] is not None and adaptation["indices"] is not None
                                      and sorted(reference["indices"]) == sorted(adaptation["indices"])),
            "passed": bool(valid and equal), "safe_to_use_on_this_input": bool(valid and equal),
            "decision": "rule_parity_on_this_input_only" if valid and equal else "fail_closed_do_not_claim_parity"}


def run_audit(*, upstream_path=DEFAULT_UPSTREAM):
    """Return a sealed JSON-compatible selector audit; caller owns persistence."""
    upstream, source = load_upstream_rule(upstream_path)
    generator = torch.Generator(device="cpu").manual_seed(42)
    random = torch.randn(32, 16, generator=generator)
    fixtures = [("random_fp32", random), ("random_fp64", random.double()),
                ("random_bf16", random.bfloat16()), ("orthogonal_ties", torch.eye(8)),
                ("identical_duplicates", torch.ones(8, 4)),
                ("paired_duplicates", torch.eye(4).repeat_interleave(2, dim=0)),
                ("zero_rows", torch.zeros(8, 4)),
                ("one_zero_row", torch.cat([torch.zeros(1, 4), torch.eye(4)])),
                ("single_token", torch.ones(1, 4))]
    cases = []
    for name, features in fixtures:
        for k in sorted({0, 1, max(1, len(features) // 2), len(features)}):
            cases.append(compare_features(features, k, name=name, upstream=upstream))
    ours = Path(__file__).resolve().parents[1] / "llava_selectors.py"
    invalid = [case for case in cases if not case["passed"]]
    return _seal({"format": FORMAT, "seed": 42, "device": "cpu", "torch_version": torch.__version__,
                  "upstream": source, "adaptation_source_sha256": hashlib.sha256(ours.read_bytes()).hexdigest(),
                  "cases": cases, "case_count": len(cases), "passed_cases": len(cases) - len(invalid),
                  "failed_cases": len(invalid), "all_cases_pass": not invalid,
                  "claim": "selector-only restricted parity; not an official pipeline reproduction",
                  "failure_policy": "do not silently deduplicate, reseed, renormalize, or change either method",
                  "pipeline_validated": False,
                  "not_audited": ["native VLM integration", "crop boundaries", "protected image separators",
                                  "position IDs", "token source feature representation", "GPU tie ordering"],
                  "precision_difference": "adaptation always converts to FP32; upstream retains input dtype",
                  "degeneracy_note": "both rules omit an already-selected mask; upstream also divides by zero at zero norms"})
