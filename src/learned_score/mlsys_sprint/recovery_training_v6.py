"""Bounded v6 soft-gradient recovery pilot; v5 source/artifacts stay untouched.

The checkpoint envelope remains v5-loader compatible. Its identity binds the
v6 gate, source, diagnostic cache, and 64-row official-training subset. Runtime
patches are process-local, restored in finally, and unsupported concurrently.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import random

import torch

from learned_score.mlsys_scorer.checkpoints import atomic_json, atomic_torch_save, canonical_digest, load_checkpoint
from learned_score.mlsys_scorer.protocol import OFFICIAL_TRAIN_DATASETS, digest_file, load_manifest
from . import selector_training_v5 as v5


FORMAT = "asymvlm_recovery_training_v6"
GATES = ("detached", "corrected_kth")
SNAPSHOTS = (0, 16, 32, 64)
RETENTIONS = (0.2, 0.3)
_ORIGINAL_PENALTY = v5.attention_key_penalty
_ACTIVE = False


class DiagnosticGateFailure(RuntimeError):
    """A saved diagnostic crossed the preregistered stop threshold."""


def attention_key_penalty(scores, positions, sequence_length, keep_count, *,
                          gate_gradient="corrected_kth", hard=False, C=5.0, tau=0.1):
    if gate_gradient not in GATES or hard:
        raise ValueError("v6 supports registered soft gradients only")
    # Reuse the historical input validation and exact endpoint semantics.
    original = _ORIGINAL_PENALTY(scores, positions, sequence_length, keep_count,
                                 hard=False, C=C, tau=tau)
    if gate_gradient == "detached" or keep_count == len(scores) or C == 0:
        return original
    selected = torch.argsort(scores, descending=True, stable=True)[:keep_count]
    threshold = scores[selected[-1]]  # Deliberately NOT detached.
    return scores.new_zeros(sequence_length).scatter(
        0, positions, -C * torch.sigmoid((threshold - scores) / tau))


def manifest_payload(source, selection_seed=20261017):
    """Identity-only, deterministic 32+32 subset; never inspect labels."""
    parent = v5.pilot_manifest(source)
    rows = []
    for dataset in sorted(OFFICIAL_TRAIN_DATASETS):
        candidates = sorted((r for r in parent.splits["scorer_train"] if r["dataset"] == dataset),
                            key=lambda r: r["row_hash"])
        random.Random(selection_seed).shuffle(candidates)
        rows.extend(candidates[:32])
    rows.sort(key=lambda r: (r["dataset"], r["row_hash"]))
    payload = {"format": FORMAT + "_manifest", "dataset": "mixed",
               "metadata": {"development_only": True, "fresh_confirmation": False,
                            "source_manifest_sha256": parent.file_sha256,
                            "selection_seed": selection_seed,
                            "selection_rule": "sorted_dataset_rowhash_seeded_shuffle_first32_each",
                            "training_rows": 64, "no_evaluation_labels_consumed": True},
               "splits": {"scorer_train": rows}}
    for key in ("source_catalog_path", "source_catalog_sha256", "source_catalog_partial"):
        if key in parent.metadata:
            payload["metadata"][key] = parent.metadata[key]
    payload["manifest_sha256"] = canonical_digest(payload)
    return payload


def create_manifest(source, output, selection_seed=20261017):
    output = Path(output)
    payload = manifest_payload(source, selection_seed)
    if output.resolve() == Path(source).resolve():
        raise ValueError("cannot overwrite parent manifest")
    if output.exists():
        if json.loads(output.read_text()) != payload:
            raise ValueError("refusing to overwrite a different manifest")
    else:
        atomic_json(output, payload)
    return output


def pilot_manifest(path):
    manifest = load_manifest(path, expected_training_count=64)
    payload = dict(manifest.payload)
    claimed = payload.pop("manifest_sha256", None)
    if payload.get("format") != FORMAT + "_manifest" or claimed != canonical_digest(payload):
        raise ValueError("invalid v6 subset manifest identity")
    if Counter(r["dataset"] for r in manifest.splits["scorer_train"]) != Counter(
            {name: 32 for name in OFFICIAL_TRAIN_DATASETS}):
        raise ValueError("v6 requires exactly 32 official rows per dataset")
    if set(manifest.splits) != {"scorer_train"}:
        raise ValueError("recovery manifest must contain training only")
    return manifest


def diagnostic_cache_identity(args, manifest, warm):
    old_hash = v5.cache_membership(args.cache_manifest, manifest.splits["scorer_train"])
    hashes = []
    for row in manifest.splits["scorer_train"]:
        path = args.cache / (row["row_hash"] + ".pt")
        record = torch.load(path, map_location="cpu", weights_only=True)
        v5.validate_record(record, row, old_hash, warm["identity"]["backbone"])
        hashes.append((row["row_hash"], digest_file(path)))
    return {"manifest_sha256": old_hash, "records_sha256": canonical_digest(hashes),
            "record_hashes": dict(hashes)}


@torch.no_grad()
def teacher_mass_diagnostic(scorer, entries, cache, record_hashes):
    """Fixed visual fractions, NOT claims about achieved KV-cache reduction."""
    device = next(scorer.parameters()).device
    groups = {}
    for row in entries:
        path = Path(cache) / (row["row_hash"] + ".pt")
        if digest_file(path) != record_hashes[row["row_hash"]]:
            raise ValueError("diagnostic cache bytes changed during run")
        record = torch.load(path, map_location="cpu", weights_only=True)
        scores = scorer(record["vision"].to(device), record["question"].to(device))
        saliency = record["saliency"].float().to(device)
        target = ((saliency - saliency.mean()) / saliency.std(unbiased=False).clamp_min(1e-6)).softmax(0)
        order = torch.argsort(scores, descending=True, stable=True)
        for retention in RETENTIONS:
            k = max(1, int(len(scores) * retention))
            values = {"teacher_target_mass": float(target[order[:k]].sum()),
                      "score_mean": float(scores.mean()), "score_std": float(scores.std(unbiased=False)),
                      "score_above_099_fraction": float((scores > .99).float().mean()),
                      "soft_keep_fraction": float(torch.sigmoid((scores - scores[order[k-1]]) / .1).mean()),
                      "physical_visual_fraction": k / len(scores)}
            for dataset in ("all", row["dataset"]):
                key = dataset + ":" + str(retention)
                groups.setdefault(key, []).append(values)
    return {key: {"rows": len(values), **{name: sum(v[name] for v in values) / len(values)
                                         for name in values[0]}}
            for key, values in groups.items()}


def mass_gate(diagnostic, baseline, minimum_ratio=.8):
    if not baseline or set(diagnostic) != set(baseline) or not 0 < minimum_ratio <= 1:
        raise ValueError("mass gate requires matching nonempty diagnostic groups")
    if any(not math.isfinite(float(group["teacher_target_mass"]))
           for collection in (diagnostic, baseline) for group in collection.values()):
        raise ValueError("mass gate requires finite teacher-target masses")
    ratios = {key: values["teacher_target_mass"] / max(baseline[key]["teacher_target_mass"], 1e-12)
              for key, values in diagnostic.items()}
    return {"minimum_ratio": minimum_ratio, "ratios_to_initial": ratios,
            "pass": all(value >= minimum_ratio for value in ratios.values()),
            "interpretation": "conservative training diagnostic, not generation-quality validation"}


@contextmanager
def runtime(args, manifest, cache_identity):
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError("v6 runtime is process-scoped and cannot nest")
    _ACTIVE = True
    names = ("attention_key_penalty", "pilot_manifest", "source_hashes", "_train_identity", "checkpoint")
    saved = {name: getattr(v5, name) for name in names}

    def sources():
        result = saved["source_hashes"]()
        for path in (Path(__file__), Path(__file__).parents[1] / "llava_paper_mask.py"):
            result[str(path.resolve())] = digest_file(path)
        return result

    def identity(*positional):
        result = saved["_train_identity"](*positional)
        result["recovery_v6"] = {"format": FORMAT, "gate_gradient": args.gate_gradient,
                                  "snapshot_steps": list(SNAPSHOTS), "diagnostic_retentions": list(RETENTIONS),
                                  "minimum_mass_ratio": args.minimum_mass_ratio,
                                  "diagnostic_cache_sha256": cache_identity["records_sha256"],
                                  "source_manifest_sha256": manifest.metadata["source_manifest_sha256"],
                                  "plain_rank32_only": True}
        return result

    def snapshot(scorer, payload):
        next_index = payload["next_index"]
        if next_index in SNAPSHOTS:
            atomic_torch_save(args.output / f"step_{next_index:04d}.pt", payload)
            if next_index == 0:
                atomic_torch_save(args.output / "epoch_zero.pt", payload)
            diagnostic = teacher_mass_diagnostic(scorer, manifest.splits["scorer_train"], args.cache,
                                                 cache_identity["record_hashes"])
            baseline_path = args.output / "diagnostic_0000.json"
            baseline = diagnostic if next_index == 0 else json.loads(baseline_path.read_text())["groups"]
            report = {"format": FORMAT + "_diagnostic", "step": next_index,
                      "identity_sha256": payload["identity_sha256"], "groups": diagnostic,
                      "gate": mass_gate(diagnostic, baseline, args.minimum_mass_ratio)}
            atomic_json(args.output / f"diagnostic_{next_index:04d}.json", report)
            print(json.dumps({"event": "recovery_diagnostic", **report}), flush=True)
            if not report["gate"]["pass"]:
                atomic_json(args.output / "stopped.json", report)
                raise DiagnosticGateFailure(f"teacher-target mass below initial ratio at step {next_index}")

    def save(path, scorer, optimizer, run_identity, next_index, metrics):
        payload = saved["checkpoint"](path, scorer, optimizer, run_identity, next_index, metrics)
        if Path(path).name == "latest.pt":
            snapshot(scorer, payload)
        return payload

    try:
        v5.attention_key_penalty = lambda *a, **kw: attention_key_penalty(
            *a, gate_gradient=args.gate_gradient, **kw)
        v5.pilot_manifest = pilot_manifest
        v5.source_hashes = sources
        v5._train_identity = identity
        v5.checkpoint = save
        yield snapshot
    finally:
        for name, value in saved.items():
            setattr(v5, name, value)
        _ACTIVE = False


def train_run(args):
    if args.recipe != "answer_soft" or args.extract_only or args.adapter not in {"qwen2_5_vl", "llava_next"}:
        raise ValueError("v6 recovery is plain answer_soft on Qwen/LLaVA only")
    if args.gate_gradient not in GATES or not 0 < args.minimum_mass_ratio <= 1:
        raise ValueError("invalid recovery gate configuration")
    if not args.cache or not args.cache_manifest:
        raise ValueError("audited read-only saliency cache is required for recovery diagnostics")
    manifest = pilot_manifest(args.manifest)
    # Prove the deterministic subset came from the specified, unchanged v5 parent.
    if manifest.payload != manifest_payload(args.parent_manifest,
                                            manifest.metadata["selection_seed"]):
        raise ValueError("subset does not match its v5 parent")
    if (args.output / "stopped.json").exists():
        raise DiagnosticGateFailure("run already stopped by diagnostic gate; choose a new run, do not resume")
    warm = load_checkpoint(args.warmstart, device="cpu")
    cache_identity = diagnostic_cache_identity(args, manifest, warm)
    with runtime(args, manifest, cache_identity) as snapshot:
        # Recover an interrupted diagnostic before allowing further optimizer steps.
        latest = args.output / "latest.pt"
        if latest.exists():
            from .selector_models_v5 import load_selector
            scorer, payload = load_selector(latest)
            order = sorted(manifest.splits["scorer_train"], key=lambda r: (r["dataset"], r["row_hash"]))
            random.Random(args.seed).shuffle(order)
            expected = v5._train_identity(args, manifest, warm, order, None)
            if payload["identity"] != expected:
                raise ValueError("resume source/config/data/checkpoint identity changed")
            step = payload["next_index"]
            if not (args.output / "diagnostic_0000.json").exists() and step != 0:
                initial, initial_payload = load_selector(args.output / "epoch_zero.pt")
                snapshot(initial, initial_payload)
            if step in SNAPSHOTS and not (args.output / f"diagnostic_{step:04d}.json").exists():
                snapshot(scorer, payload)
        return v5.train_run(args)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("manifest")
    make.add_argument("--parent-manifest", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--selection-seed", type=int, default=20261017)
    run = sub.add_parser("train", add_help=False)
    run.add_argument("--gate-gradient", choices=GATES, required=True)
    run.add_argument("--parent-manifest", type=Path, required=True)
    run.add_argument("--minimum-mass-ratio", type=float, default=.8)
    known, remaining = parser.parse_known_args(argv)
    if known.command == "manifest":
        if remaining:
            parser.error("unrecognized manifest arguments")
        print(create_manifest(known.parent_manifest, known.output, known.selection_seed))
        return
    args = v5.parse_args(remaining)
    args.gate_gradient, args.parent_manifest = known.gate_gradient, known.parent_manifest
    args.minimum_mass_ratio = known.minimum_mass_ratio
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "training.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another worker owns this recovery output") from exc
        train_run(args)


if __name__ == "__main__":
    main()
