"""Group-disjoint, four-benchmark development data for adaptive allocation v4.

The frozen v3 schema is intentionally retained for existing model adapters.
``screening`` now means a separate physical policy-selection panel, not a
subset of evaluation.  Every panel remains development evidence because these
sources and/or historical results have already been inspected.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from learned_score.mlsys_eval.audit import row_identity_keys
from learned_score.mlsys_eval.common import (
    canonical_hash, file_sha256, load_manifest as load_sealed_manifest,
    seal_manifest, write_new_json,
)
from learned_score.mlsys_sprint import adaptive_data_v3 as v3


DEFAULT_OUTPUT = Path("reproduction_results/mlsys_adaptive_20261016_v4")
V3_MANIFEST = Path("reproduction_results/mlsys_adaptive_20261016_v3/data/manifest.json")
FAMILIES = ("DocVQA", "TextVQA", "ChartQA_TEST", "MME")
MME_CATEGORIES = (
    "OCR", "artwork", "celebrity", "code_reasoning", "color",
    "commonsense_reasoning", "count", "existence", "landmark",
    "numerical_calculation", "position", "posters", "scene", "text_translation",
)
DATA_SPLITS = ("fit", "screening", "evaluation")


def identity_keys(rows):
    return {key for row in rows for key in v3._split_keys(row)}


def select_disjoint_rows(rows, count, *, salt, excluded=(), historical_keys=()):
    """One question per image/document group, selected without reading labels.

    Prefer identities not present in the resolved historical audit, then use a
    salted identity hash.  Historical reuse is permitted only as development
    evidence and is recorded explicitly in the manifest audit.
    """
    if count < 0:
        raise ValueError("selection count must be nonnegative")
    if count == 0:
        return []
    forbidden, historical = set(excluded), set(historical_keys)
    ordered = sorted(rows, key=lambda row: (
        bool(row_identity_keys(row) & historical),
        canonical_hash([salt, row["dataset"], row["row_hash"]]),
    ))
    selected = []
    for row in ordered:
        keys = v3._split_keys(row)
        if keys & forbidden:
            continue
        selected.append(dict(row))
        forbidden.update(keys)
        if len(selected) == count:
            return selected
    raise ValueError(f"insufficient disjoint candidates: {len(selected)}/{count}")


def select_mme_pairs(rows, pairs_per_category, *, salt, excluded=(), historical_keys=(),
                     expected_categories=MME_CATEGORIES):
    """Keep complete MME question pairs and require every expected category."""
    if pairs_per_category < 1:
        raise ValueError("MME pairs per category must be positive")
    expected = set(expected_categories)
    actual = {row.get("category") for row in rows}
    if actual != expected:
        raise ValueError(f"MME categories mismatch: missing={sorted(expected - actual)}, "
                         f"extra={sorted(actual - expected)}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["category"], row["group_id"])].append(row)
    forbidden, historical = set(excluded), set(historical_keys)
    candidates = defaultdict(list)
    for (category, group), members in grouped.items():
        if len(members) != 2 or len({row["row_hash"] for row in members}) != 2:
            continue
        if len({row["image_sha256"] for row in members}) != 1:
            raise ValueError(f"MME image pair has mismatched image bytes: {category}/{group}")
        if identity_keys(members) & forbidden:
            continue
        candidates[category].append((group, members))
    # Constrained categories first protects small categories from shared-image
    # collisions in larger ones.  A shortfall is an error, never silently omitted.
    category_order = sorted(expected, key=lambda category: (len(candidates[category]), category))
    selected = []
    for category in category_order:
        ordered = sorted(candidates[category], key=lambda item: (
            any(row_identity_keys(row) & historical for row in item[1]),
            canonical_hash([salt, category, item[0]]),
        ))
        count = 0
        for _, members in ordered:
            keys = identity_keys(members)
            if keys & forbidden:
                continue
            selected.extend(sorted((dict(row) for row in members), key=lambda row: row["row_hash"]))
            forbidden.update(keys)
            count += 1
            if count == pairs_per_category:
                break
        if count != pairs_per_category:
            raise ValueError(f"MME category {category} has only {count}/{pairs_per_category} "
                             "disjoint complete image pairs")
    return selected


def _validate_mme(rows, split, expected_categories=MME_CATEGORIES, expected_pairs=None):
    mme = [row for row in rows if v3.dataset_family(row["dataset"]) == "MME"]
    categories = {row.get("category") for row in mme}
    if categories != set(expected_categories):
        raise ValueError(f"MME category coverage mismatch in {split}")
    groups = defaultdict(list)
    for row in mme:
        groups[(row["category"], row["group_id"])].append(row)
    for (_, group), members in groups.items():
        if len(members) != 2 or len({row["image_sha256"] for row in members}) != 1:
            raise ValueError(f"MME incomplete or inconsistent pair in {split}: {group}")
    per_category = Counter(category for category, _ in groups)
    if len(set(per_category.values())) != 1:
        raise ValueError(f"MME unbalanced category pairs in {split}")
    if expected_pairs is not None and set(per_category.values()) != {expected_pairs}:
        raise ValueError(f"MME pair count mismatch in {split}")


def validate_payload(payload, original_training, *, previous_evaluation=None,
                     expected_families=FAMILIES, expected_mme_categories=MME_CATEGORIES):
    """Fail closed on training leakage or overlap between any experimental panels."""
    metadata, splits = payload["metadata"], payload["splits"]
    if payload.get("format") != "mlsys_adaptive_data_v3" or metadata.get("study_version") != "adaptive_v4":
        raise ValueError("not a v4 development manifest using the v3-compatible schema")
    if set(splits) != {*DATA_SPLITS, "phi_train"}:
        raise ValueError("v4 requires fit, screening, evaluation and phi_train splits")
    v3.validate_payload(payload, original_training)
    for split in DATA_SPLITS:
        rows = splits[split]
        if {v3.dataset_family(row["dataset"]) for row in rows} != set(expected_families):
            raise ValueError(f"benchmark coverage mismatch in {split}")
        if any(not row.get("evaluation_stage", "").startswith("development_") for row in rows):
            raise ValueError("v4 panels must explicitly remain development evidence")
        if "MME" in expected_families:
            _validate_mme(rows, split, expected_mme_categories,
                          metadata.get("mme_pairs_per_category", {}).get(split))
    for index, split in enumerate(DATA_SPLITS):
        keys = identity_keys(splits[split])
        for other in DATA_SPLITS[index + 1:]:
            overlaps = keys & identity_keys(splits[other])
            if overlaps:
                raise ValueError(f"v4 {split}/{other} identity overlap: {sorted(overlaps)[:3]}")
    if previous_evaluation is not None:
        old = {row["row_hash"]: row for row in previous_evaluation}
        new = {row["row_hash"]: row for row in splits["evaluation"]}
        if old.keys() != new.keys():
            raise ValueError("v4 evaluation must preserve every v3 evaluation identity")
        for digest in old:
            for field in ("dataset", "group_id", "image_sha256", "source_file",
                          "source_row_index", "source_shard_sha256"):
                if old[digest].get(field) != new[digest].get(field):
                    raise ValueError(f"changed v3 evaluation provenance: {digest}/{field}")
    return True


def _decorate(rows, split, source):
    result = []
    for row in rows:
        entry = v3._decorate(row, f"development_v4_{split}")
        entry["v4_identity_source"] = source
        entry["historical_evaluation_stage"] = row.get("evaluation_stage", "historical_status_unverified")
        result.append(entry)
    return result


def prepare(output_root=DEFAULT_OUTPUT, *, project_root=None, fit_per_dataset=120,
            selection_per_dataset=60, mme_fit_pairs=4, mme_selection_pairs=2):
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = root / output_root
    previous_path = root / V3_MANIFEST
    previous = load_sealed_manifest(previous_path)
    base_path = root / v3.BASE_MANIFEST
    base = load_sealed_manifest(base_path)
    training = base["splits"]["scorer_train"]
    history_path = root / v3.HISTORY_AUDIT
    history = json.loads(history_path.read_text())
    historical_keys = set(history.get("identity_keys", []))
    evaluation = _decorate(previous["splits"]["evaluation"], "previously_examined_evaluation",
                           "unchanged_v3_evaluation_identity")
    excluded = identity_keys(training + evaluation)
    fit, selection = [], []
    source_counts = {}
    for family in ("DocVQA", "TextVQA"):
        candidates = [row for row in previous["splits"]["fit"]
                      if v3.dataset_family(row["dataset"]) == family]
        source_counts[family] = len(candidates)
        chosen = select_disjoint_rows(candidates, fit_per_dataset, salt=f"adaptive-v4-{family}-fit",
                                      excluded=excluded)
        fit.extend(_decorate(chosen, "fit", "previously_inspected_v3_fit"))
        excluded.update(identity_keys(chosen))
        chosen = select_disjoint_rows(candidates, selection_per_dataset,
                                      salt=f"adaptive-v4-{family}-physical-selection", excluded=excluded)
        selection.extend(_decorate(chosen, "physical_selection", "previously_inspected_v3_fit"))
        excluded.update(identity_keys(chosen))
    pools = {}
    for dataset in ("ChartQA_TEST", "MME"):
        rows = v3._identity_tsv(root / ".data/LMUData" / f"{dataset}.tsv", dataset)
        # Hash before partitioning: duplicate chart images can have different
        # filenames, and cross-benchmark duplicate bytes must remain together.
        v3._hash_tsv_images(rows, root)
        pools[dataset] = rows
        source_counts[dataset] = len(rows)
    chart = pools["ChartQA_TEST"]
    for count, target, split in ((fit_per_dataset, fit, "fit"),
                                 (selection_per_dataset, selection, "physical_selection")):
        chosen = select_disjoint_rows(chart, count, salt=f"adaptive-v4-chart-{split}",
                                      excluded=excluded, historical_keys=historical_keys)
        target.extend(_decorate(chosen, split, "chart_test_development_historical_audit_incomplete"))
        excluded.update(identity_keys(chosen))
    for count, target, split in ((mme_fit_pairs, fit, "fit"),
                                 (mme_selection_pairs, selection, "physical_selection")):
        chosen = select_mme_pairs(pools["MME"], count, salt=f"adaptive-v4-mme-{split}",
                                 excluded=excluded, historical_keys=historical_keys)
        target.extend(_decorate(chosen, split, "mme_test_development_historical_audit_incomplete"))
        excluded.update(identity_keys(chosen))
    order = lambda row: (v3.dataset_family(row["dataset"]), row["row_hash"])
    splits = {"fit": sorted(fit, key=order), "screening": sorted(selection, key=order),
              "evaluation": sorted(evaluation, key=order), "phi_train": previous["splits"]["phi_train"]}
    metadata = {
        "study_version": "adaptive_v4", "schema_compatibility": "mlsys_adaptive_data_v3",
        "source_catalog_path": base["metadata"].get("source_catalog_path"),
        "source_catalog_sha256": base["metadata"].get("source_catalog_sha256"),
        "original_training_manifest": str(base_path),
        "original_training_manifest_sha256": file_sha256(base_path),
        "previous_manifest": str(previous_path), "previous_manifest_sha256": file_sha256(previous_path),
        "historical_audit_path": str(history_path), "historical_audit_sha256": file_sha256(history_path),
        "historical_audit_complete": False,
        "historical_unresolved_artifacts": len(history.get("unresolved", [])),
        "historical_resolver_reported_complete": bool(history.get("complete")),
        "split_roles": {"fit": "fit budget-response models; image groups disjoint from other panels",
                        "screening": "independent physical policy selection; no evaluation labels",
                        "evaluation": "previously examined development panel; never fresh confirmation",
                        "phi_train": "unchanged original official scorer training subset"},
        "counts": {split: len(rows) for split, rows in splits.items()},
        "counts_by_benchmark": {split: dict(Counter(v3.dataset_family(row["dataset"]) for row in rows))
                                for split, rows in splits.items()},
        "unique_groups_by_benchmark": {
            split: {family: len({row["group_id"] for row in rows
                                if v3.dataset_family(row["dataset"]) == family}) for family in FAMILIES}
            for split, rows in splits.items()},
        "mme_pairs_per_category": {"fit": mme_fit_pairs, "screening": mme_selection_pairs,
                                    "evaluation": previous["metadata"]["mme_pairs_per_category"]},
        "mme_categories": list(MME_CATEGORIES), "source_candidate_counts": source_counts,
        "resolved_historical_matches": {
            split: dict(Counter(v3.dataset_family(row["dataset"]) for row in rows
                                if row_identity_keys(row) & historical_keys))
            for split, rows in splits.items()},
        "label_selection": "identity-only; no question, answer, prediction, or quality used to select rows",
        "historical_note": "All panels are development; excluded scorer training and v3 evaluation "
                           "from both fitting and physical selection. Chart/MME source test splits "
                           "are explicitly repurposed as disjoint development panels.",
        "pairwise_split_identity_intersections": {"fit_screening": 0, "fit_evaluation": 0,
                                                  "screening_evaluation": 0},
        "scorer_training_identity_intersections": {split: 0 for split in DATA_SPLITS},
    }
    payload = seal_manifest({"format": "mlsys_adaptive_data_v3", "dataset": "ASYMVLM_ADAPTIVE_V4",
                             "splits": splits, "metadata": metadata})
    validate_payload(payload, training, previous_evaluation=previous["splits"]["evaluation"])
    path = output_root / "data/manifest.json"
    write_new_json(path, payload)
    summary = {"manifest": str(path), "manifest_sha256": file_sha256(path), **metadata}
    write_new_json(output_root / "data/summary.json", summary)
    return summary


def load_manifest(path):
    manifest = v3.load_manifest(path)
    if manifest.metadata.get("study_version") != "adaptive_v4":
        raise ValueError("not an adaptive v4 manifest")
    return manifest


class ManifestDatasetV4(v3.ManifestDatasetV3):
    name = "ASYMVLM_ADAPTIVE_V4"

    def __init__(self, manifest_path, allowed_splits=DATA_SPLITS, *, project_root=None):
        # The selection alias is only an API convenience, never a duplicated
        # manifest split that could undermine pairwise disjointness checks.
        load_manifest(manifest_path)
        allowed_splits = tuple("screening" if split == "selection" else split for split in allowed_splits)
        super().__init__(manifest_path, allowed_splits, project_root=project_root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
