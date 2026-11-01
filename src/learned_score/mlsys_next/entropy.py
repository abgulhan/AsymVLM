"""Answer-free normalized-entropy extraction from completed frozen rank32 scorers.

One model/image preparation serves both seeds. Each seed has independently bound,
atomic shards. The parquet reader explicitly excludes all answer columns and does
not open the historical source catalog. Only scorer_dev is permitted.
"""

import argparse
import hashlib
import io
import json
import math
from contextlib import contextmanager
from pathlib import Path

from learned_score.mlsys_eval.common import canonical_hash, file_sha256, load_manifest, write_new_json


FORMULA = "H(softmax((s-mean(s))/max(std_population(s),1e-8)))/log(number_of_patches)"


class AnswerFreeDataset:
    name = "ASYMVLM_NATIVE_MIXTURE"

    def __init__(self, manifest_path, *, project_root=None):
        self.payload = load_manifest(manifest_path)
        self.metadata, self.splits = self.payload["metadata"], self.payload["splits"]
        self.root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        self.allowed = {entry["row_hash"]: entry for entry in self.splits["scorer_dev"]}
        self._verified = set()
        self._table, self._path = None, None

    def row_for_entry(self, entry):
        if self.allowed.get(entry["row_hash"]) != entry:
            raise ValueError("answer-free extraction permits only explicit scorer_dev entries")
        path = (self.root / entry["source_file"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("source path escapes the project")
        expected = entry.get("source_shard_sha256")
        if not expected:
            raise ValueError("unbound source shard")
        if path not in self._verified:
            if file_sha256(path) != expected:
                raise ValueError("source shard hash mismatch")
            self._verified.add(path)
        if self._path != path:
            import pyarrow.parquet as pq
            self._table = pq.read_table(path, columns=["image", "question_id", "question"])
            self._path = path
        row = self._table.slice(int(entry["source_row_index"]), 1).to_pylist()[0]
        if str(row["question_id"]) != str(entry["question_id"]):
            raise ValueError("source question identity mismatch")
        image_bytes = row["image"]["bytes"]
        if hashlib.sha256(image_bytes).hexdigest() != entry["image_sha256"]:
            raise ValueError("source image hash mismatch")
        return {"image": io.BytesIO(image_bytes), "question": row["question"],
                "_manifest_row_hash": entry["row_hash"], "group_id": entry["group_id"],
                "_source_dataset": entry["dataset"]}

    def build_prompt(self, row):
        return [{"type": "image", "value": row["image"]},
                {"type": "text", "value": row["question"] +
                 "\nAnswer the question using a single word or phrase."}]


@contextmanager
def blind_dataset_factory(manifest_path):
    from learned_score.mlsys_scorer import backbone
    original = backbone.make_dataset
    backbone.make_dataset = lambda manifest, allowed_splits=None: AnswerFreeDataset(manifest_path)
    try:
        yield
    finally:
        backbone.make_dataset = original


def normalized_entropy(scores):
    import torch
    if scores.ndim != 1 or scores.numel() < 2 or not torch.isfinite(scores).all():
        raise ValueError("entropy requires at least two finite scores")
    values = scores.detach().float()
    standardized = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-8)
    probabilities = torch.softmax(standardized, dim=0)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum() / math.log(values.numel())
    return min(1.0, max(0.0, float(entropy)))


def validate_feature(row, entry, identity):
    if row.get("identity_sha256") != canonical_hash(identity):
        raise ValueError("feature identity differs from extraction metadata")
    for key, value in {"request_id": entry["row_hash"], "group_id": entry["group_id"],
                       "dataset": entry["dataset"], "image_sha256": entry["image_sha256"],
                       "seed": identity["seed"], "model": identity["backbone"]["model"]}.items():
        if row.get(key) != value:
            raise ValueError(f"feature {key} mismatch")
    entropy = row.get("normalized_entropy")
    if isinstance(entropy, bool) or not isinstance(entropy, (int, float)) or not math.isfinite(entropy) or not 0 <= entropy <= 1:
        raise ValueError("invalid entropy feature")
    if not isinstance(row.get("visual_patches"), int) or row["visual_patches"] < 2:
        raise ValueError("invalid feature patch count")
    if any(key in row for key in ("answer", "answers", "quality_score", "prediction")):
        raise ValueError("label-bearing entropy shard")


def collect_features(directory, manifest_path):
    directory = Path(directory)
    manifest = load_manifest(manifest_path)
    identities = [json.loads(path.read_text()) for path in sorted(directory.glob("rank*.metadata.json"))]
    if not identities or any(value != identities[0] for value in identities):
        raise ValueError("missing or inconsistent feature rank metadata")
    identity = identities[0]
    if (identity.get("format") != "mlsys_answer_free_entropy_v1" or identity.get("split") != "scorer_dev"
            or identity.get("manifest_sha256") != file_sha256(manifest_path)
            or identity.get("labels_read") is not False or identity.get("formula") != FORMULA):
        raise ValueError("unsupported or unbound answer-free extraction")
    entries = sorted(manifest["splits"]["scorer_dev"], key=lambda item: (item["dataset"], item["row_hash"]))
    if identity.get("limit") is not None:
        entries = entries[:identity["limit"]]
    expected = {entry["row_hash"]: entry for entry in entries}
    rows = {}
    for path in sorted((directory / "rows").glob("*.json")):
        row = json.loads(path.read_text())
        request = row.get("request_id")
        if request not in expected or request in rows or path.name != request + ".json":
            raise ValueError("unexpected feature shard")
        validate_feature(row, expected[request], identity)
        rows[request] = row
    for rank in range(identity["world_size"]):
        path = directory / f"rank{rank}.complete.json"
        if not path.is_file():
            raise ValueError("feature extraction is incomplete")
        marker = json.loads(path.read_text())
        requests = [entry["row_hash"] for entry in entries[rank::identity["world_size"]]]
        hashes = {request: file_sha256(directory / "rows" / (request + ".json")) for request in requests}
        if marker != {"identity_sha256": canonical_hash(identity), "rank": rank, "row_sha256": hashes}:
            raise ValueError("feature completion integrity mismatch")
    if set(rows) != set(expected):
        raise ValueError("missing feature requests")
    return identity, rows


def extract(args):
    import torch
    from learned_score.mlsys_scorer.backbone import load_backbone, prepared_inputs, frozen_versions, assert_frozen
    from learned_score.mlsys_scorer.checkpoints import load_scorer
    from learned_score.mlsys_scorer.protocol import load_manifest as training_manifest, entries_for_split
    from learned_score.mlsys_sprint.benchmark import require_completed_checkpoint, validate_checkpoint_identity
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("invalid rank/world size")
    manifest = training_manifest(args.manifest)
    entries = entries_for_split(manifest, "scorer_dev")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("smoke limit must be positive")
        entries = entries[:args.limit]
    with blind_dataset_factory(args.manifest):
        adapter, dataset, backbone = load_backbone(args, manifest)
    model_versions = frozen_versions(adapter.model)
    scorers = []
    for checkpoint in args.checkpoint:
        scorer, payload = load_scorer(checkpoint, device=args.device)
        require_completed_checkpoint(checkpoint, payload)
        compatibility = validate_checkpoint_identity(payload, manifest.file_sha256, backbone)
        if payload["variant"] != "rank32" or payload["identity"]["seed"] not in (42, 43):
            raise ValueError("this fixed diagnostic uses only existing rank32 seeds 42 and 43")
        scorer.eval().requires_grad_(False)
        seed = payload["identity"]["seed"]
        if any(record[0]["seed"] == seed for record in scorers):
            raise ValueError("duplicate scorer seed")
        identity = {"format": "mlsys_answer_free_entropy_v1", "split": "scorer_dev",
                    "manifest_sha256": manifest.file_sha256, "checkpoint_sha256": file_sha256(checkpoint),
                    "checkpoint_identity_sha256": payload["identity_sha256"], "seed": seed,
                    "backbone": backbone, "checkpoint_compatibility": compatibility,
                    "formula": FORMULA, "labels_read": False, "world_size": args.world_size,
                    "limit": args.limit, "source_sha256": file_sha256(__file__)}
        output = Path(args.output) / f"seed{seed}"
        write_new_json(output / f"rank{args.rank}.metadata.json", identity)
        scorers.append((identity, scorer, output, frozen_versions(scorer)))
    for entry in entries[args.rank::args.world_size]:
        todo = []
        for identity, scorer, output, versions in scorers:
            path = output / "rows" / (entry["row_hash"] + ".json")
            if path.exists():
                validate_feature(json.loads(path.read_text()), entry, identity)
            else:
                todo.append((identity, scorer, path, versions))
        if not todo:
            continue
        with torch.no_grad():
            row = dataset.row_for_entry(entry)
            prepared, vision, question = prepared_inputs(adapter, row, include_answer=False)
            for identity, scorer, path, versions in todo:
                scores = scorer(vision, question)
                value = {"identity_sha256": canonical_hash(identity), "request_id": entry["row_hash"],
                         "dataset": entry["dataset"], "group_id": entry["group_id"],
                         "image_sha256": entry["image_sha256"], "seed": identity["seed"],
                         "model": backbone["model"], "normalized_entropy": normalized_entropy(scores),
                         "visual_patches": int(scores.numel())}
                validate_feature(value, entry, identity)
                write_new_json(path, value)
                assert_frozen(scorer, versions)
            del prepared, vision, question, scores, row
        assert_frozen(adapter.model, model_versions)
        print(json.dumps({"request_id": entry["row_hash"], "rank": args.rank, "seeds": [value[0]["seed"] for value in todo]}), flush=True)
    for identity, scorer, output, versions in scorers:
        hashes = {entry["row_hash"]: file_sha256(output / "rows" / (entry["row_hash"] + ".json"))
                  for entry in entries[args.rank::args.world_size]}
        write_new_json(output / f"rank{args.rank}.complete.json",
                       {"identity_sha256": canonical_hash(identity), "rank": args.rank, "row_sha256": hashes})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--adapter", choices=("qwen2_5_vl", "llava_next"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--max-visual-patches", type=int, default=1280)
    parser.add_argument("--backbone-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--attention-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    extract(parser.parse_args(argv))


if __name__ == "__main__":
    main()
