"""Step 1: validate (and optionally subset) the scorer-training manifest.

The released checkpoints were trained on a fixed manifest whose `scorer_train`
split holds 5600 official-*train* rows (2800 DocVQA single-page + 2800 TextVQA);
every image/document group in that split is disjoint from all evaluation,
development and calibration splits.  This script runs the same checks the
trainers run (`learned_score.mlsys_scorer.protocol.load_manifest`) and can write
the 512-row Phi-3.5-V subset (256 rows per dataset, `phi_train` split).

    python training/scripts/prepare_training_manifest.py --manifest data/manifest.json
    python training/scripts/prepare_training_manifest.py --manifest data/manifest.json \
        --phi-subset data/phi_manifest.json
"""
import argparse
import hashlib
import json
from pathlib import Path

from _common import setup_path

setup_path()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="scorer-training manifest JSON")
    parser.add_argument("--phi-subset", type=Path, help="write a 256+256-row phi_train manifest here")
    parser.add_argument("--seed", type=int, default=42, help="subset selection seed (hash-ordered)")
    args = parser.parse_args(argv)

    from learned_score.mlsys_scorer.protocol import OFFICIAL_TRAIN_DATASETS, load_manifest
    manifest = load_manifest(args.manifest)  # raises on overlap / wrong counts / non-train rows
    train = manifest.splits["scorer_train"]
    print(json.dumps({"manifest_sha256": manifest.file_sha256, "scorer_train_rows": len(train),
                      "splits": {k: len(v) for k, v in manifest.splits.items()}}, indent=1))
    if args.phi_subset:
        rows = []
        for dataset in sorted(OFFICIAL_TRAIN_DATASETS):
            pool = [e for e in train if e["dataset"] == dataset]
            key = lambda e: hashlib.sha256(f"{args.seed}:{e['row_hash']}".encode()).hexdigest()
            rows += sorted(pool, key=key)[:256]
        payload = dict(manifest.payload)
        payload["splits"] = {"phi_train": rows}
        payload.setdefault("metadata", {})["derived_from_sha256"] = manifest.file_sha256
        args.phi_subset.write_text(json.dumps(payload, indent=1))
        print(f"wrote {len(rows)} phi_train rows to {args.phi_subset}")


if __name__ == "__main__":
    main()
