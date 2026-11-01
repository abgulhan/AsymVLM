"""Step 4 (Qwen2.5-VL only): refine the reference scorer with position-corrected
reference-answer attention.

Released Qwen2.5-VL scorer = reference rank-32 scorer (train_scorer.py, 5600 rows)
  -> stage 2: 3 epochs on 100 training rows, target = corrected attention
              (mlsys_sprint.attention_pilot_v19.fit, arm 'corrected'; 300 steps)
  -> stage 3: 3 epochs on 132 rows (the 100 + 32 extra questions on the same
              images), mode 'retrained', loss divided by #questions per image
              (mlsys_sprint.query_selector_v24.fit; 396 steps).
Both stages: AdamW lr 1e-4, wd 0.01, grad-clip 1.0, batch 1, seed 42, shuffle
random.Random(42+epoch).  Only W_v / W_q change; the VLM is frozen.

Target (attention_labels_v19): mean attention from the answer-predicting
positions to each visual patch over the middle half of decoder layers, divided by
an 8x8 spatial prior (mean relative attention per native-geometry cell over the
*other* training images; leave-image-out, divisor clamped at >=1), renormalized.

Phases
  extract  (GPU)  write <labels>/<row_hash>.pt with vision, question, raw_attention,
                  patch_rectangles, image_sha256 for every row of --manifest
  fit      (CPU)  run stage 2 then stage 3 from the reference checkpoint

The fit loop calls the released functions (position_bias, corrected_attention,
evidence_loss, optimizer_step, Rank32Scorer, QuestionScorer); only the protocol
bookkeeping of the original campaign (hash pins, deadlines) is omitted.

    python training/scripts/refine_qwen_scorer.py extract --manifest data/qwen_refine_manifest.json \
        --labels work/qwen25/attention --rank 0 --world-size 1
    python training/scripts/refine_qwen_scorer.py fit --manifest data/qwen_refine_manifest.json \
        --labels work/qwen25/attention --reference work/qwen25/scorer/latest.pt --output work/qwen25/refined
The refine manifest's `scorer_train` split lists the 132 rows; the 32 second
questions carry `paired_with: <row_hash of the first question on that image>`.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from _common import hf_model, load_config, setup_path

setup_path()


def extract(args):
    import torch
    from learned_score.mlsys_scorer.backbone import load_backbone, prepared_inputs, read_entry, frozen_versions, assert_frozen
    from learned_score.mlsys_scorer.checkpoints import atomic_torch_save
    from learned_score.mlsys_scorer.protocol import load_manifest
    from learned_score.mlsys_sprint.attention_pilot_v19 import attention_map
    from learned_score.mlsys_sprint.evidence_annotation_v18 import qwen_patch_rectangles
    config = load_config("qwen25")
    entries = json.loads(args.manifest.read_text())["splits"]["scorer_train"]
    manifest = load_manifest(args.manifest, expected_training_count=len(entries))
    ns = SimpleNamespace(adapter="qwen2_5_vl", model=hf_model(config, args.hf_model),
                         model_revision=config["hf_revision"], max_visual_patches=config["max_visual_patches"],
                         backbone_dtype="bfloat16", device=args.device, limit=None)
    adapter, dataset, _ = load_backbone(ns, manifest)
    versions = frozen_versions(adapter.model)
    args.labels.mkdir(parents=True, exist_ok=True)
    for entry in entries[args.rank::args.world_size]:
        path = args.labels / (entry["row_hash"] + ".pt")
        if path.exists():
            continue
        row = read_entry(dataset, entry)
        with torch.no_grad():
            prepared, visual, question = prepared_inputs(adapter, row, False)   # answer-free scorer inputs
            rectangles = qwen_patch_rectangles(prepared).cpu()
        supervised, _, _ = prepared_inputs(adapter, row, True)                  # reference answer appended
        raw = attention_map(adapter, supervised, supervised.answer_positions - 1)
        atomic_torch_save(path, dict(row_hash=entry["row_hash"], image_sha256=entry["image_sha256"],
                                     paired_with=entry.get("paired_with"), raw_attention=raw,
                                     patch_rectangles=rectangles,
                                     vision=visual.detach().cpu().to(torch.bfloat16),
                                     question=question.detach().cpu().to(torch.bfloat16)))
        assert_frozen(adapter.model, versions)
        print(json.dumps({"event": "attention_label", "row_hash": entry["row_hash"]}), flush=True)


def _epochs(scorer, records, targets, weights, optimizer, epochs=3, seed=42):
    from learned_score.mlsys_scorer.models import optimizer_step
    from learned_score.mlsys_sprint.evidence_learning_v18 import evidence_loss
    steps = 0
    for epoch in range(epochs):
        order = list(range(len(records)))
        random.Random(seed + epoch).shuffle(order)
        for i in order:
            optimizer.zero_grad(set_to_none=True)
            scores = scorer(records[i]["vision"], records[i]["question"])
            optimizer_step(evidence_loss(scores, targets[i]) / weights[i], scorer, optimizer, 1.0)
            steps += 1
    return steps


def fit(args):
    import torch
    from learned_score.mlsys_scorer.checkpoints import atomic_torch_save, load_scorer
    from learned_score.mlsys_sprint import attention_labels_v19 as labels
    from learned_score.mlsys_sprint.query_selector_v24 import FORMAT, QuestionScorer
    torch.set_num_threads(2)
    entries = json.loads(args.manifest.read_text())["splits"]["scorer_train"]
    records = [torch.load(args.labels / (e["row_hash"] + ".pt"), map_location="cpu", weights_only=True) for e in entries]
    originals = [r for r in records if not r.get("paired_with")]          # one question per image
    targets = {}
    for r in records:
        targets[r["row_hash"]] = labels.corrected_attention(
            r, labels.position_bias(originals, r["image_sha256"])).float()
    args.output.mkdir(parents=True, exist_ok=True)

    # Stage 2: corrected-attention warm-start fit on the original (unpaired) rows.
    torch.manual_seed(42)
    scorer, _ = load_scorer(args.reference, "cpu")
    scorer.train().requires_grad_(True)
    opt = torch.optim.AdamW(scorer.parameters(), lr=1e-4, weight_decay=0.01)
    s2 = _epochs(scorer, originals, [targets[r["row_hash"]] for r in originals], [1] * len(originals), opt)
    stage2 = {k: v.detach().cpu() for k, v in scorer.state_dict().items()}
    atomic_torch_save(args.output / "stage2_corrected.pt", dict(state_dict=stage2, steps=s2, hidden_size=scorer.hidden_size))

    # Stage 3: 'retrained' question scorer, image-balanced, on all rows.
    torch.manual_seed(42)
    q = QuestionScorer(scorer.hidden_size, "retrained")
    missing, extra = q.load_state_dict(stage2, strict=False)
    if extra or missing:
        raise ValueError("unexpected warm-start keys")
    q.train().requires_grad_(True)
    opt = torch.optim.AdamW(q.parameters(), lr=1e-4, weight_decay=0.01)
    mult = Counter(r["image_sha256"] for r in records)
    s3 = _epochs(q, records, [targets[r["row_hash"]] for r in records], [mult[r["image_sha256"]] for r in records], opt)
    payload = dict(format=FORMAT, hidden_size=q.hidden_size, mode="retrained",
                   identity=dict(mode="retrained", epochs=3, seed=42, learning_rate=1e-4, weight_decay=0.01,
                                 image_balanced=True, answer_inputs=False, training_ids=[r["row_hash"] for r in records]),
                   epoch=3, steps=s3, state_dict={k: v.detach().cpu() for k, v in q.state_dict().items()})
    atomic_torch_save(args.output / "latest.pt", payload)
    print(json.dumps({"stage2_steps": s2, "stage3_steps": s3, "output": str(args.output / "latest.pt")}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("extract", "fit"))
    parser.add_argument("--manifest", type=Path, required=True, help="refinement manifest (scorer_train = 132 rows)")
    parser.add_argument("--labels", type=Path, required=True, help="attention-label directory")
    parser.add_argument("--reference", type=Path, help="fit: reference scorer latest.pt from train_scorer.py")
    parser.add_argument("--output", type=Path, help="fit: output directory")
    parser.add_argument("--hf-model", help="extract: HF id or local snapshot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args(argv)
    if args.phase == "fit" and (args.reference is None or args.output is None):
        parser.error("fit requires --reference and --output")
    (extract if args.phase == "extract" else fit)(args)


if __name__ == "__main__":
    main()
