# Training the AsymVLM scorers from scratch

The scorer is a rank-32, two-projection, query-aware patch scorer
(`learned_score.mlsys_scorer.models.Rank32Scorer`; for Qwen2.5-VL the
checkpoint is saved as `learned_score.mlsys_sprint.query_selector_v24.QuestionScorer`
in mode `retrained`, which is the same forward pass):

    s_i = max_j < normalize(W_v * normalize(v_i)), normalize(W_q * normalize(q_j)) >

`v_i` are the frozen visual-token embeddings, `q_j` the frozen question-token
embeddings (answer-free). Only `W_v`, `W_q` (32 x H each, no bias) are trained;
the VLM stays frozen (asserted by `frozen_versions/assert_frozen`).

**Loss.** Cross-entropy between a target importance distribution `p` and the
softmax of the standardized scores:
`L = -sum_i p_i log softmax(z(s))_i`, with `z(x) = (x - mean) / std`.
* `mlsys_scorer.models.saliency_loss`: `p = softmax(z(saliency))` (stage 1).
* `mlsys_sprint.evidence_learning_v18.evidence_loss`: `p` is an already
  normalized distribution (Qwen refinement stages).

**Targets.**
* Answer-gradient saliency (`multimodel.saliency.compute_answer_gradient_saliency`;
  Phi uses the equivalent `mlsys_sprint.phi_adaptive_v3.phi_saliency`): add a
  zero bias to the attention logits of every visual key, backpropagate the
  reference-answer cross-entropy, and take `ReLU(-dL/dbias_i)` (z-scored):
  how much increasing attention to patch `i` would reduce the answer loss.
* Qwen2.5-VL refinement (`mlsys_sprint.attention_labels_v19`): attention from the
  answer-predicting positions to each visual patch, averaged over heads and the
  middle half of decoder layers, divided by a leave-image-out 8x8 spatial prior
  (mean relative attention per native-geometry cell over the other training
  images; divisor clamped at >= 1), then renormalized. This removes recurring
  position "sinks" from the attention target.

## Pipeline

| Step | Script | Hardware | Wraps |
|---|---|---|---|
| 1. Scorer-training manifest | `training/scripts/prepare_training_manifest.py` | CPU | `mlsys_scorer.protocol.load_manifest` |
| 2. Target extraction | `training/scripts/extract_targets.py` | GPU, frozen VLM | `mlsys_scorer extract`, `phi_adaptive_v3 --phase extract` |
| 3. Fit rank-32 scorer | `training/scripts/train_scorer.py` | CPU or 1 GPU | `mlsys_scorer train`, `phi_adaptive_v3 --phase train` |
| 4. Qwen refinement | `training/scripts/refine_qwen_scorer.py` | GPU (extract), CPU (fit) | `attention_labels_v19`, `attention_pilot_v19.attention_map`, `evidence_loss`, `QuestionScorer` |

### 1. Manifest
`scorer_train` = 5600 rows from the official **train** splits only:
2800 `pixparse/docvqa-single-page-questions` + 2800 `lmms-lab/textvqa`.
`load_manifest` rejects any validation/extra rows and any image, document group
or row that also appears in a development/evaluation split. Phi-3.5-V uses a
512-row subset (256 per dataset, split `phi_train`). The Qwen refinement uses a
132-row manifest (100 rows + 32 second questions on the same images,
`paired_with` field), 66 DocVQA / 66 TextVQA, also train-only.

### 2. Targets (per row, `<row_hash>.pt`)
`vision` [N, H] bf16, `question` [T, H] bf16, `saliency` [N] fp32.
Restartable; shard with `--rank/--world-size`.

### 3. Fit
One example per step (batch size 1), AdamW, grad-norm clip 1.0, FP32 scorer,
deterministic shuffle `random.Random(seed + epoch)`.

### 4. Qwen refinement
Stage 2 (warm start from the stage-1 reference scorer): 3 epochs on 100 rows with
corrected-attention targets. Stage 3: 3 epochs on 132 rows, loss divided by the
number of questions sharing the image (image-balanced).

## Hyperparameters (read from the released checkpoint payloads)

| | Qwen2.5-VL-3B (reference) | Qwen refine stage 2 | Qwen refine stage 3 (released) | LLaVA-NeXT-7B (Mistral) | Phi-3.5-V |
|---|---|---|---|---|---|
| HF revision | `66285546d2b8` | | | `2424fdd47412` | `12b77fb40b63` |
| Resolution | max 1280 visual patches | same | same | native any-res grid | 16 crops |
| Target | answer-gradient | corrected attention | corrected attention | answer-gradient | answer-gradient |
| Rows | 5600 | 100 | 132 | 5600 | 512 |
| Init | random (seed 42) | stage 1 | stage 2 | random (seed 42) | random (seed 42) |
| LR | 1e-3 | 1e-4 | 1e-4 | 1e-3 | 1e-3 |
| Weight decay | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| Epochs | 1 | 3 | 3 | 1 | 1 |
| Steps | 5600 | 300 | 396 | 5600 | 512 |
| Batch | 1 | 1 | 1 | 1 | 1 |
| Seed | 42 | 42 | 42 | 42 | 42 |
| Rank / H | 32 / 2048 | 32 / 2048 | 32 / 2048 | 32 / 4096 | 32 / 3072 |
| Grad clip | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

Machine-readable copies: `training/configs/{qwen25,llava_next,phi35}.json`.
Sources: the `identity` dict in each checkpoint (`seed`, `epochs`,
`learning_rate`, `weight_decay`, `rows`, `objective`, `variant`, backbone
revision), `complete.json` step counts, and the trainer code
(`mlsys_scorer/__main__.py::train`, `phi_adaptive_v3.py::train`,
`attention_pilot_v19.py::fit`, `query_selector_v24.py::fit`).

## Commands

```bash
export ASYMVLM_ROOT=$PWD PYTHONPATH=$PWD/src HF_HOME=/path/to/hf_cache
M=data/manifest.json
python training/scripts/prepare_training_manifest.py --manifest $M --phi-subset data/phi_manifest.json

# LLaVA-NeXT (same for qwen25 = Qwen reference stage)
for r in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$r python training/scripts/extract_targets.py \
   --model llava_next --manifest $M --output work/llava_next/targets --rank $r --world-size 4 & done; wait
python training/scripts/train_scorer.py --model llava_next --manifest $M \
   --cache work/llava_next/targets --output work/llava_next/scorer

# Phi-3.5-V
python training/scripts/extract_targets.py --model phi35 --manifest data/phi_manifest.json --output work/phi35/targets
python training/scripts/train_scorer.py --model phi35 --manifest data/phi_manifest.json \
   --cache work/phi35/targets --output work/phi35/scorer

# Qwen2.5-VL refinement
python training/scripts/refine_qwen_scorer.py extract --manifest data/qwen_refine_manifest.json --labels work/qwen25/attention
python training/scripts/refine_qwen_scorer.py fit --manifest data/qwen_refine_manifest.json \
   --labels work/qwen25/attention --reference work/qwen25/scorer/latest.pt --output work/qwen25/refined
```

## Compute estimate
Target extraction dominates (one forward + backward through the frozen VLM per
row, with decoder gradient checkpointing; bf16, eager attention). Rough
single-A100-40GB figures: Qwen2.5-VL-3B ~1-2 s/row (5600 rows: ~2-3 GPU-h);
LLaVA-NeXT-7B ~3-5 s/row (~5-8 GPU-h); Phi-3.5-V ~2-4 s/row (512 rows: <0.5
GPU-h); Qwen attention labels (forward with `output_attentions`) 132 rows: <0.2
GPU-h. Fitting the scorer is cheap: minutes on CPU or one GPU (5600 steps of a
2x32xH model). Target caches: ~N x H x 2 bytes per row (a few MB/row for
LLaVA-NeXT; ~15-30 GB total for 5600 rows). These are estimates, not logged
measurements.

## Expected outputs
* `work/<model>/scorer/latest.pt`: format `asymvlm_mlsys_scorer_v1`, keys
  `identity`, `variant='rank32'`, `hidden_size`, `state_dict`
  (`visual.weight`, `question.weight`, each [32, H]), `optimizer`, `epoch=1`,
  `steps` (5600 / 512); plus `complete.json`, `run_metadata.json`,
  `epoch_zero.pt`, `epoch_001.pt`.
* `work/qwen25/refined/latest.pt`: format `mlsys_query_selector_v24`,
  `mode='retrained'`, `epoch=3`, `steps=396`.
* Released checkpoints map as: `checkpoints/llava_next/scorer.pt`,
  `checkpoints/phi35/scorer.pt` <- step 3; `checkpoints/qwen25/reference_scorer.pt`
  <- step 3 (Qwen); `checkpoints/qwen25/scorer.pt` <- step 4.

## Verifying against the released checkpoints
Bit-exact reproduction is not expected (GPU nondeterminism in bf16 backward
passes changes the targets slightly; library versions differ). Compare
*rankings*, which is what pruning consumes:

```python
import torch
from learned_score.mlsys_scorer.models import Rank32Scorer, select_tokens
def load(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    m = Rank32Scorer(d["hidden_size"]); m.load_state_dict(d["state_dict"], strict=False); return m.eval()
ref, new = load("checkpoints/llava_next/scorer.pt"), load("work/llava_next/scorer/latest.pt")
rec = torch.load("work/llava_next/targets/<row_hash>.pt")    # held-out or training row
with torch.no_grad():
    a, b = ref(rec["vision"], rec["question"]), new(rec["vision"], rec["question"])
ka, kb = (set(select_tokens(x, rec["vision"], 0.3).tolist()) for x in (a, b))
print("top-30% overlap", len(ka & kb) / len(ka))   # Spearman of a vs b is also useful
```
Expect high top-k overlap (substantially above the random baseline equal to the
keep ratio) and Spearman correlation on a few dozen images, and benchmark
accuracy within run-to-run noise when the retrained scorer is substituted in the
evaluation scripts.

## Adaptive gate calibration
`training/calibrate_adaptive_gate.py` (CPU) picks the self-consistency gate
threshold `t`: run 10% and 20% keep; if normalized answers agree and the minimum
per-token top-1 probability >= `t`, keep 10%, else use the uniform cap `K`.
`--eps` mode chooses the smallest mean kept % with mean positive harm vs the
uniform-`K` policy <= eps; `--target-kept` maximizes quality under a mean-kept
budget (searching caps). `--crossfit` fits on one image-group fold
(sha256(group) mod 2) and applies to the other. See the script docstring for
the record format.

## Notes on exact reproduction
* The exact training rows are released in `training/manifests/` (see below). For Phi-3.5-Vision use
  `scorer_train_phi35.json` (split `phi_train`); the 256+256 subset derived by `prepare_training_manifest.py`
  is a deterministic alternative, not the original row selection.
* `refine_qwen_scorer.py` uses the released label and loss functions and extracts embeddings and attention
  in a single pass (the original reused stage-1 visual embeddings for the 100 base rows), so refitted weights
  agree closely with, but are not bit-identical to, the released checkpoint.

## Released training manifests (row lists)
`training/manifests/` contains the exact rows used to train the released scorers (paths removed):
- `scorer_train_llava_qwen_reference.json` — split `scorer_train` (5,600 rows: 2,800 DocVQA + 2,800 TextVQA) for LLaVA-NeXT and the Qwen2.5-VL reference scorer; `scorer_dev` (1,000) for validation.
- `scorer_train_phi35.json` — split `phi_train` (512 rows) for Phi-3.5-Vision.
- `qwen_query_refinement_v24.json` — split `scorer_train` (132 rows: 100 originals + 32 second questions on the same images) for the final Qwen refinement stage.
Each row keeps dataset id/revision, source split and row index, question/image hashes and `row_hash`, so rows can be rebuilt from the public datasets and verified. The `original_manifest_sha256` field matches the manifest hashes recorded in the released checkpoints.
