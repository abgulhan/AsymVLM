# AsymVLM visual-token scorers

One small learned scorer per VLM backbone. The VLM itself is frozen; only the scorer is trained.

## Architecture (`learned_score.mlsys_scorer.models.Rank32Scorer`)

Inputs are the VLM's own input embeddings: visual tokens `V` (N x d) and question tokens `Q` (M x d).

    v_i = normalize(W_v · normalize(V_i))      W_v: 32 x d
    q_j = normalize(W_q · normalize(Q_j))      W_q: 32 x d
    score_i = max_j  cos(v_i, q_j)

AsymVLM keeps the top `k = ceil(N * r)` tokens by score, in original order, and physically removes the rest
before prefill. No answer tokens are ever used at selection time. Weights are stored in fp32
(`state_dict` keys `visual.weight` = W_v, `question.weight` = W_q); bias-free.

| file | backbone (HF id @ revision) | d | format | training |
|---|---|---|---|---|
| `qwen25/scorer.pt` | Qwen/Qwen2.5-VL-3B-Instruct @ 66285546d2b821cf421d4f5eb2576359d3770cd3 (max_visual_patches 1280) | 2048 | `mlsys_query_selector_v24`, mode `retrained` | query-aware retraining initialized from the reference scorer: 3 epochs, 396 steps, lr 1e-4, wd 0.01, seed 42, 132 image-balanced training ids, question-only inputs |
| `qwen25/reference_scorer.pt` | same | 2048 | `asymvlm_mlsys_scorer_v1`, variant `rank32` | answer-gradient-saliency objective, 1 epoch, 5600 rows, lr 1e-3, wd 0.01, seed 42, retentions {0.5, 0.7, 0.9} |
| `llava_next/scorer.pt` | llava-hf/llava-v1.6-mistral-7b-hf @ 2424fdd47412fccc66d91719126b420e9fbd7065 | 4096 | `asymvlm_mlsys_scorer_v1`, variant `rank32` | answer-gradient-saliency, 1 epoch, 5600 rows, lr 1e-3, wd 0.01, seed 42, retentions {0.5, 0.7, 0.9} |
| `phi35/scorer.pt` | microsoft/Phi-3.5-vision-instruct @ 12b77fb40b63a2c73c68243d3f767aab688a1b2a (eager attention, 16 crops) | 3072 | `asymvlm_mlsys_scorer_v1`, variant `rank32` | answer-gradient-saliency, 1 epoch, 512 rows (DocVQA single-page + TextVQA), lr 1e-3, wd 0.01, seed 42, retentions {0.3..1.0} |

All backbones were run in bfloat16. Training used frozen backbones (gradient checkpointing, native resolution and
attention backend unchanged).

### Qwen2.5-VL: reference vs. selected scorer

The frozen runtime (`learned_score/mlsys_sprint/budget_execute_v31.py::load_runtime`) first loads the backbone
together with `reference_scorer.pt`, which pins the backbone-integration identity, then swaps in the
query-aware retrained `scorer.pt` that produced every Qwen2.5-VL result in the paper.
`qwen25/complete.json` is the completion marker that `query_selector_v24.load` verifies next to `scorer.pt`.

## Loading

```python
from asymvlm import load_scorer          # CPU is fine
scorer = load_scorer("llava_next")       # or "qwen25", "phi35"
scores = scorer(visual_embeds, question_embeds)   # [N]
```

## Sanitization note

The original payloads recorded the local filesystem path of the Hugging Face snapshot in
`identity.backbone.model_path` (and `identity.reference_identity.backbone.model_path`). For anonymity these
strings were rewritten to `${HF_HOME}/hub/models--<org>--<name>/snapshots/<revision>` and `identity_sha256`
was recomputed with the same canonical-JSON digest the loaders verify. Weights, optimizer state and every
other field are bit-identical to the originals. Consequently the file hashes differ from the original
artifacts (originals: qwen25 2c87bc5c..., llava_next c7665448..., phi35 2900b9a7..., qwen25 reference
f4c21b02...); `identity.reference_checkpoint_sha256` inside `qwen25/scorer.pt` still refers to the original
reference file hash. Use `SHA256SUMS` in this directory to verify the released files.
