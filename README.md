# AsymVLM — supplementary code (anonymous submission)

AsymVLM treats vision and text tokens differently during VLM inference:

* **Vision tokens** are pruned *before prefill* by a lightweight learned, query-aware scorer
  (two learned 32-dim projections; score = max cosine similarity over question tokens).
  Top-k patches are kept in their original order (**AsymVLM uniform**).
* **AsymVLM adaptive** (self-consistency gate): run the pruned model at 10% and 20% of visual tokens;
  if both answers agree and are confident (min answer-token probability ≥ t), return the 10% answer,
  otherwise run at a cap (default 40%). One threshold `t`, no training, calibrated on held-out data.
  A training-free single-pass alternative uses the entropy of the scorer's patch scores.
* **Generated text** in long outputs is bounded by protected-prefix recency eviction: retained image
  and prompt tokens are always kept, plus the most recent N generated tokens.

This repository contains the exact runtime used for the paper's results, the trained scorer checkpoints,
scripts to retrain them from scratch, scripts that regenerate every reported result, and CPU tests.

## Repository layout

| Path | Contents |
|---|---|
| `src/learned_score/` | Frozen research runtime that produced all paper results (adapters, scorer, pruning, bounded/evicting KV cache generation, baselines, evaluators). |
| `asymvlm/` | Small public API on top of it: `pruning`, `gate`, `text_eviction`, `config`. |
| `checkpoints/` | Trained scorers for Qwen2.5-VL-3B, LLaVA-NeXT-7B, Phi-3.5-Vision (+ `SHA256SUMS`, `MODEL_CARD.md`). |
| `training/` | Train the scorers from scratch (target extraction, scorer fit, Qwen refinement), released training row lists, gate calibration. |
| `evaluation/` | Benchmarks and baselines, adaptive gate capture/calibration, latency/TTFT, joint vision+text eviction (olmOCR-Bench), table/figure builders. |
| `results/` | Small summary JSONs behind the paper's tables and figures. |
| `tests/` | CPU test suite (`pytest`), optional GPU smoke test. |
| `examples/` | `quickstart.py`: one-command end-to-end example. |
| `docs/` | `INSTALL.md`, `TESTING.md`, `TRAINING.md`, `REPRODUCE.md`. |

## Quick start

```bash
conda env create -f environment.yml        # or: python -m venv .venv && pip install -r requirements.txt
conda activate asymvlm
pip install -e .
pytest -q tests                            # CPU only, ~1-2 minutes, no model downloads
python examples/quickstart.py              # one GPU: downloads Qwen2.5-VL-3B, prints uniform + adaptive answers
```

See `docs/INSTALL.md` for model downloads (pinned Hugging Face revisions) and environment variables
(`ASYMVLM_ROOT`, `HF_HOME`, dataset locations).

### Minimal usage

```python
from asymvlm import pruning, gate
rt = pruning.load_model_and_scorer("qwen25")                # frozen VLM + released scorer (GPU)
print(pruning.answer(rt, "page.png", "What is the invoice number?", keep_ratio=0.4))   # AsymVLM uniform
prepared = pruning.prepare(rt, "page.png", "What is the invoice number?")
print(gate.self_consistency_gate(rt, prepared))             # AsymVLM adaptive -> (answer, kept_percent)
```

## Reproducing the paper

`docs/REPRODUCE.md` maps every table and figure to the command that regenerates it, with expected GPU
hours. `results/` holds the summaries the paper reports, so the tables and figures can be rebuilt on CPU
with `python evaluation/make_tables.py` and `python evaluation/make_figures.py`.

## Training from scratch

`docs/TRAINING.md`: extract target importance labels with the frozen VLM, fit the rank-32 scorer
(AdamW, lr 1e-3, 1 epoch, seed 42), and the two-stage Qwen refinement. Exact training rows are in
`training/manifests/`. Expect close agreement with the released checkpoints (ranking overlap), not
bit-identical weights.

## Notes

* Released checkpoints are metadata-sanitized copies of the originals: model paths were rewritten, and
  the weights are unchanged. `checkpoints/MODEL_CARD.md` lists both hashes.
* Baselines (DivPrune, CDPruner, VisionZip, FastV, VL-Cache) are native-feature re-implementations
  inside this runtime, not the upstream end-to-end code.
* GPU results in the paper used A100-40GB, batch size 1, greedy decoding.
* Kept-token count is `ceil(N * keep_ratio)` (as in all paper results and `asymvlm.pruning.keep_count`);
  the older helper `learned_score.mlsys_scorer.models.select_tokens` uses floor and is not used for results.
* The released Qwen2.5-VL scorer (query-aware refinement, format `mlsys_query_selector_v24`) is loaded through
  `asymvlm.pruning.load_scorer` / the runtime; `checkpoints/qwen25/reference_scorer.pt` is the stage-1 scorer.
* Long-generation text eviction (`asymvlm.text_eviction`) supports Qwen2.5-VL and LLaVA-NeXT, the two models in
  the paper's joint olmOCR study; Phi-3.5-Vision is not supported by that cache path.
* The adaptive gate compares answers after lowercasing and stripping non-alphanumerics (articles are kept),
  exactly as in the reported experiments.
* Raw per-request records written by the runtime include the local hostname; sanitize with
  `evaluation/tools/sanitize_results.py` before sharing.
* Verified on one A100: `tests/gpu_smoke.py`, the `asymvlm` API (uniform, adaptive gate, long-generation
  text eviction) on Qwen2.5-VL, and token-exact parity of `evaluation/run_benchmark.py` with the paper's
  recorded outputs (3/3 requests). Because released checkpoints are metadata-sanitized, the runtime's
  identity check ignores only the rewritten `${HF_HOME}` paths and source hashes (revision, resolution,
  dtype and training manifest are still enforced).

## License

The code and the released scorer checkpoints in this repository are under the MIT License (`LICENSE`).
Third-party components keep their own licenses and are not redistributed here: the base models
Qwen2.5-VL-3B-Instruct, LLaVA-NeXT (llava-v1.6-mistral-7b-hf), Phi-3.5-Vision-Instruct, and the
optional CLIP ViT-L/14-336 are downloaded from Hugging Face under their respective model licenses, and
the evaluation datasets (DocVQA, TextVQA, ChartQA, InfoVQA, OCRVQA, olmOCR-Bench) are used under their own
terms. Baseline re-implementations follow the methods described in the cited papers.
