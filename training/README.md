# training/

Scripts to retrain the AsymVLM rank-32 query-aware visual-token scorers from
scratch, as for the released checkpoints. Full documentation, hyperparameter
table, compute estimate and verification recipe: [`docs/TRAINING.md`](../docs/TRAINING.md).

```
training/
  configs/{qwen25,llava_next,phi35}.json   hyperparameters read from released checkpoints
  scripts/prepare_training_manifest.py     1. validate manifest / derive Phi 512-row subset (CPU)
  scripts/extract_targets.py               2. answer-gradient saliency with the frozen VLM (GPU)
  scripts/train_scorer.py                  3. fit W_v, W_q (rank 32) on cached targets (CPU/GPU)
  scripts/refine_qwen_scorer.py            4. Qwen2.5-VL: position-corrected attention refinement
  calibrate_adaptive_gate.py               self-consistency gate threshold calibration (CPU)
```

All scripts are thin wrappers over `src/learned_score` (no reimplemented math).
Run from the repo root with `PYTHONPATH=src` (or set `ASYMVLM_ROOT`); models are
fetched by pinned Hugging Face revision through `HF_HOME`, or pass
`--hf-model /path/to/snapshot` / `ASYMVLM_MODEL_PATH`.

Summary of the recipe: AdamW, weight decay 0.01, grad clip 1.0, batch 1, seed 42.
LLaVA-NeXT and Qwen2.5-VL reference: 5600 rows, 1 epoch, lr 1e-3. Phi-3.5-V:
512 rows, 1 epoch, lr 1e-3. Qwen2.5-VL release: + 3 epochs on 100 rows and
3 epochs on 132 rows of position-corrected answer attention, lr 1e-4.

## Released training manifests (row lists)
`training/manifests/` contains the exact rows used to train the released scorers (paths removed):
- `scorer_train_llava_qwen_reference.json` — split `scorer_train` (5,600 rows: 2,800 DocVQA + 2,800 TextVQA) for LLaVA-NeXT and the Qwen2.5-VL reference scorer; `scorer_dev` (1,000) for validation.
- `scorer_train_phi35.json` — split `phi_train` (512 rows) for Phi-3.5-Vision.
- `qwen_query_refinement_v24.json` — split `scorer_train` (132 rows: 100 originals + 32 second questions on the same images) for the final Qwen refinement stage.
Each row keeps dataset id/revision, source split and row index, question/image hashes and `row_hash`, so rows can be rebuilt from the public datasets and verified. The `original_manifest_sha256` field matches the manifest hashes recorded in the released checkpoints.
