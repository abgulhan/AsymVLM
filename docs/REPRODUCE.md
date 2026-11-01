# Reproducing the paper results

All commands run from the repository root with `PYTHONPATH=src:.`. Every GPU script is
batch-1 (except the systems batch sweep), resumable (existing per-request JSONs are
skipped), and shardable over GPUs with `--shard i --num-shards n` (launch one process per
GPU with `CUDA_VISIBLE_DEVICES=i`; no scheduler is required).

Environment variables (all optional):

| variable | meaning | default |
|---|---|---|
| `ASYMVLM_ROOT` | repository root | parent of `evaluation/` |
| `ASYMVLM_DATA` | dataset root; manifest `source_file`s resolve here | `$ASYMVLM_ROOT/data` |
| `ASYMVLM_OUT` | raw generations and per-request records | `$ASYMVLM_ROOT/runs` |
| `HF_HOME` | Hugging Face cache for backbones and datasets | HF default |

Backbones (pinned revisions in `evaluation/configs/campaign.json`): `Qwen/Qwen2.5-VL-3B-Instruct`,
`llava-hf/llava-v1.6-mistral-7b-hf`, `microsoft/Phi-3.5-vision-instruct`. Scorers:
`checkpoints/{qwen25,llava_next,phi35}/scorer.pt` (Qwen2.5-VL additionally uses
`checkpoints/qwen25/reference_scorer.pt` if present; see `evaluation/common.py:build_config`).
Package versions: torch 2.6.0, transformers 4.51.3, accelerate 1.4.0 (also pinned in the config).

## 1. Data

| dataset | source | split used |
|---|---|---|
| TextVQA | HF `lmms-lab/textvqa` | validation |
| DocVQA | HF `pixparse/docvqa-single-page-questions` | validation |
| ChartQA | ChartQA test (HF `HuggingFaceM4/ChartQA`, or VLMEvalKit `ChartQA_TEST.tsv`) | test |
| MME | HF `lmms-lab/MME` (or VLMEvalKit `MME.tsv`) | test |
| InfographicVQA | VLMEvalKit `InfoVQA_VAL.tsv` (or HF `lmms-lab/DocVQA`, config `InfographicVQA`) | validation |
| OCRVQA | VLMEvalKit `OCRVQA_TESTCORE.tsv` (test-core subset) | test-core |
| olmOCR-Bench | HF `allenai/olmOCR-bench` (revision in `results/panels/olmocr_joint_panel.json`) | bench PDFs |

```bash
export ASYMVLM_DATA=$PWD/data
huggingface-cli download lmms-lab/textvqa --repo-type dataset --local-dir $ASYMVLM_DATA/textvqa
huggingface-cli download pixparse/docvqa-single-page-questions --repo-type dataset --local-dir $ASYMVLM_DATA/docvqa
huggingface-cli download lmms-lab/MME --repo-type dataset --local-dir $ASYMVLM_DATA/mme
huggingface-cli download allenai/olmOCR-bench --repo-type dataset --local-dir $ASYMVLM_DATA/olmocr/bench
# VLMEvalKit TSVs (ChartQA_TEST, InfoVQA_VAL, OCRVQA_TESTCORE): download from the VLMEvalKit
# dataset URLs into $ASYMVLM_DATA/vlmeval/
```

### Manifest format

The evaluation manifest (`$ASYMVLM_DATA/manifest.json`, format `mlsys_adaptive_data_v3`)
is a JSON object `{dataset, format, metadata, splits: {evaluation: [...]}, manifest_sha256}`.
Each row identifies one question without copying the data:

```
dataset, dataset_revision, source_file, source_format (parquet|tsv), source_split,
source_row_index, source_shard_sha256, question_id, question_sha256, image_id,
image_sha256, group_id (image/document identity; used for 2-fold gate calibration),
metric (vqa|anls|relaxed|mme|exact), panel_order (locked order; first N = panel),
row_hash (request id)
```

Build it with the `learned_score` data modules (`learned_score.mlsys_sprint.adaptive_data_v3`,
`manuscript_data_v35`) or, if you received our manifest, relocate its `source_file`s:

```bash
python evaluation/tools/relocate_manifest.py manifest.json $ASYMVLM_DATA/manifest.json \
    --map <old_prefix>=$ASYMVLM_DATA/
```

Hashes are checked at load time, so a relocated manifest selects exactly the paper examples.
The canonical benchmark panel is the first 512 rows per dataset in `panel_order`; the
public gate/latency panel (256/dataset) is `results/panels/public_panel.json`.
The olmOCR manifest is built with `learned_score.mlsys_eval.olmocr.build_olmocr_manifest`
(pages rendered at longest side 2048); the 64 joint pages are listed in
`results/panels/olmocr_joint_panel.json`.

## 2. Paper artifact -> command

| paper artifact | command(s) | GPU cost (A100-class, rough) |
|---|---|---|
| Comparison figure (6 benchmarks + average, 3 models), quality tables 30/50/70 (`quality30/50/70`, `paired_intervals50`) | `evaluation/run_benchmark.py --model M` (all methods, keep 10..90, 512/dataset) then `make_tables.py aggregate`, `make_tables.py quality --keep K`, `make_figures.py` | largest cost: ~3k requests x (1 + 6 methods x 9 levels) per model; ~150-300 GPU-h total over 3 models |
| Adaptive self-consistency gate (adaptive table, stars in comparison figure, teaser) | `evaluation/gate/capture.py --model M --panel first512` (10%/20% passes with per-token probabilities), then `evaluation/gate/evaluate.py` (2-fold group cross-calibration; needs uniform 10..90 + all-token records from step above), `make_tables.py gate`, `make_figures.py --gate ...` | ~0.5 GPU-h per model for 3k requests |
| Gate latency vs uniform-40 | `evaluation/gate/latency.py --model M --per-dataset 8` | < 0.5 GPU-h per model |
| Teaser TTFT/E2E (all tokens, uniform 30/20, gate at 70/80% removal) | `evaluation/gate/ttft.py --model M --gate70 CAP,T --gate80 CAP,T` (CAP,T from `gate/evaluate.py`) | < 0.5 GPU-h per model |
| Joint vision + generated-text eviction on olmOCR-Bench (joint table) | `evaluation/joint/generate.py --model {qwen25,llava_next} --olmocr-manifest ... --panel combined64`, then `evaluation/joint/score.py --evaluator-root ...`, `make_tables.py joint` | ~100 s/request; 64 pages x 4 cells x 2 models ~ 14 GPU-h |
| Systems table (TTFT, E2E, KV MiB, peak memory, batch 1/4/8) | `evaluation/systems/measure.py --model M` (dedicated idle GPU), `make_tables.py systems` | ~1-2 GPU-h per model |

Example full benchmark on 4 GPUs for one model:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i PYTHONPATH=src:. python evaluation/run_benchmark.py \
      --model qwen25 --per-dataset 512 --shard $i --num-shards 4 &
done; wait
python evaluation/make_tables.py aggregate && python evaluation/make_figures.py
```

## 3. Shipped results (`results/`)

Small, sanitized summaries of the runs reported in the paper (no per-request dumps):

- `paper/{main,additional,adaptive,joint}-results.json` - evidence files the tables were built from.
- `comparison_512/*.json` - audits of the comparison figure (per-cell means, n, paired intervals).
- `teaser/*.audit.json`, `tables/*.audit.json` - teaser and table audits (systems table included).
- `gate/latency/<model>.json`, `gate/ttft/<model>.json` - gate latency / TTFT summaries.
- `gate/public/*.json` - gate calibration/evaluation reports on the public panel.
- `joint/panel_dev32_<model>.json`, `joint/panel_new32_combined64_<model>.json` - olmOCR joint reports.
- `panels/` - request-id lists for the public panel and the olmOCR joint panel.

`evaluation/tools/sanitize_results.py` was used to strip machine-specific fields.

## 4. Notes and limitations

- Baselines are native-feature adaptations/ports (DivPrune, CDPruner, VisionZip, FastV,
  VL-Cache), not upstream end-to-end releases; fidelity notes are in
  `evaluation/configs/campaign.json` (`baseline_fidelity`). FastV/VL-Cache cells whose layer
  schedule cannot meet a budget are recorded as `unsupported_budget`, never substituted.
- The Phi-3.5 VisionZip port hashes the upstream VisionZip `clip_encoder.py` for provenance;
  set `VISIONZIP_SRC` to a checkout of the VisionZip repository.
- olmOCR scoring needs the official olmOCR evaluator checkout, Playwright browsers
  (`PLAYWRIGHT_BROWSERS_PATH`) and an equation-render cache (`OLMOCR_EQUATION_CACHE_DIR`).
- Timings depend on the GPU; quality numbers are deterministic (greedy) up to kernel
  nondeterminism. The paper runs used one A100-class GPU per process.
