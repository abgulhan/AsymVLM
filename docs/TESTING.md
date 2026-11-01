# Testing

The CPU test suite checks the algorithmic pieces of AsymVLM (scorer, visual-token
selection, bounded text-KV cache, bounded generation, the adaptive gate, the model
registry, and repository anonymity). It needs no GPU, no network, and no model
downloads. The whole suite runs in well under two minutes on a laptop CPU
(about 10-15 s).

## Running

```bash
pip install -e .            # or: export PYTHONPATH=src:.
pytest -q                   # uses pytest.ini (testpaths = tests, pythonpath = src .)
pytest -q tests -rs         # also list skipped tests and why
```

Requirements: Python >= 3.10, `torch` (CPU build is enough), `transformers`
(4.51 tested), `numpy`, `pytest`. `tests/conftest.py` puts `src/` and the repo
root on `sys.path` and forces offline, CPU-only execution.

Tests for the public `asymvlm.*` API use `pytest.importorskip` / `hasattr` and
are **skipped** (not failed) when a module or function is not present.
Checkpoint tests are skipped when `checkpoints/<model>/scorer.pt` is absent.

## What each file covers

| File | Covers |
|---|---|
| `tests/test_scorer.py` | `Rank32Scorer` / diagonal scorer on random tensors: output shape `[N]`, FP32 and finite, bounded cosine scores; max-over-question-token semantics (adding a question token equal to a patch raises that patch to the maximum, never lowers any score, equals the per-token max); permutation equivariance over patches and invariance to question-token order; rank-32 projection shapes; invalid-input rejection. Loads each released `checkpoints/{qwen25,llava_next,phi35}/scorer.pt` on CPU, checks `[32, hidden]` projections, finite weights, scores a random input; verifies every entry in `checkpoints/SHA256SUMS`. |
| `tests/test_selection.py` | Reference top-k (`k = ceil(N*r)`): exact count, unique, sorted in original order, includes the arg-max patch, kept scores dominate dropped scores. Runtime `select_tokens` (documented as `floor(N*r)`, min one) for learned / coverage variants, determinism and tie order. DivPrune (`stable_divprune_order`) and CDPruner (`stable_cdpruner_order`) return the requested number of valid unique indices, including degenerate (duplicate / rank-one) features. Optional `asymvlm.pruning` top-k check. |
| `tests/test_cache.py` | `ProtectedPrefixCache` + `RecencyPolicy`: the protected (image+prompt) prefix is never evicted or rewritten; at most `budget (+slack)` generated tokens are retained; retained generated tokens are the most recent, contiguous and in order; original (non-contiguous) source positions are preserved; capacity / memory accounting; invalid configurations; overflow requires explicit compaction. Optional `asymvlm.text_eviction` check. |
| `tests/test_generation.py` | `generate_bounded` on a tiny randomly initialized Llama (2 layers, hidden 32, SDPA) with a fake pruned prefix that keeps non-contiguous original positions: greedy determinism, bounded active cache, prefix positions preserved, compaction happens; with an unbounded budget the output equals a plain Hugging Face `DynamicCache` greedy loop; EOS stops generation at its first occurrence; padded inputs rejected. |
| `tests/test_gate.py` | Pure-numpy reference of the adaptive self-consistency gate: answer normalization; accept the 10% answer iff the normalized answers agree **and** min token probability `>= t`; threshold calibration chooses the minimum mean kept fraction subject to mean positive harm `<= eps` (known answer on a synthetic example + brute-force optimality on random instances); 2-fold cross-calibration never uses a fold to tune its own threshold; scorer-entropy normalization lies in `[0,1]` (uniform -> 1, peaked -> 0). Same properties are checked on `asymvlm.gate` when present. |
| `tests/test_config.py` | `asymvlm.config` registry: aliases, pinned revisions, gate thresholds in `(0,1)`, probe levels below the cap, and that each registered scorer checkpoint matches its backbone hidden size (Qwen2.5-VL-3B 2048, LLaVA-NeXT-7B 4096, Phi-3.5-V 3072) and declared checkpoint format. |

## Optional GPU smoke test

`tests/gpu_smoke.py` is excluded from the CPU suite. It runs one TextVQA-style
image+question (a synthetic "STOP" sign) through each released model at 40%
visual-token keep, then through the adaptive self-consistency gate (probes at
10%/20%, cap 40%). It skips unless CUDA is available, the model snapshot is
already in the local Hugging Face cache, the scorer checkpoint is present, and
the `asymvlm.pruning` / `asymvlm.gate` entry points exist.

```bash
python tests/gpu_smoke.py                       # all three models
python tests/gpu_smoke.py --models llava_next   # one model
pytest -q -s -rs tests/gpu_smoke.py
```

Peak memory is roughly that of the backbone in bf16 (about 8 GB for
Qwen2.5-VL-3B, 16 GB for LLaVA-NeXT-7B and Phi-3.5-V).
