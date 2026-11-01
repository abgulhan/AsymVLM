# Installation

Tested with Python 3.11, PyTorch 2.6.0 (CUDA 12.4 wheels), transformers 4.51.3 on NVIDIA A100 GPUs.
Visual-token selection and the checkpoints load on CPU; generation experiments need a GPU.

## 1. Environment

```bash
conda env create -f environment.yml      # creates env "asymvlm" (Python 3.11 + pinned pip deps)
conda activate asymvlm
pip install -e .                          # installs `asymvlm`, `evaluation`, and the frozen `learned_score` package from src/
```

Alternatively: `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pip install -e .`

## 2. Environment variables

| variable | meaning | default |
|---|---|---|
| `ASYMVLM_ROOT` | repository / project root used for relative data and result paths | `.` (frozen code) or the repo root (`asymvlm`) |
| `ASYMVLM_CHECKPOINTS` | scorer checkpoint directory | `$ASYMVLM_ROOT/checkpoints` |
| `HF_HOME` | Hugging Face cache (model snapshots are resolved under `$HF_HOME/hub`) | `~/.cache/huggingface` |
| `ASYMVLM_PHI35_SNAPSHOT` | explicit local Phi-3.5-vision snapshot directory (optional) | `$HF_HOME/hub/models--microsoft--Phi-3.5-vision-instruct/snapshots/<rev>` |
| `ASYMVLM_CLIP_SNAPSHOT` | CLIP ViT-L/14-336 snapshot for the CDPruner-CLIP baseline (optional) | `$HF_HOME/hub/models--openai--clip-vit-large-patch14-336/snapshots/<rev>` |
| `ASYMVLM_VLMEVAL_DATASETS` | VLMEvalKit `DATASETS` directory (OCRVQA TESTCORE TSV) | `data/VLMEvalKit/DATASETS` |

## 3. Model weights (exact revisions)

```bash
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct       --revision 66285546d2b821cf421d4f5eb2576359d3770cd3
huggingface-cli download llava-hf/llava-v1.6-mistral-7b-hf  --revision 2424fdd47412fccc66d91719126b420e9fbd7065
huggingface-cli download microsoft/Phi-3.5-vision-instruct  --revision 12b77fb40b63a2c73c68243d3f767aab688a1b2a
# optional, CDPruner-CLIP baseline only:
huggingface-cli download openai/clip-vit-large-patch14-336 --revision ce19dc912ca5cd21c8a653c79e251e808ccabcd1
```

Runtime settings used in the paper: bfloat16; Qwen2.5-VL and LLaVA-NeXT with SDPA attention
(Qwen `max_visual_patches=1280`); Phi-3.5-vision with eager attention and 16 crops (loaded with
`trust_remote_code=True` from the local snapshot); greedy decoding, `max_new_tokens=128`,
`generated_budget=512`; protected-recency text eviction with budget 256 for long generation.

The trained scorers ship in `checkpoints/` (verify with `cd checkpoints && sha256sum -c SHA256SUMS`).

## 4. Quick CPU check

```bash
python -c "import asymvlm, learned_score.mlsys_scorer.models; print('ok')"
python -c "
from asymvlm import MODELS, load_scorer
for m in MODELS:
    s = load_scorer(m); print(m, type(s).__name__, s.hidden_size)"
```

## 5. One-command GPU check

```bash
python examples/quickstart.py                    # Qwen2.5-VL-3B on a synthetic image; prints uniform + adaptive answers
python examples/quickstart.py --model llava_next --image my_page.png --question "What is the title?"
```

## 6. Minimal GPU usage (Python API)

```python
from asymvlm import load_model_and_scorer, prepare, prune_and_generate, self_consistency_gate
rt = load_model_and_scorer("qwen25")                     # downloads/loads the pinned snapshot
prepared = prepare(rt, "example.jpg", "What is the title of this document?")
text, result = prune_and_generate(rt, prepared, keep_ratio=0.4)   # AsymVLM uniform 40%
text, kept = self_consistency_gate(rt, prepared)                  # adaptive 10/20 -> 40 gate
```
