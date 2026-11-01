"""Shared configuration helpers for the evaluation CLIs.

All locations are configurable; nothing is hard-coded to a particular machine.

Environment variables (all optional):
  ASYMVLM_ROOT   repository root (default: parent of this directory)
  ASYMVLM_DATA   root that relative ``source_file`` entries of the manifest resolve
                 against (default: $ASYMVLM_ROOT/data)
  ASYMVLM_OUT    where raw generations / per-request records are written
                 (default: $ASYMVLM_ROOT/runs)
  HF_HOME        standard Hugging Face cache; backbones are resolved from it
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get('ASYMVLM_ROOT', Path(__file__).resolve().parents[1])).resolve()
DATA = Path(os.environ.get('ASYMVLM_DATA', ROOT/'data')).resolve()
OUT = Path(os.environ.get('ASYMVLM_OUT', ROOT/'runs')).resolve()
CONFIG = Path(__file__).with_name('configs')/'campaign.json'

MODELS = ('qwen25', 'llava_next', 'phi35')
DATASETS = ('TextVQA', 'DocVQA', 'ChartQA', 'MME', 'InfographicVQA', 'OCRVQA')
PERCENTS = (10, 20, 30, 40, 50, 60, 70, 80, 90)
# Methods evaluated per model in the main comparison (all_tokens is the 100% reference).
METHODS = {
    'qwen25': ('asymvlm_uniform', 'divprune', 'cdpruner', 'visionzip_native', 'fastv_native', 'vlcache_native'),
    'llava_next': ('asymvlm_uniform', 'divprune', 'cdpruner', 'visionzip_llava_native', 'fastv_native', 'vlcache_native'),
    'phi35': ('asymvlm_uniform', 'divprune', 'cdpruner', 'visionzip_phi_native', 'fastv_native', 'vlcache_native'),
}
FAMILY = {'ChartQA_TEST': 'ChartQA', 'OCRVQA_TESTCORE': 'OCRVQA'}


def family(name):
    return FAMILY.get(name, name)


def add_common_args(p: argparse.ArgumentParser, *, model_required=True):
    p.add_argument('--model', choices=MODELS, required=model_required)
    p.add_argument('--manifest', type=Path, default=DATA/'manifest.json',
                   help='evaluation manifest (see docs/REPRODUCE.md, "Manifest format")')
    p.add_argument('--checkpoints', type=Path, default=ROOT/'checkpoints',
                   help='directory holding <model>/scorer.pt')
    p.add_argument('--out', type=Path, default=OUT, help='output root for raw generations and records')
    p.add_argument('--shard', type=int, default=0, help='shard index (0-based)')
    p.add_argument('--num-shards', type=int, default=1, help='total shards; run one process per GPU')
    return p


def resolve_backbone(model_id, revision):
    """Return a local snapshot directory under HF_HOME (downloads if allowed)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, revision=revision,
                             local_files_only=os.environ.get('HF_HUB_OFFLINE') == '1')


def build_config(model, *, manifest, checkpoints, out, training_manifest=None):
    """Build the runtime config consumed by ``learned_score`` (``Runtime(config, model)``).

    ``configs/campaign.json`` holds the frozen, path-free settings used in the paper
    (backbone ids/revisions, generation settings, baseline-fidelity notes, hashes).
    """
    cfg = copy.deepcopy(json.loads(CONFIG.read_text()))
    spec = cfg['models'][model]
    spec['model_path'] = resolve_backbone(spec['model_id'], spec['revision'])
    ckpt = Path(checkpoints)/model
    spec['checkpoint'] = str(ckpt/'scorer.pt')
    # Qwen2.5-VL's query-aware scorer is loaded on top of a reference scorer that
    # fixes the backbone feature contract (see learned_score.mlsys_sprint.budget_execute_v31).
    if (ckpt/'reference_scorer.pt').exists():
        spec['reference_checkpoint'] = str(ckpt/'reference_scorer.pt')
    cfg.update(manifest=str(Path(manifest).resolve()), cwd=str(DATA), output_root=str(Path(out).resolve()),
               scorer_training_manifest=str(training_manifest or (DATA/'scorer_training_manifest.json')))
    return cfg


def load_entries(manifest, split='evaluation'):
    return json.loads(Path(manifest).read_text())['splits'][split]


def first_panel(entries, per_dataset=512, datasets=None):
    """Canonical panel: first ``per_dataset`` rows per dataset in locked ``panel_order``."""
    grouped = {}
    for e in entries:
        if datasets and family(e['dataset']) not in datasets:
            continue
        grouped.setdefault(e['dataset'], []).append(e)
    return {d: sorted(v, key=lambda e: (e['panel_order'], e['row_hash']))[:per_dataset] for d, v in sorted(grouped.items())}


def interleave(panels):
    """Round-robin over datasets so partial runs stay balanced."""
    order, n = [], max((len(v) for v in panels.values()), default=0)
    for i in range(n):
        for d in sorted(panels):
            if i < len(panels[d]):
                order.append(panels[d][i])
    return order


def shard(items, index, count):
    if not 0 <= index < count:
        raise ValueError('--shard must be in [0, --num-shards)')
    return [x for i, x in enumerate(items) if i % count == index]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value))
    tmp.replace(path)


PUBLIC_PANEL = ROOT/'results'/'panels'/'public_panel.json'


def select_panel(entries, panel='first512', per_dataset=512):
    """``first512``: canonical first-N per dataset; ``public``: the fixed 256/dataset id list."""
    entries = list(entries)
    if panel == 'first512':
        return first_panel(entries, per_dataset)
    by_id = {e['row_hash']: e for e in entries}
    ids = json.loads(PUBLIC_PANEL.read_text())['panels']
    return {d: [by_id[r] for r in rids if r in by_id] for d, rids in sorted(ids.items())}
