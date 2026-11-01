"""New Phi-3.5-Vision integration; never alters the historical Phi adapter.

The native, pinned HF snapshot supplies the model/processor. Image row separators,
the global-crop separator, BOS and all prompt tokens are protected. Only actual
2x2-merged visual patches are prunable. Training uses the existing rank-32
answer-gradient objective; no historical Phi scorer is reused.
"""

import os
import argparse
from contextlib import contextmanager
from functools import partial
import inspect
import json
from pathlib import Path
import random
import re
from types import SimpleNamespace

import torch
from torch import nn

from learned_score.mlsys_scorer.checkpoints import (
    atomic_json, atomic_torch_save, canonical_digest, load_checkpoint, save_checkpoint,
)
from learned_score.mlsys_scorer.protocol import digest_file
from learned_score.multimodel.adapters.base import PreparedBatch, PrunedBatch, VLMAdapter


MODEL_ID = "microsoft/Phi-3.5-vision-instruct"
DEFAULT_MODEL = os.environ.get("ASYMVLM_PHI35_SNAPSHOT", os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub", "models--microsoft--Phi-3.5-vision-instruct/snapshots/"
                 "12b77fb40b63a2c73c68243d3f767aab688a1b2a"))
TRAIN_DATASETS = {"pixparse/docvqa-single-page-questions", "lmms-lab/textvqa"}
IMAGE_TAG = re.compile(r"<\|image_(\d+)\|>")


def phi_image_layout(input_ids, image_sizes):
    """Return prunable positions, protected separators and patch crop identities.

    Native Phi packs a stitched sub-crop grid (12 merged patches/crop edge),
    one newline per grid row, one global separator, then the global 12x12 grid.
    The geometry, rather than embedding equality, identifies separators.
    """
    ids = torch.as_tensor(input_ids)
    if ids.ndim == 2 and ids.shape[0] == 1:
        ids = ids[0]
    if ids.ndim != 1:
        raise ValueError("Phi layout requires one unpadded input sequence")
    sizes = torch.as_tensor(image_sizes).reshape(-1, 2).tolist()
    if not sizes:
        raise ValueError("Phi requires at least one image")
    all_visual, all_protected, all_groups = [], [], []
    for image_index, (height, width) in enumerate(sizes, 1):
        if int(height) != height or int(width) != width or min(height, width) < 336 or height % 336 or width % 336:
            raise ValueError("Phi processed image sizes must be positive multiples of 336")
        h, w = int(height) // 336, int(width) // 336
        source = (ids == -image_index).nonzero(as_tuple=False).flatten()
        sub_count = h * 12 * (w * 12 + 1)
        expected = sub_count + 1 + 12 * 13
        if len(source) != expected or bool((source[1:] != source[:-1] + 1).any()):
            raise ValueError("Phi image placeholders do not match native crop packing")
        protected, groups = [], []
        for row in range(h * 12):
            for col in range(w * 12 + 1):
                newline = col == w * 12
                protected.append(newline)
                groups.append(-1 if newline else (row // 12) * w + col // 12 + 1)
        protected.append(True)
        groups.append(-1)
        for row in range(12):
            for col in range(13):
                protected.append(col == 12)
                groups.append(-1 if col == 12 else 0)
        mask = torch.tensor(protected, device=ids.device, dtype=torch.bool)
        all_visual.append(source[~mask])
        all_protected.append(source[mask])
        # Pair image index and crop index; unlike a flat boundary this never
        # mistakes BOS or crop separator tokens for selectable image patches.
        all_groups.extend((image_index, group) for group in groups if group >= 0)
    negative = ((ids < 0) & (ids > -1_000_000_000)).sum().item()
    if negative != sum(len(v) + len(p) for v, p in zip(all_visual, all_protected)):
        raise ValueError("unaccounted Phi image-token placeholders")
    return torch.cat(all_visual), torch.cat(all_protected), all_groups


def _plain_span(tokenizer, text, start, end):
    """Token positions inside one text chunk, with verified slow fallback."""
    try:
        encoded = tokenizer(text, return_offsets_mapping=True)
        positions = [i for i, (left, right) in enumerate(encoded["offset_mapping"])
                     if right > left and right > start and left < end]
    except (TypeError, ValueError, KeyError, NotImplementedError):
        encoded = tokenizer(text)
        full = encoded["input_ids"]
        before = tokenizer(text[:start])["input_ids"]
        through = tokenizer(text[:end])["input_ids"]
        common = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        positions = list(range(common(full, before), common(full, through)))
    return list(encoded["input_ids"]), positions


def phi_text_span_positions(tokenizer, prompt, start, end, input_ids):
    """Follow the native processor's separately tokenized image-delimited chunks."""
    ids = torch.as_tensor(input_ids).reshape(-1).tolist()
    if not 0 <= start < end <= len(prompt):
        raise ValueError("invalid question/answer character span")
    cursor = char_cursor = 0
    selected = []
    for match in [*IMAGE_TAG.finditer(prompt), None]:
        char_end = match.start() if match else len(prompt)
        chunk = prompt[char_cursor:char_end]
        chunk_ids, local = _plain_span(tokenizer, chunk, start - char_cursor, end - char_cursor)
        if ids[cursor:cursor + len(chunk_ids)] != chunk_ids:
            raise ValueError("Phi text span tokenization differs from native processor")
        if start < char_end and end > char_cursor:
            selected.extend(cursor + i for i in local)
        cursor += len(chunk_ids)
        if match:
            image_id = -int(match.group(1))
            image_start = cursor
            while cursor < len(ids) and ids[cursor] == image_id:
                cursor += 1
            if cursor == image_start:
                raise ValueError("Phi prompt image placeholder has no image tokens")
            char_cursor = match.end()
    if cursor != len(ids) or not selected or any(ids[i] < 0 for i in selected):
        raise ValueError("Phi semantic span is empty, incomplete, or includes image tokens")
    return torch.tensor(selected, device=torch.as_tensor(input_ids).device, dtype=torch.long)


class PhiRuntimeModel(nn.Module):
    """Expose native Phi through the current bounded-cache runner's interface.

    Native Phi computes RoPE from explicit position_ids and appends to HF Cache;
    cache_position is therefore accepted, checked, and not forwarded. No model
    weights, attention operations, or positions are changed.
    """
    def __init__(self, native):
        super().__init__()
        self.native = native
        self.config = native.config
        self.generation_config = getattr(native, "generation_config", None)

    @property
    def model(self):
        return self.native.model

    @property
    def lm_head(self):
        return self.native.lm_head

    def forward(self, *, logits_to_keep=0, cache_position=None, return_dict=True, **kwargs):
        if not return_dict:
            raise ValueError("Phi v3 wrapper requires return_dict=True")
        if cache_position is not None:
            inputs = kwargs.get("inputs_embeds", kwargs.get("input_ids"))
            length = inputs.shape[-2] if inputs.ndim == 3 else inputs.shape[-1]
            if cache_position.ndim != 1 or len(cache_position) != length:
                raise ValueError("invalid physical cache positions")
        output = self.native.model(return_dict=True, **kwargs)
        hidden = output.last_hidden_state
        if logits_to_keep:
            hidden = hidden[:, -int(logits_to_keep):]
        return SimpleNamespace(logits=self.native.lm_head(hidden).float(),
                               past_key_values=output.past_key_values,
                               hidden_states=getattr(output, "hidden_states", None),
                               last_hidden_state=output.last_hidden_state)


@contextmanager
def visual_key_bias(model, bias):
    """Scope additive key-bias hooks to one teacher forward/backward only."""
    handles = []
    def add_bias(module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None or mask.ndim != 4 or mask.shape[-1] != len(bias):
            raise ValueError("Phi saliency requires an explicit full causal attention mask")
        return args, {**kwargs, "attention_mask": mask + bias.to(mask.dtype)[None, None, None, :]}
    try:
        for layer in model.model.layers:
            handles.append(layer.self_attn.register_forward_pre_hook(add_bias, with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def checkpointed_decoder(model, enabled=True):
    """Recompute frozen decoder layers without enabling attention dropout."""
    decoder = model.model
    if not enabled:
        yield
        return
    from torch.utils.checkpoint import checkpoint
    before = (decoder.training, getattr(decoder, "gradient_checkpointing", False),
              getattr(decoder, "_gradient_checkpointing_func", None))
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Phi teacher must be frozen")
    if any(module.training for module in model.modules()):
        raise ValueError("Phi teacher must start in evaluation mode")
    try:
        decoder.training = True
        decoder.gradient_checkpointing = True
        decoder._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        yield
    finally:
        decoder.training, decoder.gradient_checkpointing = before[:2]
        if before[2] is None:
            del decoder._gradient_checkpointing_func
        else:
            decoder._gradient_checkpointing_func = before[2]


class PhiAdaptiveAdapterV3(VLMAdapter):
    name, version = "phi3_vision", "mlsys_adaptive_v3"

    def __init__(self, dataset=None, device="cuda:0"):
        self.dataset, self.device = dataset, torch.device(device)

    def load_model_and_processor(self, config):
        from transformers import AutoModelForCausalLM, AutoProcessor
        path = Path(config.model).resolve()
        if not path.is_dir():
            raise ValueError("Phi v3 requires an existing pinned local snapshot")
        dtype_name = getattr(config, "backbone_dtype", "bfloat16")
        if dtype_name not in {"float16", "bfloat16"}:
            raise ValueError("Phi backbone dtype must be float16 or bfloat16")
        self.processor = AutoProcessor.from_pretrained(
            str(path), trust_remote_code=True, local_files_only=True,
            num_crops=getattr(config, "num_crops", 16))
        native = AutoModelForCausalLM.from_pretrained(
            str(path), trust_remote_code=True, local_files_only=True,
            torch_dtype=getattr(torch, dtype_name), attn_implementation="eager",
            low_cpu_mem_usage=True).to(self.device)
        self.model = PhiRuntimeModel(native).eval().requires_grad_(False)
        self.hidden_size = int(native.config.hidden_size)
        self.snapshot_path = path
        self.dtype_name = dtype_name
        self.num_crops = int(getattr(config, "num_crops", 16))
        return self

    def prepare_sample(self, row, include_answer=False):
        from PIL import Image
        from learned_score.eval_llava_adapter import split_message
        from learned_score.mlsys_sprint.benchmark import parse_references
        if self.dataset is None:
            raise ValueError("Phi adapter requires the audited dataset prompt builder")
        image_path, question = split_message(self.dataset.build_prompt(row))
        if hasattr(image_path, "seek"):
            image_path.seek(0)
        image = Image.open(image_path).convert("RGB")
        prompt = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": "<|image_1|>\n" + question}],
            tokenize=False, add_generation_prompt=True)
        question_start = prompt.index(question)
        answer = None
        answer_start = None
        if include_answer:
            answers = parse_references(row.get("answer", ""))
            if not answers:
                raise ValueError("Phi saliency requires a training reference answer")
            answer = answers[0]
            answer_start = len(prompt)
            prompt += answer
        inputs = self.processor(prompt, [image], return_tensors="pt")
        dtype = next(self.model.parameters()).dtype
        inputs = {key: value.to(self.device, dtype=dtype if value.is_floating_point() else value.dtype)
                  for key, value in inputs.items()}
        if not bool(inputs["attention_mask"].bool().all()):
            raise ValueError("Phi v3 supports unpadded single examples only")
        with torch.no_grad():
            embeddings = self.model.model.vision_embed_tokens(
                inputs["input_ids"], pixel_values=inputs["pixel_values"], image_sizes=inputs["image_sizes"]).detach()
        visual, protected, groups = phi_image_layout(inputs["input_ids"], inputs["image_sizes"])
        semantic = phi_text_span_positions(self.processor.tokenizer, prompt, question_start,
                                          question_start + len(question), inputs["input_ids"])
        answer_positions = torch.empty(0, device=self.device, dtype=torch.long)
        ordinary = (inputs["input_ids"][0] >= 0).nonzero(as_tuple=False).flatten()
        all_nonvisual = torch.cat((ordinary, protected)).sort().values
        prompt_nonvisual = all_nonvisual
        if answer is not None:
            answer_positions = phi_text_span_positions(self.processor.tokenizer, prompt, answer_start,
                                                       len(prompt), inputs["input_ids"])
            if int(semantic.max()) >= int(answer_positions.min()):
                raise ValueError("Phi reference answer leaked into question-only scorer inputs")
            prompt_nonvisual = all_nonvisual[all_nonvisual < answer_positions.min()]
        if not len(visual) or len(visual) + len(all_nonvisual) != embeddings.shape[1]:
            raise ValueError("Phi visual/protected partition does not conserve prefix tokens")
        return PreparedBatch(embeddings, inputs["attention_mask"], visual, semantic,
                             prompt_nonvisual, answer_positions, inputs,
                             {"question": question, "answer": answer, "prompt": prompt,
                              "protected_visual_positions": protected,
                              "protected_visual_tokens": len(protected), "visual_crop_groups": groups,
                              "image_sizes": inputs["image_sizes"].cpu().tolist(),
                              "position_policy": "original", "num_crops": self.num_crops})

    def teacher_forward(self, prepared):
        with torch.no_grad():
            output = self.model(inputs_embeds=prepared.inputs_embeds,
                                attention_mask=prepared.attention_mask, use_cache=False,
                                return_dict=True)
        return {"hidden": output.last_hidden_state.detach(), "logits": output.logits.detach()}

    def student_forward_with_key_penalty(self, prepared, key_penalty):
        with visual_key_bias(self.model, key_penalty):
            output = self.model(inputs_embeds=prepared.inputs_embeds,
                                attention_mask=prepared.attention_mask, use_cache=False,
                                return_dict=True)
        return {"hidden": output.last_hidden_state, "logits": output.logits}

    def position_id_policy(self, prepared, retained_source_positions, policy):
        if policy != "original":
            raise ValueError("Phi v3 preserves original positions; compact positions are not allowed")
        return retained_source_positions.unsqueeze(0)

    def physically_prune(self, prepared, selected_visual_indices, position_policy="original"):
        selected = torch.as_tensor(selected_visual_indices, device=prepared.visual_positions.device)
        if (selected.dtype != torch.long or selected.ndim != 1 or not len(selected)
                or len(selected.unique()) != len(selected) or int(selected.min()) < 0
                or int(selected.max()) >= len(prepared.visual_positions)):
            raise ValueError("Phi selector must return unique in-range visual-local int64 indices")
        retained = torch.cat((prepared.visual_positions[selected], prepared.prompt_text_positions)).sort().values
        if len(retained.unique()) != len(retained):
            raise ValueError("Phi prompt/visual positions overlap")
        embeddings = prepared.inputs_embeds[:, retained]
        return PrunedBatch(embeddings, torch.ones(embeddings.shape[:2], device=embeddings.device,
                                                  dtype=prepared.attention_mask.dtype),
                           self.position_id_policy(prepared, retained, position_policy), retained,
                           {"position_policy": position_policy, "visual_tokens": len(selected),
                            "protected_visual_tokens": prepared.metadata["protected_visual_tokens"]})

    def forward_pruned_with_answer(self, pruned):
        output = self.model(inputs_embeds=pruned.inputs_embeds, attention_mask=pruned.attention_mask,
                            position_ids=pruned.position_ids, use_cache=False)
        return {"hidden": output.last_hidden_state, "logits": output.logits}

    def generate_from_pruned(self, pruned, generation_config):
        from learned_score.mlsys_runtime.generation import generate_bounded
        cap = int(generation_config["max_new_tokens"])
        return generate_bounded(self.model, pruned, max_new_tokens=cap, generated_budget=cap,
                                slack=1, require_fused_attention=False).ids


def phi_saliency(adapter, prepared, checkpoint=True):
    """Compute exactly ReLU(-d answer CE / d attention-key bias), then z-score."""
    from learned_score.multimodel.objectives import saliency_from_answer_ce_gradient
    if not len(prepared.answer_positions):
        raise ValueError("answer-gradient extraction requires answer positions")
    bias = torch.zeros(prepared.inputs_embeds.shape[1], device=adapter.device,
                       dtype=prepared.inputs_embeds.dtype, requires_grad=True)
    # Hooks must remain installed during checkpoint recomputation/backward.
    with checkpointed_decoder(adapter.model, checkpoint), visual_key_bias(adapter.model, bias):
        output = adapter.model(inputs_embeds=prepared.inputs_embeds,
                               attention_mask=prepared.attention_mask, use_cache=False)
        logits = output.logits[:, prepared.answer_positions - 1]
        targets = prepared.native_inputs["input_ids"][:, prepared.answer_positions]
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.flatten())
        gradient = torch.autograd.grad(loss, bias)[0]
    result = saliency_from_answer_ce_gradient(gradient[prepared.visual_positions])
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("non-finite Phi teacher saliency")
    return result


def _manifest(path):
    payload = json.loads(Path(path).read_text())
    if "phi_train" in payload.get("splits", {}):
        from .adaptive_data_v3 import load_manifest
        return load_manifest(path)
    from learned_score.mlsys_scorer.protocol import load_manifest
    return load_manifest(path)


def load_phi_backbone(args, manifest):
    """Factory with the existing (adapter, dataset, identity) return convention."""
    if "phi_train" in manifest.splits:
        from .adaptive_data_v3 import ManifestDatasetV3
        split = getattr(args, "split", None)
        phase = getattr(args, "phase", None)
        default = ("phi_train",) if phase == "extract" else (("fit",) if phase == "parity" else ("fit", "screening", "evaluation"))
        permitted = getattr(args, "dataset_splits", None) or ((split,) if split else default)
        dataset = ManifestDatasetV3(manifest.path, allowed_splits=permitted)
    else:
        from learned_score.mlsys_scorer.backbone import make_dataset
        dataset = make_dataset(manifest, getattr(args, "dataset_splits", None))
    adapter = PhiAdaptiveAdapterV3(dataset, args.device).load_model_and_processor(args)
    source_paths = [Path(__file__), Path(inspect.getfile(type(adapter.model.native))),
                    Path(inspect.getfile(type(adapter.processor))),
                    Path(inspect.getfile(type(adapter.model.native.config)))]
    source_paths += [Path(__file__).parents[1] / "mlsys_scorer" / "models.py",
                     Path(__file__).parents[1] / "multimodel" / "objectives.py"]
    identity = {"adapter": "phi3_vision", "adapter_version": adapter.version, "model": MODEL_ID,
                "model_path": str(adapter.snapshot_path), "requested_revision": getattr(args, "model_revision", None),
                "resolved_revision": adapter.snapshot_path.name,
                "resolution": {"num_crops": adapter.num_crops, "crop_edge": 336, "merged_patch_grid": [12, 12]},
                "backbone_dtype": adapter.dtype_name, "attention_implementation": "eager",
                "position_policy": "original", "question_only": True,
                "source_sha256": {path.name: digest_file(path) for path in source_paths}}
    return adapter, dataset, identity


def read_entry(dataset, entry):
    return dataset.row_for_entry(entry) if hasattr(dataset, "row_for_entry") else dataset.read_entry(entry)


def selected_entries(manifest, args):
    split = args.split or ("phi_train" if "phi_train" in manifest.splits else "scorer_train")
    if args.phase == "parity" and args.split is None:
        split = "fit" if "fit" in manifest.splits else "scorer_dev"
    if split not in manifest.splits:
        raise ValueError("requested Phi split is absent from manifest")
    entries = sorted(manifest.splits[split], key=lambda x: (x["dataset"], x["row_hash"]))
    if args.phase in {"extract", "train"} and split not in {"phi_train", "scorer_train"}:
        raise ValueError("Phi scorer training/extraction is restricted to official training rows")
    if args.phase == "parity" and split not in {"fit", "scorer_dev"}:
        raise ValueError("Phi parity never opens screening/evaluation labels")
    if args.request_ids:
        payload = json.loads(args.request_ids.read_text())
        if isinstance(payload, dict):
            payload = payload.get("row_hashes", payload.get("rows", payload.get("request_ids")))
        if not isinstance(payload, list):
            raise ValueError("request IDs must be a list or row_hashes/rows/request_ids list")
        requested = [entry["row_hash"] if isinstance(entry, dict) else entry for entry in payload]
        if len(set(requested)) != len(requested) or set(requested) - {e["row_hash"] for e in entries}:
            raise ValueError("duplicate or out-of-split Phi request IDs")
        entries = [entry for entry in entries if entry["row_hash"] in set(requested)]
    if args.phase in {"extract", "train"}:
        if (len(entries) != 512 or any(e.get("source_split") != "train" for e in entries)
                or {d: sum(e["dataset"] == d for e in entries) for d in TRAIN_DATASETS}
                != {d: 256 for d in TRAIN_DATASETS}):
            raise ValueError("new Phi scorer requires exactly 256 official training rows per dataset")
    elif args.limit:
        entries = entries[:args.limit]
    if not entries:
        raise ValueError("empty Phi request selection")
    return split, entries


def parity(adapter, dataset, entries, args, identity):
    """Native image-input vs prepared full-token logits and bounded-cache IDs."""
    from transformers.cache_utils import DynamicCache
    from learned_score.mlsys_runtime.generation import generate_bounded
    from learned_score.mlsys_scorer.backbone import frozen_versions, assert_frozen
    class NativeCache(DynamicCache):
        def get_usable_length(self, new_seq_length, layer_idx=0):
            return self.get_seq_length(layer_idx)
        def get_max_length(self):
            return None
    versions = frozen_versions(adapter.model)
    for entry in entries[args.rank::args.world_size]:
        target = args.output / (entry["row_hash"] + ".json")
        prepared = adapter.prepare_sample(read_entry(dataset, entry))
        selected = torch.arange(len(prepared.visual_positions), device=adapter.device)
        pruned = adapter.physically_prune(prepared, selected)
        if not torch.equal(pruned.inputs_embeds, prepared.inputs_embeds):
            raise RuntimeError("Phi full-token physical pruning changed the prompt")
        with torch.no_grad():
            reference = adapter.model.native(**prepared.native_inputs, use_cache=False, return_dict=True)
            actual = adapter.model(inputs_embeds=pruned.inputs_embeds, attention_mask=pruned.attention_mask,
                                   position_ids=pruned.position_ids, use_cache=False)
            error = float((reference.logits[:, -1].float() - actual.logits[:, -1]).abs().max())
            if not torch.allclose(reference.logits[:, -1].float(), actual.logits[:, -1], rtol=2e-3, atol=2e-3):
                raise RuntimeError(f"Phi native/prepared logits parity failed: max error {error}")
            del reference, actual
            cache = NativeCache()
            output = adapter.model.native(**prepared.native_inputs, past_key_values=cache,
                                          use_cache=True, return_dict=True)
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            ids = [token]
            for offset in range(args.max_new_tokens - 1):
                length = cache.get_seq_length()
                output = adapter.model.native(input_ids=token,
                    attention_mask=torch.ones((1, length + 1), device=adapter.device, dtype=torch.long),
                    position_ids=torch.tensor([[prepared.inputs_embeds.shape[1] + offset]], device=adapter.device),
                    past_key_values=cache, use_cache=True, return_dict=True)
                token = output.logits[:, -1].argmax(-1, keepdim=True)
                ids.append(token)
            reference_ids = torch.cat(ids, 1)
            del output, cache
            bounded = generate_bounded(adapter.model, pruned, args.max_new_tokens, 512, slack=1,
                                       stop_on_eos=False, require_fused_attention=False)
            if not torch.equal(reference_ids, bounded.ids):
                raise RuntimeError("Phi native and protected-prefix generated IDs differ")
            cache_metrics = bounded.cache.memory_stats()
            if cache_metrics["compaction_events"] or cache_metrics["evicted_tokens"]:
                raise RuntimeError("Phi parity unexpectedly evicted tokens")
            # A half-retention smoke test checks original-position gaps/cache plumbing.
            half = adapter.physically_prune(prepared, selected[::2])
            smoke = generate_bounded(adapter.model, half, 2, 512, slack=1,
                                     stop_on_eos=False, require_fused_attention=False)
        assert_frozen(adapter.model, versions)
        atomic_json(target, {"format": "mlsys_phi_parity_v3", "status": "passed", "row_hash": entry["row_hash"],
                             "backbone": identity, "manifest_sha256": digest_file(args.manifest),
                             "input_visual_tokens": len(selected), "protected_visual_tokens": prepared.metadata["protected_visual_tokens"],
                             "input_prefix_tokens": prepared.inputs_embeds.shape[1], "full_logit_max_error": error,
                             "generated_ids": reference_ids.cpu().tolist(), "bounded_cache": cache_metrics,
                             "pruned_smoke_ids": smoke.ids.cpu().tolist(), "original_positions_preserved": True})
        print(json.dumps({"event": "phi_parity_passed", "row_hash": entry["row_hash"], "visual_tokens": len(selected)}), flush=True)
        del bounded, smoke, prepared, pruned, half
    atomic_json(args.output / f"rank{args.rank}.complete.json", {"status": "passed", "backbone": identity,
                "rows": len(entries[args.rank::args.world_size]), "manifest_sha256": digest_file(args.manifest)})


def extract(adapter, dataset, entries, args, identity):
    from learned_score.mlsys_scorer.backbone import frozen_versions, assert_frozen
    versions = frozen_versions(adapter.model)
    manifest_hash = digest_file(args.manifest)
    for entry in entries[args.rank::args.world_size]:
        target = args.output / (entry["row_hash"] + ".pt")
        if target.exists():
            previous = torch.load(target, map_location="cpu", weights_only=True)
            if (previous.get("backbone") != identity or previous.get("manifest_sha256") != manifest_hash
                    or previous.get("row_hash") != entry["row_hash"]):
                raise ValueError("existing Phi saliency provenance differs")
            continue
        prepared = adapter.prepare_sample(read_entry(dataset, entry), include_answer=True)
        saliency = phi_saliency(adapter, prepared)
        atomic_torch_save(target, {"format": "asymvlm_mlsys_saliency_v1", "backbone": identity,
            "manifest_sha256": manifest_hash, "row_hash": entry["row_hash"], "dataset": entry["dataset"],
            "image_sha256": entry["image_sha256"],
            "vision": prepared.inputs_embeds[0, prepared.visual_positions].cpu().to(torch.bfloat16),
            "question": prepared.inputs_embeds[0, prepared.semantic_text_positions].cpu().to(torch.bfloat16),
            "saliency": saliency.detach().float().cpu()})
        assert_frozen(adapter.model, versions)
        print(json.dumps({"event": "phi_extracted", "row_hash": entry["row_hash"],
                          "visual_tokens": len(saliency), "saliency_std": float(saliency.std(unbiased=False))}), flush=True)
        del prepared, saliency
    atomic_json(args.output / f"rank{args.rank}.complete.json", {"backbone": identity,
                "rows": len(entries[args.rank::args.world_size]), "manifest_sha256": manifest_hash})


def train(entries, args):
    """Exactly one new 512-example rank32 pass, seed 42; resumable checkpoints."""
    from learned_score.mlsys_scorer.models import Rank32Scorer, saliency_loss, optimizer_step
    if args.rank or args.world_size != 1 or args.cache is None:
        raise ValueError("Phi train uses one worker and an explicit completed saliency cache")
    manifest_hash = digest_file(args.manifest)
    first = torch.load(args.cache / (entries[0]["row_hash"] + ".pt"), map_location="cpu", weights_only=True)
    backbone = first["backbone"]
    if backbone.get("adapter_version") != PhiAdaptiveAdapterV3.version:
        raise ValueError("Phi v3 refuses old scorer/saliency artifacts")
    cache_hash = canonical_digest([(e["row_hash"], digest_file(args.cache / (e["row_hash"] + ".pt"))) for e in entries])
    identity = {"variant": "rank32", "manifest_sha256": manifest_hash, "backbone": backbone,
                "seed": 42, "epochs": 1, "learning_rate": .001, "weight_decay": .01,
                "retentions": [.3, .4, .5, .6, .7, .8, .9, 1.], "rows": 512,
                "cache_sha256": cache_hash, "limit": 512, "objective": "answer_gradient_saliency",
                "coverage_fraction": 0., "new_training_ids_sha256": canonical_digest([e["row_hash"] for e in entries])}
    torch.manual_seed(42)
    scorer = Rank32Scorer(first["vision"].shape[1]).to(args.device).float()
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=.001, weight_decay=.01)
    latest = args.output / "latest.pt"
    offset = steps = 0
    if latest.exists():
        if not args.resume:
            raise FileExistsError("Phi scorer already exists; --resume requires identical provenance")
        saved = load_checkpoint(latest, identity, args.device)
        scorer.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        offset, steps = saved["next_index"], saved["steps"]
        if saved["epoch"] == 1:
            atomic_json(args.output / "complete.json", {"steps": steps, "identity_sha256": canonical_digest(identity)})
            return
    elif args.resume:
        raise FileNotFoundError("Phi --resume requires an existing checkpoint")
    else:
        save_checkpoint(args.output / "epoch_zero.pt", scorer, optimizer, identity, 0, 0, 0)
    atomic_json(args.output / "run_metadata.json", identity)
    ordered = list(entries)
    random.Random(42).shuffle(ordered)
    for index in range(offset, len(ordered)):
        entry = ordered[index]
        cached = torch.load(args.cache / (entry["row_hash"] + ".pt"), map_location=args.device, weights_only=True)
        if (cached.get("format") != "asymvlm_mlsys_saliency_v1" or cached.get("manifest_sha256") != manifest_hash
                or cached.get("backbone") != backbone or cached.get("row_hash") != entry["row_hash"]
                or cached.get("image_sha256") != entry["image_sha256"]):
            raise ValueError("Phi cache training identity mismatch")
        optimizer.zero_grad(set_to_none=True)
        loss = saliency_loss(scorer(cached["vision"], cached["question"]), cached["saliency"])
        optimizer_step(loss, scorer, optimizer)
        steps += 1
        if steps % 32 == 0:
            save_checkpoint(latest, scorer, optimizer, identity, 0, index + 1, steps)
            print(json.dumps({"event": "phi_training", "steps": steps, "loss": float(loss.detach())}), flush=True)
    save_checkpoint(latest, scorer, optimizer, identity, 1, 0, steps)
    atomic_json(args.output / "complete.json", {"steps": steps, "identity_sha256": canonical_digest(identity)})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("parity", "extract", "train"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--request-ids", type=Path)
    parser.add_argument("--split")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default="12b77fb40b63a2c73c68243d3f767aab688a1b2a")
    parser.add_argument("--backbone-dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--num-crops", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.num_crops != 16 or args.world_size < 1 or not 0 <= args.rank < args.world_size or args.max_new_tokens < 1:
        parser.error("Phi v3 requires 16 crops, valid rank/world-size, and positive generation cap")
    return args


def main(argv=None):
    args = parse_args(argv)
    manifest = _manifest(args.manifest)
    _, entries = selected_entries(manifest, args)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "train":
        train(entries, args)
    else:
        adapter, dataset, identity = load_phi_backbone(args, manifest)
        (parity if args.phase == "parity" else extract)(adapter, dataset, entries, args, identity)


if __name__ == "__main__":
    main()
