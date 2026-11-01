"""Offline evidence annotation with native Qwen geometry and answer-free caches.

The teacher sees a verified answer only to locate supporting image regions.
The learned scorer's cached inputs always come from the original question,
never from that annotation prompt or an answer-bearing prompt. Image-region
targets use union area, so overlapping boxes cannot double-count evidence.
"""

import argparse
import json
import math
from pathlib import Path
import time

from .followup_common import write_immutable
from .state import digest, file_digest

FORMAT = "mlsys_evidence_annotation_v18"
COORDINATE_SCALE = 1000.0
PROMPT_VERSION = "evidence_regions_normalized_v1"


def parse_annotation(raw):
    """Strict JSON/schema validation; absent evidence is not a negative label."""
    def invalid(reason):
        return {"status": "invalid", "boxes": [], "evidence": "", "reason": reason}

    if isinstance(raw, str):
        value = raw.strip()
        if value.startswith("```"):
            lines = value.splitlines()
            if len(lines) < 3 or lines[-1].strip() != "```":
                return invalid("unterminated_json_fence")
            value = "\n".join(lines[1:-1]).strip()
        try:
            raw = json.loads(value)
        except (ValueError, TypeError):
            return invalid("invalid_json")
    if not isinstance(raw, dict):
        return invalid("expected_object")
    if raw.get("coordinate_system", "normalized_0_1000") != "normalized_0_1000":
        return invalid("unexpected_coordinate_system")
    status, boxes = raw.get("status"), raw.get("boxes")
    evidence = raw.get("evidence", "")
    if not isinstance(evidence, str):
        return invalid("evidence_must_be_text")
    if status == "cannot_locate":
        if boxes not in (None, []):
            return invalid("cannot_locate_with_boxes")
        return {"status": status, "boxes": [], "evidence": evidence,
                "reason": "teacher_cannot_locate"}
    if status != "ok" or not isinstance(boxes, list) or not 1 <= len(boxes) <= 16:
        return invalid("expected_ok_and_one_to_sixteen_boxes")
    normalized = []
    for box in boxes:
        if (not isinstance(box, (list, tuple)) or len(box) != 4 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
            return invalid("invalid_box_values")
        x0, y0, x1, y1 = map(float, box)
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            return invalid("box_out_of_bounds_or_empty")
        normalized.append([x0, y0, x1, y1])
    return {"status": "ok", "boxes": normalized, "evidence": evidence, "reason": ""}


def qwen_patch_rectangles(prepared, spatial_merge_size=2):
    """Native merged tokens mapped to original-image normalized rectangles.

    Qwen's single-image processor resizes the entire image without cropping or
    padding. Thus original and resized image coordinates have equal normalized
    coordinates. Validate actual mRoPE token ordering, never guess a square grid.
    """
    import torch
    from learned_score.mlsys_runtime.spatial import qwen_patch_coordinates
    coordinates, (height, width) = qwen_patch_coordinates(prepared, spatial_merge_size)
    y, x = coordinates[:, 0].double(), coordinates[:, 1].double()
    return torch.stack((x / width, y / height, (x + 1) / width,
                        (y + 1) / height), dim=1).mul(COORDINATE_SCALE)


def _disjoint_box_union(boxes):
    """Decompose rectangular union into nonoverlapping vertical strips."""
    edges = sorted({box[i] for box in boxes for i in (0, 2)})
    result = []
    for x0, x1 in zip(edges, edges[1:]):
        intervals = sorted((b[1], b[3]) for b in boxes if b[0] < x1 and b[2] > x0)
        merged = []
        for y0, y1 in intervals:
            if merged and y0 <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], y1)
            else:
                merged.append([y0, y1])
        result.extend([x0, y0, x1, y1] for y0, y1 in merged)
    return result


def evidence_overlap(boxes, rectangles):
    """Return union-overlap area per token, not center-inside labels."""
    import torch
    parsed = parse_annotation({"status": "ok", "boxes": boxes})
    if parsed["status"] != "ok":
        raise ValueError("valid nonempty evidence boxes required: " + parsed["reason"])
    rectangles = torch.as_tensor(rectangles, dtype=torch.float64)
    if (rectangles.ndim != 2 or rectangles.shape[1] != 4 or len(rectangles) == 0 or
            not bool(torch.isfinite(rectangles).all()) or bool((rectangles < 0).any()) or
            bool((rectangles > 1000).any()) or
            bool((rectangles[:, :2] >= rectangles[:, 2:]).any())):
        raise ValueError("finite nonempty normalized token rectangles required")
    area = rectangles.new_zeros(len(rectangles))
    for box in _disjoint_box_union(parsed["boxes"]):
        b = rectangles.new_tensor(box)
        size = (torch.minimum(rectangles[:, 2:], b[2:]) -
                torch.maximum(rectangles[:, :2], b[:2])).clamp_min(0)
        area += size.prod(dim=1)
    if float(area.sum()) <= 0:
        raise ValueError("no evidence intersects the image tokens")
    return area


def evidence_distribution(boxes, rectangles, smoothing=.05):
    if type(smoothing) not in (int, float) or not math.isfinite(smoothing) or not 0 <= smoothing < 1:
        raise ValueError("smoothing must be finite and in [0,1)")
    area = evidence_overlap(boxes, rectangles)
    return ((1 - smoothing) * area / area.sum() + smoothing / len(area)).float()


def annotation_prompt(question, answers):
    if not isinstance(question, str) or not question.strip() or not answers:
        raise ValueError("training question and verified answers are required")
    # JSON quoting keeps source question/answer strings visibly data, not new
    # template instructions. They are supervision-only, never scorer inputs.
    return (
        "Locate the visual evidence needed to support the verified answer to the question. "
        "Mark tight rectangles around the relevant text or objects, including all regions "
        "needed for comparisons. Use coordinates of the ORIGINAL WHOLE IMAGE, normalized "
        "from 0 to 1000 on each axis: [left, top, right, bottom]. Do not use pixel indices. "
        "Do not mark the whole image unless the question genuinely needs all of it. "
        "If you cannot locate the evidence, explicitly say cannot_locate. "
        "Return only JSON, no prose or markdown, with this schema: "
        '{"status":"ok","coordinate_system":"normalized_0_1000",'
        '"boxes":[[100,200,300,400]],"evidence":"short description"}. '
        'For failure return {"status":"cannot_locate","boxes":[],"evidence":"reason"}. '
        "Use at most 16 boxes.\nQuestion: " + json.dumps(question, ensure_ascii=False) +
        "\nVerified answer(s): " + json.dumps(list(answers), ensure_ascii=False)
    )


def _image(value):
    from PIL import Image
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if hasattr(value, "seek"):
        value.seek(0)
    return Image.open(value).convert("RGB")


def annotate_qwen(adapter, image, prompt, max_new_tokens=384):
    """Use full image tokens for offline annotation; no pruning hook applies."""
    import torch
    messages = [{"role": "user", "content": [{"type": "image"},
                 {"type": "text", "text": prompt}]}]
    text = adapter.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = adapter.processor(text=[text], images=[_image(image)], return_tensors="pt")
    inputs = {k: v.to(adapter.device) for k, v in inputs.items()}
    with torch.no_grad():
        output = adapter.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                        do_sample=False, use_cache=True)
    return adapter.processor.tokenizer.decode(
        output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def internvl_pixels(image, tile_size=448, max_tiles=12):
    """Deterministic aspect-ratio tiling + thumbnail, matching official recipe.

    Reference: https://huggingface.co/OpenGVLab/InternVL3_5-4B . Independent
    implementation; ImageNet normalization and 448px tiles are model inputs.
    """
    import numpy as np
    import torch
    from PIL import Image
    image = _image(image)
    width, height = image.size
    candidates = sorted(((w, h) for w in range(1, max_tiles + 1)
                         for h in range(1, max_tiles + 1) if w * h <= max_tiles),
                        key=lambda x: (x[0] * x[1], x[0], x[1]))
    best, distance = (1, 1), float("inf")
    for grid in candidates:
        d = abs(width / height - grid[0] / grid[1])
        if d < distance or (d == distance and width * height > .5 * tile_size**2 * grid[0] * grid[1]):
            best, distance = grid, d
    wide, tall = best
    enlarged = image.resize((wide * tile_size, tall * tile_size))
    tiles = [enlarged.crop((x * tile_size, y * tile_size, (x + 1) * tile_size, (y + 1) * tile_size))
             for y in range(tall) for x in range(wide)]
    if len(tiles) > 1:
        tiles.append(image.resize((tile_size, tile_size)))
    mean = torch.tensor([.485, .456, .406])[:, None, None]
    std = torch.tensor([.229, .224, .225])[:, None, None]
    return torch.stack([(torch.from_numpy(np.asarray(t.resize((tile_size, tile_size),
                       Image.Resampling.BICUBIC)).copy()).permute(2, 0, 1).float() / 255 - mean) / std
                       for t in tiles])


def load_internvl(spec, device="cuda:0"):
    import torch
    from transformers import AutoModel, AutoTokenizer
    path = spec["model_path"]
    revision = spec.get("revision")
    # Ops must stage a pinned snapshot. Annotation workers never download model
    # weights or executable remote code opportunistically on compute nodes.
    model = AutoModel.from_pretrained(path, revision=revision, trust_remote_code=True,
        local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=False).to(device).eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision, trust_remote_code=True,
        local_files_only=True, use_fast=False)
    return model, tokenizer


def annotate_internvl(model, tokenizer, image, prompt, max_new_tokens=384):
    import torch
    pixels = internvl_pixels(image).to(device=next(model.parameters()).device, dtype=torch.bfloat16)
    with torch.no_grad():
        return model.chat(tokenizer, pixels, "<image>\n" + prompt,
                          {"max_new_tokens": max_new_tokens, "do_sample": False})


def extract_cache(adapter, row, entry, annotation, backbone, model_spec,
                  annotation_sha256, compute_saliency=True):
    """Separate answer-free inputs from answer-aware target construction."""
    import torch
    from learned_score.mlsys_scorer.backbone import prepared_inputs
    from learned_score.multimodel.saliency import compute_answer_gradient_saliency
    if hasattr(row.get("image"), "seek"):
        row["image"].seek(0)
    with torch.no_grad():
        prepared, visual, question = prepared_inputs(adapter, row, False)
        if prepared.answer_positions.numel():
            raise ValueError("answer-free scorer extraction contains an answer span")
        rectangles = qwen_patch_rectangles(prepared)
        visual_cpu, question_cpu = visual.cpu().to(torch.bfloat16), question.cpu().to(torch.bfloat16)
        prompt_count = int(prepared.inputs_embeds.shape[1] - len(visual))
    saliency = None
    if compute_saliency:
        if hasattr(row.get("image"), "seek"):
            row["image"].seek(0)
        supervised, answer_visual, answer_question = prepared_inputs(adapter, row, True)
        if (not torch.equal(visual, answer_visual) or not torch.equal(question, answer_question)):
            raise ValueError("answer-aware preparation changed scorer inputs")
        saliency = compute_answer_gradient_saliency(adapter, supervised).detach().float().cpu()
        if saliency.shape != (len(visual),) or not bool(torch.isfinite(saliency).all()):
            raise ValueError("invalid frozen target-model gradient saliency")
        del supervised, answer_visual, answer_question
    targets = (evidence_distribution(annotation["boxes"], rectangles).cpu()
               if annotation["status"] == "ok" else None)
    result = {"format": "mlsys_evidence_cache_v18", "row_hash": entry["row_hash"],
        "dataset": entry["dataset"], "image_sha256": entry["image_sha256"],
        "group_id": entry["group_id"], "backbone": backbone, "model": model_spec,
        "vision": visual_cpu, "question": question_cpu, "saliency": saliency,
        "patch_rectangles": rectangles.cpu(), "evidence_targets": targets,
        "evidence_smoothing": .05, "annotation_status": annotation["status"],
        "annotation_sha256": annotation_sha256, "input_visual_tokens": len(visual),
        "input_text_tokens": prompt_count, "question_only": True,
        "answers_in_scorer_inputs": False, "answer_free_embeddings": True,
        "features_inputs_include_answer": False, "source_split": entry.get("source_split"),
        "target_model_checkpoint_sha256": file_digest(model_spec["checkpoint"]),
        "coordinate_system": "normalized_0_1000",
        "geometry": "verified_original_qwen_mrope_row_major_whole_image_resize"}
    return result


def _runtime(task):
    """Use the target's established adapter and checkpoint identity checks."""
    from . import compression_execution_v16 as execution
    from .evidence_learning_v18 import load_pilot_runtime
    from .adaptive_data_v3 import load_manifest
    args = execution.runtime_args(task)
    manifest = load_manifest(task["manifest"])
    return load_pilot_runtime(args, manifest)


def _entries_and_dataset(task):
    from .adaptive_data_v3 import load_manifest
    from .evidence_learning_v18 import EvidenceDataset
    manifest = load_manifest(task["manifest"])
    all_entries = manifest.splits[task["split"]]
    requested = task.get("entries")
    if requested is None:
        entries = list(all_entries)
    else:
        by_id = {entry["row_hash"]: entry for entry in all_entries}
        ids = [value["row_hash"] if isinstance(value, dict) else value for value in requested]
        if len(ids) != len(set(ids)) or any(key not in by_id for key in ids):
            raise ValueError("task entries must be a distinct subset of its permitted split")
        entries = [by_id[key] for key in ids]
    return entries, EvidenceDataset(task["manifest"], allowed_splits=(task["split"],))


def render_overlay(image, annotation, output_path, *, question="", answers=()):
    """Render original-image evidence boxes for manual review, never training."""
    import textwrap
    from PIL import Image, ImageDraw, ImageFont
    original = _image(image)
    original.thumbnail((1400, 1800))
    width, height = original.size
    text_lines = []
    for text in ("Question: " + question, "Verified answer: " + ", ".join(map(str, answers)),
                 "Teacher evidence: " + annotation.get("evidence", ""),
                 "Status: " + annotation["status"]):
        text_lines.extend(textwrap.wrap(text, width=max(35, width // 9)))
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 16) if Path(font_path).is_file() else ImageFont.load_default()
    caption_height = 22 * len(text_lines) + 12
    canvas = Image.new("RGB", (max(width, 360), height + caption_height), "white")
    canvas.paste(original, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, (x0, y0, x1, y1) in enumerate(annotation.get("boxes", [])):
        rect = (x0 * width / 1000, y0 * height / 1000, x1 * width / 1000, y1 * height / 1000)
        draw.rectangle(rect, outline="red", width=max(2, width // 300))
        draw.text((rect[0] + 2, rect[1] + 2), str(index + 1), fill="red", font=font,
                  stroke_width=1, stroke_fill="white")
    for index, line in enumerate(text_lines):
        draw.text((5, height + 6 + 22 * index), line, fill="black", font=font)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.png")
    canvas.save(temporary, format="PNG")
    temporary.replace(path)
    return {"path": str(path.resolve()), "sha256": file_digest(path),
            "original_image_size": list(_image(image).size), "rendered_image_size": [width, height]}


def run_render_overlays(task_path, limit=20):
    from benchmark_dual_kv_v2 import parse_answer_field
    task = json.loads(Path(task_path).read_text())
    entries, dataset = _entries_and_dataset(task)
    teacher = task.get("teacher", "qwen")
    root = Path(task["output_root"]).resolve() / "annotation" / teacher
    approved = Path(task["cwd"]).resolve() / "reproduction_results/mlsys_evidence_20261025_v18"
    if not root.is_relative_to(approved):
        raise ValueError("overlay destination must remain in the new v18 campaign")
    requested = task.get("overlay_row_hashes")
    if requested is not None:
        by_id = {e["row_hash"]: e for e in entries}
        if any(h not in by_id for h in requested) or len(set(requested)) != len(requested):
            raise ValueError("overlay review rows must belong to the permitted task")
        entries = [by_id[h] for h in requested]
    else:
        entries = sorted(entries, key=lambda e: e["row_hash"])[:limit]
    records = []
    for entry in entries:
        source = root / "rows" / (entry["row_hash"] + ".json")
        if not source.exists():
            continue
        annotation = json.loads(source.read_text())["annotation"]
        row = dataset.row_for_entry(entry)
        overlay = render_overlay(row["image"], annotation,
            root / "overlays" / (entry["row_hash"] + ".png"), question=row["question"],
            answers=parse_answer_field(row.get("answer", "")))
        records.append({"row_hash": entry["row_hash"], "dataset": entry["dataset"],
                        "annotation_path": str(source), "annotation_sha256": file_digest(source),
                        "annotation_status": annotation["status"], **overlay})
    result = {"format": "mlsys_evidence_overlay_index_v18", "teacher": teacher,
              "entries": records, "requested": len(entries), "rendered": len(records),
              "purpose": "manual_training_annotation_review_only"}
    from .state import atomic_json
    atomic_json(root / "overlay-index.json", result)
    return result


def run(task_path, rank=0, world_size=1, deadline_utc_epoch=None):
    import torch
    from benchmark_dual_kv_v2 import parse_answer_field
    from learned_score.mlsys_scorer.backbone import frozen_versions, assert_frozen
    from learned_score.mlsys_scorer.checkpoints import atomic_torch_save
    task_path = Path(task_path).resolve()
    task = json.loads(task_path.read_text())
    task_hash = file_digest(task_path)
    deadline = deadline_utc_epoch or task["deadline_utc_epoch"]
    if (not 0 <= rank < world_size or not math.isfinite(deadline) or time.time() >= deadline):
        raise ValueError("invalid worker rank or expired annotation deadline")
    for path, expected in task.get("input_hashes", {}).items():
        if file_digest(path) != expected:
            raise ValueError("annotation input hash changed: " + path)
    mode, teacher = task.get("mode", "annotate_extract"), task.get("teacher", "qwen")
    if mode not in ("annotate", "extract", "annotate_extract") or teacher not in ("qwen", "internvl"):
        raise ValueError("unsupported annotation mode/teacher")
    # Explicit new-campaign destination; never overwrite historical caches.
    root = Path(task["output_root"]).resolve()
    approved = Path(task["cwd"]).resolve() / "reproduction_results/mlsys_evidence_20261025_v18"
    if root != approved and not root.is_relative_to(approved):
        raise ValueError("v18 annotation output must remain in its campaign directory")
    destination = root / "annotation" / teacher
    entries, dataset = _entries_and_dataset(task)
    entries = entries[rank::world_size]
    adapter = scorer = backbone = versions = scorer_versions = external = external_versions = None
    if mode != "annotate" or teacher == "qwen":
        adapter, _, backbone, scorer = _runtime(task)
        adapter.dataset = dataset
        versions, scorer_versions = frozen_versions(adapter.model), frozen_versions(scorer)
    if teacher == "internvl" and mode != "extract":
        external = load_internvl(task["teacher_model"])
        external_versions = frozen_versions(external[0])
    records, begun = [], time.monotonic()
    for index, entry in enumerate(entries):
        if time.time() >= deadline:
            raise TimeoutError("annotation deadline reached; atomic rows preserved")
        row_hash = entry["row_hash"]
        if len(row_hash) != 64 or any(c not in "0123456789abcdef" for c in row_hash):
            raise ValueError("invalid row identifier")
        annotation_path = destination / "rows" / (row_hash + ".json")
        cache_path = destination / "cache" / (row_hash + ".pt")
        row = None
        if annotation_path.exists():
            record = json.loads(annotation_path.read_text())
            if (record.get("row_hash") != row_hash or record.get("image_sha256") != entry["image_sha256"]
                    or record.get("teacher") != teacher or record.get("prompt_version") != PROMPT_VERSION
                    or record.get("teacher_model") != (task["model"] if teacher == "qwen" else task["teacher_model"])):
                raise ValueError("resumed annotation identity changed")
        else:
            if mode == "extract":
                raise FileNotFoundError("evidence extraction requires existing teacher annotation: " + str(annotation_path))
            row = dataset.row_for_entry(entry)
            references = parse_answer_field(row.get("answer", ""))
            prompt = annotation_prompt(row["question"], references)
            started = time.monotonic()
            raw = (annotate_qwen(adapter, row["image"], prompt, task.get("max_new_tokens", 384))
                   if teacher == "qwen" else annotate_internvl(*external, row["image"], prompt,
                                                              task.get("max_new_tokens", 384)))
            parsed = parse_annotation(raw)
            record = {"format": FORMAT, "row_hash": row_hash, "dataset": entry["dataset"],
                "group_id": entry["group_id"], "image_sha256": entry["image_sha256"],
                "teacher": teacher, "teacher_model": task["model"] if teacher == "qwen" else task["teacher_model"],
                "prompt_version": PROMPT_VERSION, "prompt_sha256": digest(prompt),
                "verified_answers_sha256": digest(references), "raw_response": raw,
                "annotation": parsed, "annotation_sha256": digest(parsed),
                "task_path": str(task_path), "task_sha256": task_hash,
                "elapsed_seconds": time.monotonic() - started, "training_only": True}
            write_immutable(annotation_path, record)
        annotation_file_sha = file_digest(annotation_path)
        item = {"row_hash": row_hash, "dataset": entry["dataset"],
                "annotation_path": str(annotation_path), "annotation_sha256": annotation_file_sha,
                "annotation_status": record["annotation"]["status"], "evidence_smoothing": .05}
        if mode != "annotate":
            if cache_path.exists():
                cache = torch.load(cache_path, map_location="cpu", weights_only=True)
                if (cache.get("row_hash") != row_hash or cache.get("annotation_sha256") != annotation_file_sha
                        or cache.get("backbone") != backbone or cache.get("answers_in_scorer_inputs") is not False):
                    raise ValueError("resumed evidence cache identity changed")
            else:
                row = row if row is not None else dataset.row_for_entry(entry)
                cache = extract_cache(adapter, row, entry, record["annotation"], backbone,
                                      task["model"], annotation_file_sha, task.get("compute_saliency", True))
                cache["manifest_split"] = task["split"]
                cache["manifest_sha256"] = file_digest(task["manifest"])
                atomic_torch_save(cache_path, cache)
            item.update(path=str(cache_path), sha256=file_digest(cache_path))
            del cache
        records.append(item)
        if adapter is not None:
            assert_frozen(adapter.model, versions)
            assert_frozen(scorer, scorer_versions)
        if external is not None:
            assert_frozen(external[0], external_versions)
        print(json.dumps({"event": "evidence_annotation_progress", "teacher": teacher,
              "rank": rank, "completed": index + 1, "total": len(entries),
              "annotation_status": item["annotation_status"],
              "elapsed_seconds": time.monotonic() - begun}), flush=True)
    result = {"format": "mlsys_evidence_cache_index_v18", "task_sha256": task_hash,
              "teacher": teacher, "mode": mode, "rank": rank, "world_size": world_size,
              "entries": records, "completed": True, "backbone": backbone,
              "elapsed_seconds": time.monotonic() - begun}
    # Separate annotate-only/extract index names permit staged annotation review.
    index_name = f"index-rank{rank}.json" if mode != "annotate" else f"annotations-rank{rank}.json"
    index_path = destination / index_name
    if index_path.exists():
        previous = json.loads(index_path.read_text())
        if any(previous.get(k) != result[k] for k in ("task_sha256", "entries", "teacher", "mode")):
            raise ValueError("completed evidence shard differs")
        return previous
    write_immutable(index_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--deadline-utc-epoch", type=float)
    parser.add_argument("--render-overlays", action="store_true")
    parser.add_argument("--overlay-limit", type=int, default=20)
    args = parser.parse_args(argv)
    if args.render_overlays:
        print(json.dumps(run_render_overlays(args.task, args.overlay_limit), indent=2))
    else:
        run(args.task, args.rank, args.world_size, args.deadline_utc_epoch)


if __name__ == "__main__":
    main()
