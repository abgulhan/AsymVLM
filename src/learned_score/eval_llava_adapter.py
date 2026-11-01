"""LLaVA-1.5 pre-LLM vision-pruning correctness gate."""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
)

from benchmark_dual_kv_v2 import (
    check_answer,
    get_eval_metric,
    parse_answer_field,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", default="llava-hf/llava-1.5-7b-hf"
    )
    parser.add_argument("--dataset", default="MME")
    parser.add_argument("--eval-start", type=int, required=True)
    parser.add_argument("--eval-samples", type=int, required=True)
    parser.add_argument("--keep-ratio", type=float, default=0.75)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def split_message(message):
    image_paths = [
        item["value"] for item in message if item["type"] == "image"
    ]
    text_parts = [
        item["value"] for item in message if item["type"] == "text"
    ]
    if len(image_paths) != 1:
        raise ValueError(
            f"LLaVA gate requires one image, found {len(image_paths)}"
        )
    return image_paths[0], "\n".join(text_parts)


def greedy_generate_from_embeds(
    model, inputs_embeds, attention_mask, max_new_tokens
):
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = {eos_ids}
    else:
        eos_ids = set(eos_ids or [])
    generated = []
    with torch.no_grad():
        output = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
    cache = output.past_key_values
    next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated.append(next_token)

    for _ in range(max_new_tokens - 1):
        if next_token.item() in eos_ids:
            break
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], 1),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )
        with torch.no_grad():
            output = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
        cache = output.past_key_values
        next_token = output.logits[:, -1, :].argmax(
            dim=-1, keepdim=True
        )
        generated.append(next_token)
    return torch.cat(generated, dim=1)


def build_multimodal_embeds(model, inputs):
    input_ids = inputs["input_ids"]
    token_embeds = model.get_input_embeddings()(input_ids)
    image_features = model.get_image_features(
        pixel_values=inputs["pixel_values"],
        vision_feature_layer=model.config.vision_feature_layer,
        vision_feature_select_strategy=(
            model.config.vision_feature_select_strategy
        ),
        image_sizes=inputs.get("image_sizes"),
    )
    image_mask = input_ids.eq(model.config.image_token_index)
    if image_mask.sum().item() != image_features.shape[1]:
        raise RuntimeError(
            "expanded image-token count does not match projected features: "
            f"{image_mask.sum().item()} vs {image_features.shape[1]}"
        )
    expanded_mask = image_mask.unsqueeze(-1).expand_as(token_embeds)
    merged = token_embeds.masked_scatter(
        expanded_mask, image_features.to(token_embeds.dtype)
    )
    return merged, image_mask


def prune_image_tokens(inputs_embeds, attention_mask, image_mask, keep_ratio):
    image_indices = image_mask[0].nonzero(as_tuple=False).squeeze(1)
    text_indices = (~image_mask[0] & attention_mask[0].bool()).nonzero(
        as_tuple=False
    ).squeeze(1)
    vision = inputs_embeds[0, image_indices, :]
    text = inputs_embeds[0, text_indices, :]
    similarity = (
        F.normalize(vision.float(), dim=-1)
        @ F.normalize(text.float(), dim=-1).T
    ).max(dim=-1).values
    keep_count = max(1, int(len(image_indices) * keep_ratio))
    chosen = similarity.topk(keep_count).indices
    chosen_positions = image_indices[chosen]
    all_positions = torch.cat([text_indices, chosen_positions]).sort().values
    pruned = inputs_embeds[:, all_positions, :]
    pruned_mask = torch.ones(
        pruned.shape[:2],
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    return pruned, pruned_mask, len(image_indices), keep_count


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")

    from vlmeval.dataset import build_dataset

    dataset = build_dataset(args.dataset)
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    metric = get_eval_metric(args.dataset)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    open(args.output, "w").close()

    for offset in range(args.eval_samples):
        dataset_index = args.eval_start + offset
        line = dataset[dataset_index]
        image_path, question = split_message(dataset.build_prompt(line))
        image = Image.open(image_path).convert("RGB")
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = processor(
            images=image, text=prompt, return_tensors="pt"
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        merged, image_mask = build_multimodal_embeds(model, inputs)

        with torch.no_grad():
            native_tokens = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )[:, inputs["input_ids"].shape[1]:]
        explicit_tokens = greedy_generate_from_embeds(
            model,
            merged,
            inputs["attention_mask"],
            args.max_new_tokens,
        )
        if not torch.equal(native_tokens, explicit_tokens):
            raise RuntimeError(
                f"sample {dataset_index}: explicit unpruned path changed tokens"
            )

        pruned, pruned_mask, full_vision, kept_vision = prune_image_tokens(
            merged,
            inputs["attention_mask"],
            image_mask,
            args.keep_ratio,
        )
        if pruned.shape[1] >= merged.shape[1]:
            raise RuntimeError("physical pruning did not reduce prefill length")
        pruned_tokens = greedy_generate_from_embeds(
            model, pruned, pruned_mask, args.max_new_tokens
        )

        references = parse_answer_field(line.get("answer", ""))
        native_prediction = processor.decode(
            native_tokens[0], skip_special_tokens=True
        ).strip()
        pruned_prediction = processor.decode(
            pruned_tokens[0], skip_special_tokens=True
        ).strip()
        record = {
            "dataset_index": int(dataset_index),
            "references": references,
            "metric": metric,
            "native_prediction": native_prediction,
            "explicit_unpruned_prediction": processor.decode(
                explicit_tokens[0], skip_special_tokens=True
            ).strip(),
            "pruned_prediction": pruned_prediction,
            "native_score": float(
                check_answer(native_prediction, references, metric)
            ),
            "pruned_score": float(
                check_answer(pruned_prediction, references, metric)
            ),
            "native_equals_explicit_tokens": True,
            "full_prefill_tokens": int(merged.shape[1]),
            "pruned_prefill_tokens": int(pruned.shape[1]),
            "full_vision_tokens": int(full_vision),
            "kept_vision_tokens": int(kept_vision),
            "keep_ratio": float(args.keep_ratio),
        }
        with open(args.output, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"[{offset + 1}/{args.eval_samples}] index={dataset_index} "
            f"vision={full_vision}->{kept_vision} "
            f"native={native_prediction!r} pruned={pruned_prediction!r}"
        )


if __name__ == "__main__":
    main()
