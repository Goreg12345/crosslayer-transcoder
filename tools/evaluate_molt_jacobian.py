#!/usr/bin/env python3
"""Evaluate MOLT Jacobian correlation against its underlying Gemma layer."""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from crosslayer_transcoder.metrics.jacobian_correlation import (
    flattened_jacobian_cosine,
    module_jacobian,
    molt_jacobian,
)
from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.molt import Molt
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)


DEFAULT_CHECKPOINTS = (
    "checkpoints/molt-gemma3-4b-it-rtx6000/layer-22/preact-1e-6/training.ckpt",
    "checkpoints/molt-gemma3-4b-it-rtx6000/layer-22/baseline/training.ckpt",
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="*", default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument("--model-name", default="google/gemma-3-4b-it")
    parser.add_argument("--layer", type=int, default=22)
    parser.add_argument("--num-datapoints", type=int, default=8)
    parser.add_argument("--dataset-name", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--dataset-split", default="test_sft")
    parser.add_argument("--num-conversations", type=int, default=16)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--collection-batch-size", type=int, default=2)
    parser.add_argument("--jacobian-chunk-size", type=int, default=32)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--molt-device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def gemma_layers(model):
    if hasattr(model, "language_model"):
        language_model = model.language_model
        if hasattr(language_model, "layers"):
            return language_model.layers
        return language_model.model.layers
    return model.model.layers


def build_molt_from_checkpoint(path: str, device: torch.device) -> tuple[Molt, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    state = checkpoint["state_dict"]
    ranks = []
    rank_index = 0
    while f"model.Us.{rank_index}" in state:
        ranks.append(state[f"model.Us.{rank_index}"].shape[1])
        rank_index += 1
    if not ranks:
        raise ValueError(f"Checkpoint {path!r} does not contain MOLT transform weights")
    N = state["model.Us.0"].shape[0]
    d_acts = state["model.e.weight"].shape[1]
    n_features = state["model.e.weight"].shape[0]
    n_layers = state["model.input_standardizer.mean"].shape[0]
    molt = Molt(
        d_acts=d_acts,
        N=N,
        ranks=ranks,
        nonlinearity=JumpReLU(
            theta=0.0, bandwidth=1.0, n_layers=1, d_features=n_features
        ),
        input_standardizer=DimensionwiseInputStandardizer(n_layers, d_acts),
        output_standardizer=DimensionwiseOutputStandardizer(n_layers, d_acts),
    )
    model_state = {
        key.removeprefix("model."): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    molt.load_state_dict(model_state)
    molt.input_standardizer.is_initialized = True
    molt.output_standardizer.is_initialized = True
    molt.requires_grad_(False).eval().to(device)
    metadata = {
        "global_step": checkpoint.get("global_step"),
        "epoch": checkpoint.get("epoch"),
        "pre_actv_loss": checkpoint.get("hyper_parameters", {}).get("pre_actv_loss"),
    }
    del checkpoint, state
    return molt, metadata


def evaluation_texts(tokenizer, dataset_name: str, split: str, count: int) -> list[str]:
    """Load a deterministic prefix of held-out conversations."""
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    texts = []
    for example in dataset.take(count):
        if "messages" in example:
            texts.append(
                tokenizer.apply_chat_template(
                    example["messages"], tokenize=False, add_generation_prompt=False
                )
            )
        else:
            texts.append(example.get("text", example.get("prompt")))
    return texts


@torch.no_grad()
def collect_inputs(
    model,
    tokenizer,
    texts,
    layer: int,
    count: int,
    max_length: int,
    collection_batch_size: int,
    device: torch.device,
):
    value_batches = []
    for start in range(0, len(texts), collection_batch_size):
        encoded = tokenizer(
            texts[start : start + collection_batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(device)
        captured = []

        def hook(_module, args):
            captured.append(args[0].detach())

        handle = gemma_layers(model)[layer].pre_feedforward_layernorm.register_forward_pre_hook(hook)
        model(**encoded, use_cache=False)
        handle.remove()
        value_batches.append(
            captured[0][encoded["attention_mask"].bool()].to("cpu")
        )
    values = torch.cat(value_batches)
    if values.shape[0] < count:
        raise ValueError(f"Only {values.shape[0]} non-padding token datapoints available")
    # Evenly cover the deterministic prompt corpus rather than taking only the
    # beginning of its first sequence.
    indices = torch.linspace(0, values.shape[0] - 1, count).long()
    return values[indices].to(device)


def main() -> None:
    args = parse_args()
    model_device = torch.device(args.device)
    molt_device = torch.device(args.molt_device or args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).to(model_device).eval()
    model.requires_grad_(False)
    texts = evaluation_texts(
        tokenizer, args.dataset_name, args.dataset_split, args.num_conversations
    )
    inputs = collect_inputs(
        model,
        tokenizer,
        texts,
        args.layer,
        args.num_datapoints,
        args.max_sequence_length,
        args.collection_batch_size,
        model_device,
    )
    layer = gemma_layers(model)[args.layer]

    def target_function(x):
        return layer.post_feedforward_layernorm(layer.mlp(layer.pre_feedforward_layernorm(x)))

    # Target Jacobians are shared by every checkpoint.
    with torch.enable_grad():
        target_jacobians = module_jacobian(
            target_function,
            inputs,
            chunk_size=args.jacobian_chunk_size,
            output_device="cpu",
        )

    results = []
    for checkpoint_path in args.checkpoints:
        molt, metadata = build_molt_from_checkpoint(checkpoint_path, molt_device)
        correlation_batches = []
        l0_batches = []
        for start in range(0, args.num_datapoints, args.evaluation_batch_size):
            stop = min(start + args.evaluation_batch_size, args.num_datapoints)
            input_batch = inputs[start:stop].to(molt_device)
            replacement_jacobians = molt_jacobian(molt, input_batch, layer=args.layer)
            correlation_batches.append(
                flattened_jacobian_cosine(
                    replacement_jacobians,
                    target_jacobians[start:stop].to(molt_device),
                ).cpu()
            )
            with torch.no_grad():
                l0_batches.append(
                    (molt.nonlinearity(
                        molt.e(molt.input_standardizer(input_batch, args.layer))
                    ) != 0).float().sum(-1).cpu()
                )
            del replacement_jacobians
        correlations = torch.cat(correlation_batches)
        l0 = torch.cat(l0_batches)
        result = {
            "checkpoint": str(checkpoint_path),
            **metadata,
            "layer": args.layer,
            "num_datapoints": args.num_datapoints,
            "dataset": args.dataset_name,
            "dataset_split": args.dataset_split,
            "num_conversations": args.num_conversations,
            "jacobian_correlation": correlations.mean().item(),
            "jacobian_correlation_std": correlations.std(unbiased=False).item(),
            "jacobian_correlation_se": (
                correlations.std(unbiased=True) / correlations.numel() ** 0.5
            ).item(),
            "mean_l0": l0.mean().item(),
            "per_datapoint": correlations.cpu().tolist(),
        }
        results.append(result)
        print(json.dumps(result))
        del molt
        torch.cuda.empty_cache()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
