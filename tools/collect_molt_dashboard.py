#!/usr/bin/env python3
"""Collect scalar MOLT gates on held-out chat tokens, without evaluating transforms."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from crosslayer_transcoder.dashboard.gates import MoltGates

DEFAULT = "checkpoints/molt-qwen3-4b/layer-22/chat-ultrachat-lr4e-5-sparsity3e-5/training.ckpt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT)
    p.add_argument("--output", type=Path, default=Path("results/molt-qwen-dashboard"))
    p.add_argument("--conversations", type=int, default=128)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.conversations < 1 or args.max_length < 1:
        p.error("conversations and max-length must be positive")
    torch.set_num_threads(4)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    state = checkpoint["state_dict"]
    hp = checkpoint["datamodule_hyper_parameters"]
    layer = checkpoint["hyper_parameters"]["layer"]
    if hp["model_arch"] != "qwen3" or hp["activation_input_location"] != "pre_norm":
        raise ValueError("This collector supports Qwen3 pre_norm checkpoints")
    gates = MoltGates(state, layer).to(args.device).eval()
    ranks = []
    i = 0
    while f"model.Us.{i}" in state:
        u = state[f"model.Us.{i}"]
        ranks.extend([u.shape[1]] * u.shape[0])
        i += 1
    metadata = dict(
        checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_bytes=Path(args.checkpoint).stat().st_size,
        checkpoint_mtime=Path(args.checkpoint).stat().st_mtime,
        global_step=checkpoint["global_step"],
        model=hp["model_name"],
        layer=layer,
        dataset=hp["dataset_name"],
        split="test_sft",
        ranks=ranks,
        n_features=len(ranks),
        max_length=args.max_length,
        feature_definition="JumpReLU(e((pre_norm_input - saved_mean) / saved_std)); active iff gate > 0",
        precision="BF16 extraction and autocast, matching training",
        sampling="First conversations in held-out split; each truncated independently",
        special_tokens="Included, matching training; padding excluded",
    )
    del checkpoint, state
    tokenizer = AutoTokenizer.from_pretrained(hp["model_name"])
    model = (
        AutoModel.from_pretrained(
            hp["model_name"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(args.device)
        .eval()
    )
    model.requires_grad_(False)
    dataset = load_dataset(hp["dataset_name"], split="test_sft", streaming=True)
    token_arrays, positions, features, values, offsets = [], [], [], [], [0]

    class Captured(Exception):
        pass

    captured = []

    def hook(_module, inputs):
        captured.append(inputs[0])
        raise Captured()  # No later layers or vocabulary logits are needed.

    handle = model.layers[layer].post_attention_layernorm.register_forward_pre_hook(
        hook
    )
    try:
        with torch.inference_mode():
            for i, example in enumerate(dataset.take(args.conversations)):
                ids = tokenizer.apply_chat_template(
                    example["messages"], tokenize=True, add_generation_prompt=False
                )[: args.max_length]
                batch = torch.tensor([ids], device=args.device)
                captured.clear()
                try:
                    model(
                        input_ids=batch,
                        attention_mask=torch.ones_like(batch),
                        use_cache=False,
                    )
                except Captured:
                    pass
                if len(captured) != 1:
                    raise RuntimeError(
                        "Layer hook did not capture exactly one activation tensor"
                    )
                with torch.autocast(
                    device_type=torch.device(args.device).type, dtype=torch.bfloat16
                ):
                    acts = gates(captured.pop()[0])
                if not torch.isfinite(acts).all():
                    raise ValueError("Non-finite gates")
                nz = acts.nonzero()
                positions.append(nz[:, 0].cpu().numpy() + offsets[-1])
                features.append(nz[:, 1].cpu().numpy())
                values.append(acts[nz[:, 0], nz[:, 1]].float().cpu().numpy())
                token_arrays.append(np.array(ids, dtype=np.int32))
                offsets.append(offsets[-1] + len(ids))
                if (i + 1) % 8 == 0:
                    print(f"{i + 1} conversations, {offsets[-1]:,} tokens", flush=True)
    finally:
        handle.remove()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "activations.npz",
        tokens=np.concatenate(token_arrays),
        offsets=np.array(offsets),
        positions=np.concatenate(positions),
        features=np.concatenate(features),
        values=np.concatenate(values),
    )
    metadata.update(
        conversations=len(token_arrays),
        tokens=offsets[-1],
        sample_l0=sum(len(v) for v in values) / offsets[-1],
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    vocab = {
        str(i): tokenizer.decode([i])
        for i in np.unique(np.concatenate(token_arrays)).tolist()
    }
    (args.output / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))
    print(f"Saved {args.output}; sample L0={metadata['sample_l0']:.4f}", flush=True)


if __name__ == "__main__":
    main()
