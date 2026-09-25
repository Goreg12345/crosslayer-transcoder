#!/usr/bin/env python3
"""Collect scalar MOLT gates on held-out chat tokens, without evaluating transforms."""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

from crosslayer_transcoder.dashboard.collection import GateCollection, write_json
from crosslayer_transcoder.dashboard.gates import MoltGates

DEFAULT = "checkpoints/molt-qwen3-4b/layer-22/chat-ultrachat-lr4e-5-sparsity3e-5/training.ckpt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT)
    p.add_argument("--output", type=Path, default=Path("results/molt-qwen-dashboard"))
    p.add_argument("--conversations", type=int, help="Optional conversation limit")
    p.add_argument(
        "--tokens", type=int, help="Token target, across separate conversations"
    )
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--shard-conversations", type=int, default=128)
    p.add_argument(
        "--gpu-memory-gib", type=float, help="Cap this process's CUDA allocator"
    )
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.conversations is None and args.tokens is None:
        args.conversations = 128
    if any(
        v is not None and v < 1
        for v in (
            args.conversations,
            args.tokens,
            args.max_length,
            args.batch_size,
            args.shard_conversations,
            args.gpu_memory_gib,
        )
    ):
        p.error("limits and sizes must be positive")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda" and args.gpu_memory_gib:
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, args.gpu_memory_gib * 1024**3 / total), device
        )
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=False
    )
    state = checkpoint["state_dict"]
    hp = checkpoint["datamodule_hyper_parameters"]
    layer = checkpoint["hyper_parameters"]["layer"]
    if hp["model_arch"] != "qwen3" or hp["activation_input_location"] != "pre_norm":
        raise ValueError("This collector supports Qwen3 pre_norm checkpoints")
    gates = MoltGates(state, layer).to(device).eval()
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
        batch_size=args.batch_size,
        target_tokens=args.tokens,
        conversation_limit=args.conversations,
        feature_definition="JumpReLU(e((pre_norm_input - saved_mean) / saved_std)); active iff gate > 0",
        precision="BF16 extraction and autocast, matching training",
        sampling="First conversations in held-out split; each truncated independently; final conversation may be shortened to the token budget",
        special_tokens="Included, matching training; right padding excluded",
    )
    store = GateCollection(args.output, metadata)
    del checkpoint, state, u
    tokenizer = AutoTokenizer.from_pretrained(hp["model_name"])
    model = AutoModel.from_pretrained(
        hp["model_name"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    # The hook exits before this layer's MLP. Remove unused weights before CUDA transfer.
    model.layers = model.layers[: layer + 1]
    model.layers[layer].mlp = torch.nn.Identity()
    model.to(device).eval().requires_grad_(False)
    gc.collect()
    if device.type == "cuda":
        print(
            f"Model allocated {torch.cuda.memory_allocated(device) / 1024**3:.2f} GiB",
            flush=True,
        )
    dataset = load_dataset(hp["dataset_name"], split="test_sft", streaming=True)
    if store.conversations:
        dataset = dataset.skip(store.conversations)
    started, initial_tokens = time.monotonic(), store.tokens
    records, batch_ids = [], []
    pending_tokens = 0

    class Captured(Exception):
        pass

    captured = []

    def hook(_module, inputs):
        captured.append(inputs[0])
        raise Captured()

    def run_batch():
        width = max(map(len, batch_ids))
        batch = torch.full(
            (len(batch_ids), width),
            tokenizer.pad_token_id,
            device=device,
            dtype=torch.long,
        )
        mask = torch.zeros_like(batch)
        for row, ids in enumerate(batch_ids):
            batch[row, : len(ids)] = torch.tensor(ids, device=device)
            mask[row, : len(ids)] = 1
        captured.clear()
        try:
            model(input_ids=batch, attention_mask=mask, use_cache=False)
        except Captured:
            pass
        if len(captured) != 1:
            raise RuntimeError(
                "Layer hook did not capture exactly one activation tensor"
            )
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            acts = gates(captured.pop())
        if not torch.isfinite(acts).all():
            raise ValueError("Non-finite gates")
        for row, ids in enumerate(batch_ids):
            active = acts[row, : len(ids)]
            nz = active.nonzero()
            records.append(
                dict(
                    tokens=np.asarray(ids, dtype=np.int32),
                    positions=nz[:, 0].cpu().numpy(),
                    features=nz[:, 1].cpu().numpy(),
                    values=active[nz[:, 0], nz[:, 1]].float().cpu().numpy(),
                )
            )
        batch_ids.clear()

    def commit():
        nonlocal pending_tokens
        store.append(records)
        records.clear()
        pending_tokens = 0
        elapsed = time.monotonic() - started
        rate = (store.tokens - initial_tokens) / max(elapsed, 1e-6)
        progress = dict(
            conversations=store.conversations,
            tokens=store.tokens,
            target_tokens=args.tokens,
            active_entries=store.entries,
            tokens_per_second=rate,
            elapsed_seconds=elapsed,
            complete=False,
        )
        if device.type == "cuda":
            progress["peak_cuda_gib"] = torch.cuda.max_memory_reserved(device) / 1024**3
        write_json(args.output / "collection-progress.json", progress)
        print(
            f"{store.conversations} conversations, {store.tokens:,} tokens, {rate:,.0f} tokens/s",
            flush=True,
        )

    handle = model.layers[layer].post_attention_layernorm.register_forward_pre_hook(
        hook
    )
    try:
        with torch.inference_mode():
            for example in dataset:
                if (
                    args.conversations
                    and store.conversations + len(records) + len(batch_ids)
                    >= args.conversations
                ):
                    break
                remaining = (
                    args.tokens - store.tokens - pending_tokens
                    if args.tokens
                    else args.max_length
                )
                if remaining <= 0:
                    break
                ids = tokenizer.apply_chat_template(
                    example["messages"], tokenize=True, add_generation_prompt=False
                )[: min(args.max_length, remaining)]
                if not ids:
                    raise ValueError("Empty conversation cannot be recorded")
                batch_ids.append(ids)
                pending_tokens += len(ids)
                if len(batch_ids) >= args.batch_size:
                    run_batch()
                    if len(records) >= args.shard_conversations:
                        commit()
            if batch_ids:
                run_batch()
            if records:
                commit()
    finally:
        handle.remove()
    model = gates = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    unique_tokens = store.finish()
    metadata.update(
        conversations=store.conversations,
        tokens=store.tokens,
        sample_l0=store.entries / store.tokens,
        target_reached=args.tokens is None or store.tokens >= args.tokens,
    )
    write_json(args.output / "metadata.json", metadata)
    vocab = {str(i): tokenizer.decode([int(i)]) for i in unique_tokens}
    write_json(args.output / "vocab.json", vocab)
    write_json(
        args.output / "collection-progress.json",
        dict(
            conversations=store.conversations,
            tokens=store.tokens,
            target_tokens=args.tokens,
            active_entries=store.entries,
            complete=True,
            target_reached=metadata["target_reached"],
        ),
    )
    print(f"Saved {args.output}; sample L0={metadata['sample_l0']:.4f}", flush=True)
    if not metadata["target_reached"]:
        raise RuntimeError(
            "Dataset or conversation limit exhausted before token target"
        )


if __name__ == "__main__":
    main()
