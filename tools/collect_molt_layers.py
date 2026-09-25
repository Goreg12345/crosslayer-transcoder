#!/usr/bin/env python3
"""Collect several completed MOLT layers in one Qwen pass over a fixed token cache."""

import argparse
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from crosslayer_transcoder.dashboard.collection import GateCollection, write_json
from crosslayer_transcoder.dashboard.gates import MoltGates


def resume_batch_start(stores, batch_size):
    return min(s.conversations for s in stores.values()) // batch_size * batch_size


def main(args):
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(1, args.gpu_memory_gib * 1024**3 / total), device
        )
    reference = json.loads((args.reference / "metadata.json").read_text())
    if reference["tokens"] != 10_000_000 or not reference.get("target_reached"):
        raise ValueError("Reference must be the completed 10M-token sample")
    with np.load(args.reference / "activations.npz") as archive:
        tokens, offsets = archive["tokens"], archive["offsets"]
    token_hash = hashlib.sha256(tokens.tobytes() + offsets.tobytes()).hexdigest()
    count = min(args.limit_conversations or len(offsets) - 1, len(offsets) - 1)
    target = int(offsets[count])
    if args.batch_size != reference["batch_size"]:
        raise ValueError(
            "Use the reference batch size to preserve the extraction protocol"
        )
    stores, gates, metadata, locks, records = {}, {}, {}, [], {}
    for path in args.checkpoints:
        before = path.stat()
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if checkpoint["global_step"] != 100_000:
            raise ValueError(f"Checkpoint is still training: {path}")
        layer = checkpoint["hyper_parameters"]["layer"]
        if layer in stores:
            raise ValueError("Duplicate layer")
        hp, state = checkpoint["datamodule_hyper_parameters"], checkpoint["state_dict"]
        if (
            hp["model_arch"] != "qwen3"
            or hp["activation_input_location"] != "pre_norm"
            or hp["model_name"] != reference["model"]
            or hp["dataset_name"] != reference["dataset"]
        ):
            raise ValueError("Checkpoint and reference model/data protocol disagree")
        ranks = []
        index = 0
        while f"model.Us.{index}" in state:
            u = state[f"model.Us.{index}"]
            ranks.extend([u.shape[1]] * u.shape[0])
            index += 1
        root = args.output_root / f"molt-qwen-dashboard-campaign-layer{layer}-10m"
        root.mkdir(parents=True, exist_ok=True)
        lock = (root / ".collection.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
        settings = dict(
            checkpoint=str(path.resolve()),
            checkpoint_bytes=before.st_size,
            checkpoint_mtime=before.st_mtime,
            global_step=checkpoint["global_step"],
            model=hp["model_name"],
            layer=layer,
            dataset=hp["dataset_name"],
            split=reference["split"],
            ranks=ranks,
            n_features=len(ranks),
            max_length=reference["max_length"],
            batch_size=args.batch_size,
            target_tokens=target,
            conversation_limit=args.limit_conversations,
            feature_definition=reference["feature_definition"],
            precision=reference["precision"],
            sampling="Exact token IDs and conversation boundaries from the fixed layer-22 reference; one shared forward pass",
            special_tokens=reference["special_tokens"],
            reference=str(args.reference.resolve()),
            reference_tokens_sha256=token_hash,
            collector_version="shared-forward-v1",
        )
        metadata[layer] = settings
        stores[layer] = GateCollection(root, settings)
        if stores[layer].conversations > count or stores[layer].tokens != int(
            offsets[stores[layer].conversations]
        ):
            raise ValueError("Committed shard boundaries disagree with the reference")
        gates[layer] = MoltGates(state, layer).to(device).eval()
        if (path.stat().st_mtime_ns, path.stat().st_size) != (
            before.st_mtime_ns,
            before.st_size,
        ):
            raise ValueError(
                "Checkpoint changed while loading; wait for training to finish"
            )
        records[layer] = []
        del checkpoint, state, u
    started = time.monotonic()
    initial_tokens = min(store.tokens for store in stores.values())
    last_layer = max(stores)
    needs_forward = min(s.conversations for s in stores.values()) < count
    model = None
    handles = []
    batch_ids, batch_conversations = [], []
    seen = set()

    class Captured(Exception):
        pass

    def hook(layer):
        def capture(_module, inputs):
            if layer in seen:
                raise RuntimeError("Layer captured twice in a batch")
            seen.add(layer)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                acts = gates[layer](inputs[0])
            if not torch.isfinite(acts).all():
                raise ValueError("Non-finite gates")
            for row, (conversation, ids) in enumerate(
                zip(batch_conversations, batch_ids)
            ):
                if conversation < stores[layer].conversations:
                    continue
                if conversation != stores[layer].conversations + len(records[layer]):
                    raise ValueError("Missing or duplicate conversation during resume")
                active = acts[row, : len(ids)]
                nz = active.nonzero()
                records[layer].append(
                    dict(
                        tokens=ids,
                        positions=nz[:, 0].cpu().numpy(),
                        features=nz[:, 1].cpu().numpy(),
                        values=active[nz[:, 0], nz[:, 1]].float().cpu().numpy(),
                    )
                )
            if layer == last_layer:
                raise Captured()

        return capture

    def commit():
        for layer, store in stores.items():
            store.append(records[layer])
            records[layer].clear()
            elapsed = time.monotonic() - started
            write_json(
                store.root / "collection-progress.json",
                dict(
                    conversations=store.conversations,
                    tokens=store.tokens,
                    target_tokens=target,
                    active_entries=store.entries,
                    tokens_per_second=(
                        min(s.tokens for s in stores.values()) - initial_tokens
                    )
                    / max(elapsed, 1e-6),
                    elapsed_seconds=elapsed,
                    complete=False,
                    shared_layers=sorted(stores),
                    peak_cuda_gib=torch.cuda.max_memory_reserved(device) / 1024**3
                    if device.type == "cuda"
                    else 0,
                ),
            )
        print(
            json.dumps(
                dict(
                    event="collected",
                    layers=sorted(stores),
                    conversations=min(s.conversations for s in stores.values()),
                    tokens=min(s.tokens for s in stores.values()),
                    elapsed_seconds=time.monotonic() - started,
                )
            ),
            flush=True,
        )

    if needs_forward:
        pad_token_id = AutoTokenizer.from_pretrained(reference["model"]).pad_token_id
        model = AutoModel.from_pretrained(
            reference["model"], torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        model.layers = model.layers[: last_layer + 1]
        model.layers[last_layer].mlp = torch.nn.Identity()
        model.to(device).eval().requires_grad_(False)
        gc.collect()
        print(
            json.dumps(
                dict(
                    event="model_ready",
                    layers=sorted(stores),
                    allocated_gib=torch.cuda.memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else 0,
                )
            ),
            flush=True,
        )
        handles = [
            model.layers[layer].post_attention_layernorm.register_forward_pre_hook(
                hook(layer)
            )
            for layer in stores
        ]
        try:
            with torch.inference_mode():
                for start in range(
                    resume_batch_start(stores, args.batch_size), count, args.batch_size
                ):
                    batch_conversations = list(
                        range(start, min(start + args.batch_size, count))
                    )
                    batch_ids = [
                        tokens[offsets[c] : offsets[c + 1]] for c in batch_conversations
                    ]
                    width = max(map(len, batch_ids))
                    batch = torch.full(
                        (len(batch_ids), width),
                        pad_token_id,
                        device=device,
                        dtype=torch.long,
                    )
                    mask = torch.zeros_like(batch)
                    for row, ids in enumerate(batch_ids):
                        batch[row, : len(ids)] = torch.as_tensor(
                            ids, device=device, dtype=torch.long
                        )
                        mask[row, : len(ids)] = 1
                    seen.clear()
                    try:
                        model(input_ids=batch, attention_mask=mask, use_cache=False)
                    except Captured:
                        pass
                    if seen != set(stores):
                        raise RuntimeError("Not every requested layer was captured")
                    if max(map(len, records.values())) >= args.shard_conversations:
                        commit()
                commit()
        finally:
            for handle in handles:
                handle.remove()
    model = None
    gates.clear()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for layer, store in stores.items():
        if store.tokens != target or store.conversations != count:
            raise ValueError("Collection ended below target")
        if not (store.root / "metadata.json").exists():
            store.finish()
            gc.collect()
        final = dict(
            metadata[layer],
            conversations=store.conversations,
            tokens=store.tokens,
            sample_l0=store.entries / store.tokens,
            target_reached=True,
        )
        write_json(store.root / "metadata.json", final)
        shutil.copyfile(args.reference / "vocab.json", store.root / "vocab.json")
        write_json(
            store.root / "collection-progress.json",
            dict(
                conversations=count,
                tokens=target,
                target_tokens=target,
                active_entries=store.entries,
                complete=True,
                target_reached=True,
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        print(
            json.dumps(
                dict(
                    event="layer_complete",
                    layer=layer,
                    tokens=target,
                    entries=store.entries,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("results/molt-qwen-dashboard-campaign-layer22-10m"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-conversations", type=int, default=128)
    parser.add_argument("--gpu-memory-gib", type=float, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--limit-conversations",
        type=int,
        help="Small validation run; production uses the full reference",
    )
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or args.shard_conversations < 1
        or args.gpu_memory_gib <= 0
        or (args.limit_conversations is not None and args.limit_conversations < 1)
    ):
        parser.error("Sizes and limits must be positive")
    main(args)
