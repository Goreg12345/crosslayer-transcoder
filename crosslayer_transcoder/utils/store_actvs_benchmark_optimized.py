import os
import time

import activation_server.text_dataset as text_dataset
import datasets
import nnsight
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Set optimal thread count for single model
torch.set_num_threads(96)
torch.set_num_interop_threads(4)

print(f"PyTorch threads: {torch.get_num_threads()}")
print(f"PyTorch interop threads: {torch.get_num_interop_threads()}")


def create_model(use_float16=False):
    """Create and optimize model"""
    model = nnsight.LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
    model.requires_grad_(False)

    if use_float16:
        model.half()

    return model


def extract_activations_optimized(model, tokens, use_float16=False):
    """Optimized activation extraction"""
    # Don't convert tokens to float16 - they must stay as integers for embedding

    with model.trace(tokens):
        mlp_ins = []
        mlp_outs = []
        for i in range(12):
            mlp_in = model.transformer.h[i].ln_2.input.save()
            mlp_ins.append(mlp_in)
            mlp_out = model.transformer.h[i].mlp.output.save()
            mlp_outs.append(mlp_out)

    # Stack more efficiently
    mlp_ins = torch.stack(mlp_ins, dim=2)
    mlp_outs = torch.stack(mlp_outs, dim=2)
    mlp_acts = torch.stack([mlp_ins, mlp_outs], dim=2)

    # Convert activations to float16 after computation
    if use_float16:
        mlp_acts = mlp_acts.half()

    return mlp_acts


def benchmark_single_model(batch_size=40, use_float16=False, duration=30):
    """Benchmark single model with different batch sizes"""
    print(f"\n=== Single Model Benchmark (batch_size={batch_size}, fp16={use_float16}) ===")

    model = create_model(use_float16)
    dataset = datasets.load_dataset("Skylion007/openwebtext", split="train")

    token_dataset = text_dataset.TextDataset(
        dataset,
        model.tokenizer,
        batch_size,
        drop_last_batch=False,
        seq_len=1023,
    )

    text_dataset_loader = iter(
        DataLoader(
            token_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=8,  # Increased workers
            prefetch_factor=4,
            worker_init_fn=text_dataset.worker_init_fn,
            pin_memory=True,  # Better memory transfer
        )
    )

    start_time = time.time()
    end_time = start_time + duration
    total_tokens = 0
    total_batches = 0

    print("Running benchmark...")
    for batch in tqdm(text_dataset_loader):
        if time.time() > end_time:
            break

        # Prepend BOS token
        batch = torch.roll(batch, shifts=1, dims=1)
        batch[:, 0] = model.config.bos_token_id

        # Extract activations
        extract_activations_optimized(model, batch, use_float16)

        total_tokens += batch.numel()
        total_batches += 1

    elapsed = time.time() - start_time
    tokens_per_sec = total_tokens / elapsed

    print(f"Tokens/sec: {tokens_per_sec:,.1f}")
    print(f"Batches/sec: {total_batches / elapsed:.1f}")
    print(f"Average batch size: {total_tokens / total_batches:.1f} tokens")

    return tokens_per_sec


if __name__ == "__main__":
    results = {}
    baseline = 2100  # Your current performance

    # Test different batch sizes
    for batch_size in [40, 80, 160, 320]:
        try:
            results[f"batch_{batch_size}"] = benchmark_single_model(batch_size, False, 20)
        except Exception as e:
            print(f"Failed batch_size={batch_size}: {e}")

    # Test float16 with best batch size
    try:
        results["fp16_batch_160"] = benchmark_single_model(160, True, 20)
    except Exception as e:
        print(f"Failed fp16: {e}")

    # Test with more dataloader workers
    try:
        model = create_model(False)
        dataset = datasets.load_dataset("Skylion007/openwebtext", split="train")

        token_dataset = text_dataset.TextDataset(
            dataset,
            model.tokenizer,
            160,
            drop_last_batch=False,
            seq_len=1023,
        )

        text_dataset_loader = iter(
            DataLoader(
                token_dataset,
                batch_size=None,
                shuffle=False,
                num_workers=16,
                prefetch_factor=8,
                worker_init_fn=text_dataset.worker_init_fn,
            )
        )

        start_time = time.time()
        total_tokens = 0
        total_batches = 0

        print("\n=== Testing 16 workers ===")
        for i, batch in enumerate(tqdm(text_dataset_loader)):
            if i >= 50:  # Test 50 batches
                break

            batch = torch.roll(batch, shifts=1, dims=1)
            batch[:, 0] = model.config.bos_token_id
            mlp_acts = extract_activations_optimized(model, batch, False)

            total_tokens += batch.numel()
            total_batches += 1

        elapsed = time.time() - start_time
        results["16_workers"] = total_tokens / elapsed

    except Exception as e:
        print(f"Failed 16 workers: {e}")

    print("\n=== RESULTS SUMMARY ===")
    for config, tokens_per_sec in sorted(results.items(), key=lambda x: x[1], reverse=True):
        improvement = tokens_per_sec / baseline
        print(f"{config:20}: {tokens_per_sec:8,.1f} tokens/sec ({improvement:.1f}x)")
