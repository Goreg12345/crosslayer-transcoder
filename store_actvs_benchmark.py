import os
import time

import datasets
import h5py
import nnsight
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import activation_server.text_dataset as text_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = ""


# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

if cuda_available:
    # Number of GPUs
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs:", num_gpus)

    # List each device’s name
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {name}")
else:
    print("No CUDA devices found")


def create_model(use_fp16=False):
    """Create model with optional FP16"""
    model = nnsight.LanguageModel(
        "openai-community/gpt2", device_map="cpu", dispatch=True
    )
    model.requires_grad_(False)

    if use_fp16:
        # For CPU FP16, convert model weights to half precision
        model.half()
        print("✓ Model converted to FP16")

    return model


dataset = datasets.load_dataset("Skylion007/openwebtext", split="train")


@torch.no_grad()
def extract_activations(model, tokens):
    with model.trace(tokens) as tracer:
        mlp_ins = []
        mlp_outs = []
        for i in range(12):
            mlp_in = model.transformer.h[i].ln_2.input.save()
            mlp_ins.append(mlp_in)
            mlp_out = model.transformer.h[i].mlp.output.save()
            mlp_outs.append(mlp_out)
    # batch layer in/out d_model
    mlp_ins = torch.stack(mlp_ins, dim=2)
    mlp_outs = torch.stack(mlp_outs, dim=2)
    mlp_acts = torch.stack([mlp_ins, mlp_outs], dim=2)
    return mlp_acts  # batch seq_len in/out n_layer d_model


def benchmark_config(batch_size, use_fp16=False, duration=30):
    """Benchmark a specific configuration"""
    print(f"\n=== Benchmark: batch_size={batch_size}, fp16={use_fp16} ===")

    model = create_model(use_fp16)
    if model is None:
        return 0

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
            num_workers=8,
            prefetch_factor=4,
            worker_init_fn=text_dataset.worker_init_fn,
        )
    )

    start_time = time.time()
    end_time = start_time + duration
    total_tokens = 0
    total_batches = 0

    print("Running benchmark...")
    try:
        for batch in tqdm(text_dataset_loader):
            if time.time() > end_time:
                break

            # Prepend BOS token
            batch = torch.roll(batch, shifts=1, dims=1)
            batch[:, 0] = model.config.bos_token_id

            # Extract activations
            mlp_acts = extract_activations(model, batch)

            total_tokens += batch.numel()
            total_batches += 1

    except Exception as e:
        print(f"Error during benchmark: {e}")
        return 0

    elapsed = time.time() - start_time
    tokens_per_sec = total_tokens / elapsed

    print(f"Tokens/sec: {tokens_per_sec:,.1f}")
    print(f"Batches/sec: {total_batches/elapsed:.1f}")
    print(f"Avg batch size: {total_tokens/total_batches:.1f} tokens")

    return tokens_per_sec


if __name__ == "__main__":
    results = {}

    # Test different batch sizes
    for batch_size in [40, 80, 160, 320]:
        results[f"batch_{batch_size}"] = benchmark_config(batch_size, False, 20)

    # Test FP16 with optimal batch size
    print("\n=== Testing FP16 ===")
    results["fp16_batch_160"] = benchmark_config(160, True, 20)

    print("\n=== RESULTS ===")
    baseline = results.get("batch_40", 2100)
    for config, tokens_per_sec in sorted(
        results.items(), key=lambda x: x[1], reverse=True
    ):
        if tokens_per_sec > 0:
            improvement = tokens_per_sec / baseline
            print(
                f"{config:15}: {tokens_per_sec:8,.1f} tokens/sec ({improvement:.1f}x)"
            )
