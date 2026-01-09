#!/usr/bin/env python3
"""
Benchmark script for TextDataset and DataLoader token throughput.
Tests how many tokens per second can be generated from the text pipeline.
"""

import time

import nnsight
import torch
from data import text_dataset
from datasets import load_dataset
from torch.utils.data import DataLoader


def benchmark_textdataset(
    dataset_name: str = "Skylion007/openwebtext",
    model_name: str = "openai-community/gpt2",
    batch_size: int = 32,
    seq_len: int = 1024,
    num_workers: int = 0,
    max_batches: int = 10,
    max_duration: float = 60.0,  # Max 60 seconds
):
    """
    Benchmark TextDataset + DataLoader token throughput.

    Args:
        dataset_name: HuggingFace dataset name
        model_name: Model name for tokenizer
        batch_size: Batch size for DataLoader
        seq_len: Sequence length per sample
        num_workers: Number of DataLoader workers
        max_batches: Maximum batches to process
        max_duration: Maximum time to run benchmark (seconds)
    """
    print("=== TextDataset Benchmark ===")
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Workers: {num_workers}")
    print(f"Max batches: {max_batches}")
    print(f"Max duration: {max_duration}s")
    print()

    # Load model for tokenizer
    print("Loading model/tokenizer...")
    start_time = time.time()
    model = nnsight.LanguageModel(
        model_name,
        device_map="cpu",
        dispatch=True,
        torch_dtype=torch.float32,
    )
    model_load_time = time.time() - start_time
    print(f"✓ Model loaded in {model_load_time:.2f}s")

    # Load dataset
    print("Loading dataset...")
    start_time = time.time()
    dataset = load_dataset(dataset_name, split="train")
    dataset_load_time = time.time() - start_time
    print(f"✓ Dataset loaded in {dataset_load_time:.2f}s")
    print(f"✓ Dataset size: {len(dataset):,} samples")

    # Create TextDataset
    print("Creating TextDataset...")
    start_time = time.time()
    token_dataset = text_dataset.TextDataset(
        dataset,
        model.tokenizer,
        batch_size,
        drop_last_batch=False,
        seq_len=seq_len,
    )
    textdataset_time = time.time() - start_time
    print(f"✓ TextDataset created in {textdataset_time:.2f}s")

    # Create DataLoader
    print("Creating DataLoader...")
    start_time = time.time()

    if num_workers > 0:
        data_loader = DataLoader(
            token_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=2,
            worker_init_fn=text_dataset.worker_init_fn,
        )
    else:
        data_loader = DataLoader(
            token_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
        )

    # This is the expensive line we're benchmarking
    data_loader_iter = iter(data_loader)
    dataloader_time = time.time() - start_time
    print(f"✓ DataLoader created in {dataloader_time:.2f}s")
    print()

    # Benchmark token generation
    print("=== Starting Token Generation Benchmark ===")
    benchmark_start = time.time()

    total_tokens = 0
    total_batches = 0
    batch_times = []

    try:
        for batch_idx in range(max_batches):
            # Check timeout
            elapsed = time.time() - benchmark_start
            if elapsed > max_duration:
                print(f"⏰ Timeout reached ({max_duration}s)")
                break

            # Get next batch
            batch_start = time.time()
            try:
                batch = next(data_loader_iter)
            except StopIteration:
                print("🔚 Dataset exhausted")
                break
            batch_time = time.time() - batch_start

            # Count tokens in batch
            batch_tokens = batch.numel()
            total_tokens += batch_tokens
            total_batches += 1
            batch_times.append(batch_time)

            # Progress update
            tokens_per_sec = batch_tokens / batch_time if batch_time > 0 else 0
            print(
                f"Batch {batch_idx + 1:3d}: "
                f"{batch_tokens:,} tokens in {batch_time:.3f}s "
                f"({tokens_per_sec:,.0f} tokens/s) "
                f"shape={list(batch.shape)}"
            )

    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    # Calculate final statistics
    total_time = time.time() - benchmark_start

    print()
    print("=== Benchmark Results ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Total batches: {total_batches}")
    print(f"Total tokens: {total_tokens:,}")

    if total_time > 0:
        avg_tokens_per_sec = total_tokens / total_time
        print(f"Average tokens/sec: {avg_tokens_per_sec:,.0f}")

    if batch_times:
        avg_batch_time = sum(batch_times) / len(batch_times)
        print(f"Average batch time: {avg_batch_time:.3f}s")
        print(f"Min batch time: {min(batch_times):.3f}s")
        print(f"Max batch time: {max(batch_times):.3f}s")

    print()
    print("=== Setup Times ===")
    print(f"Model load: {model_load_time:.2f}s")
    print(f"Dataset load: {dataset_load_time:.2f}s")
    print(f"TextDataset: {textdataset_time:.2f}s")
    print(f"DataLoader init: {dataloader_time:.2f}s")
    print(f"Total setup: {model_load_time + dataset_load_time + textdataset_time + dataloader_time:.2f}s")


def compare_workers():
    """Compare performance with different numbers of workers."""
    print("=== Worker Comparison Benchmark ===\n")

    worker_configs = [0, 1, 2, 4]

    for num_workers in worker_configs:
        print(f"\n{'=' * 50}")
        print(f"Testing with {num_workers} workers")
        print("=" * 50)

        benchmark_textdataset(
            num_workers=num_workers,
            max_batches=5,  # Shorter test for comparison
            max_duration=30.0,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark TextDataset token throughput")
    parser.add_argument("--dataset", default="Skylion007/openwebtext", help="Dataset name")
    parser.add_argument("--model", default="openai-community/gpt2", help="Model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers")
    parser.add_argument("--max-batches", type=int, default=10, help="Max batches to process")
    parser.add_argument("--max-duration", type=float, default=60.0, help="Max duration (seconds)")
    parser.add_argument("--compare-workers", action="store_true", help="Compare different worker counts")

    args = parser.parse_args()

    if args.compare_workers:
        compare_workers()
    else:
        benchmark_textdataset(
            dataset_name=args.dataset,
            model_name=args.model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_workers=args.workers,
            max_batches=args.max_batches,
            max_duration=args.max_duration,
        )
