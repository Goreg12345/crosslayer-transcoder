#!/usr/bin/env python3
"""
Comprehensive DataLoader Benchmark

Tests dataloader performance with different worker counts and storage types.
Creates a nice visualization of the results.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from buffer import DiscBuffer

# Set up nice plotting style
plt.style.use("default")
sns.set_palette("husl")


def test_dataloader_performance(
    data_path, num_workers, batch_size=1000, num_batches=100
):
    """Test dataloader performance for given configuration"""

    buffer = DiscBuffer(data_path, "tensor")

    loader = torch.utils.data.DataLoader(
        buffer,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        batch_size=batch_size,
        shuffle=True,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
    )

    print(f"  Testing {num_workers} workers... ", end="", flush=True)

    # Warmup - less for 0 workers, more for higher worker counts
    if num_workers == 0:
        warmup_batches = 2
    else:
        warmup_batches = max(5, min(10, num_workers) * 3)

    loader_iter = iter(loader)
    for i in range(warmup_batches):
        next(loader_iter)

    # Benchmark
    start_time = time.perf_counter()
    batch_count = 0

    for i in range(num_batches):
        next(loader_iter)
        batch_count += 1

    end_time = time.perf_counter()
    total_time = end_time - start_time

    batches_per_second = batch_count / total_time if total_time > 0 else 0
    samples_per_second = batches_per_second * batch_size

    print(f"{batches_per_second:.1f} batches/s ({samples_per_second:.0f} samples/s)")

    # Clean up
    del loader, buffer

    return batches_per_second, samples_per_second


def copy_file_to_ssd():
    """Copy the dataset to SSD and return the path"""
    nfs_path = "/var/local/glang/activations/clt-activations.h5"

    if not os.path.exists(nfs_path):
        print(f"Warning: {nfs_path} not found, using 10M dataset")
        nfs_path = "/var/local/glang/activations/clt-activations-10M.h5"

    # Create temp directory on SSD
    temp_dir = tempfile.mkdtemp(prefix="dataloader_benchmark_")
    ssd_path = os.path.join(temp_dir, "clt-activations.h5")

    print(f"Copying {nfs_path} to SSD...")
    print(f"Source: {nfs_path}")
    print(f"Destination: {ssd_path}")

    # Get file size
    file_size_gb = os.path.getsize(nfs_path) / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB")

    # Copy with progress indication
    start_time = time.time()
    shutil.copy2(nfs_path, ssd_path)
    copy_time = time.time() - start_time

    copy_speed = file_size_gb / copy_time
    print(f"Copy completed in {copy_time:.1f}s ({copy_speed:.1f} GB/s)")

    return ssd_path, temp_dir


def create_beautiful_plot(results):
    """Create a beautiful plot of the benchmark results"""

    # Set up the figure with custom styling
    plt.rcParams.update(
        {
            "font.size": 12,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "axes.linewidth": 1.2,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.major.size": 5,
            "ytick.minor.size": 3,
            "grid.alpha": 0.3,
            "legend.frameon": False,
        }
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.patch.set_facecolor("white")

    # Extract data
    worker_counts = results["worker_counts"]
    nfs_batches = results["nfs_batches_per_sec"]
    ssd_batches = results["ssd_batches_per_sec"]
    nfs_samples = results["nfs_samples_per_sec"]
    ssd_samples = results["ssd_samples_per_sec"]

    # Colors
    nfs_color = "#e74c3c"  # Red
    ssd_color = "#3498db"  # Blue

    # Plot 1: Batches per second
    ax1.plot(
        worker_counts,
        nfs_batches,
        "o-",
        color=nfs_color,
        linewidth=3,
        markersize=8,
        label="NFS Storage",
        markerfacecolor="white",
        markeredgewidth=2,
        markeredgecolor=nfs_color,
    )
    ax1.plot(
        worker_counts,
        ssd_batches,
        "s-",
        color=ssd_color,
        linewidth=3,
        markersize=8,
        label="SSD Storage",
        markerfacecolor="white",
        markeredgewidth=2,
        markeredgecolor=ssd_color,
    )

    ax1.set_xlabel("Number of DataLoader Workers", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Batches per Second", fontsize=14, fontweight="bold")
    ax1.set_title(
        "DataLoader Performance: Batches per Second",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax1.legend(loc="upper left", fontsize=12)
    ax1.set_xscale("symlog", linthresh=1)
    ax1.set_xticks(worker_counts)
    ax1.set_xticklabels([str(w) for w in worker_counts])

    # Add value annotations
    for i, (wc, nfs_val, ssd_val) in enumerate(
        zip(worker_counts, nfs_batches, ssd_batches)
    ):
        if i % 2 == 0:  # Annotate every other point to avoid crowding
            ax1.annotate(
                f"{nfs_val:.1f}",
                (wc, nfs_val),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                color=nfs_color,
            )
            ax1.annotate(
                f"{ssd_val:.1f}",
                (wc, ssd_val),
                textcoords="offset points",
                xytext=(0, -15),
                ha="center",
                fontsize=9,
                color=ssd_color,
            )

    # Plot 2: Samples per second (thousands)
    nfs_samples_k = [s / 1000 for s in nfs_samples]
    ssd_samples_k = [s / 1000 for s in ssd_samples]

    ax2.plot(
        worker_counts,
        nfs_samples_k,
        "o-",
        color=nfs_color,
        linewidth=3,
        markersize=8,
        label="NFS Storage",
        markerfacecolor="white",
        markeredgewidth=2,
        markeredgecolor=nfs_color,
    )
    ax2.plot(
        worker_counts,
        ssd_samples_k,
        "s-",
        color=ssd_color,
        linewidth=3,
        markersize=8,
        label="SSD Storage",
        markerfacecolor="white",
        markeredgewidth=2,
        markeredgecolor=ssd_color,
    )

    ax2.set_xlabel("Number of DataLoader Workers", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Samples per Second (thousands)", fontsize=14, fontweight="bold")
    ax2.set_title(
        "DataLoader Performance: Samples per Second",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax2.legend(loc="upper left", fontsize=12)
    ax2.set_xscale("symlog", linthresh=1)
    ax2.set_xticks(worker_counts)
    ax2.set_xticklabels([str(w) for w in worker_counts])

    # Add horizontal line for training speed reference
    training_speed = 5.6  # 4-GPU training at 1.4 it/s * 4000 samples = 5600 samples/s
    ax2.axhline(
        y=training_speed,
        color="#95a5a6",
        linestyle="--",
        linewidth=2,
        label=f"4-GPU Training Speed ({training_speed:.1f}k samples/s)",
    )
    ax2.legend(loc="upper left", fontsize=12)

    # Improve layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)

    # Add summary text box
    max_nfs = max(nfs_samples) if nfs_samples else 0
    max_ssd = max(ssd_samples) if ssd_samples else 0
    best_nfs_workers = worker_counts[nfs_samples.index(max_nfs)] if max_nfs > 0 else 0
    best_ssd_workers = worker_counts[ssd_samples.index(max_ssd)] if max_ssd > 0 else 0

    speedup_text = f"{max_ssd/max_nfs:.1f}x faster" if max_nfs > 0 else "N/A"
    summary_text = f"""Performance Summary:
• Best NFS: {max_nfs:.0f} samples/s ({best_nfs_workers} workers)
• Best SSD: {max_ssd:.0f} samples/s ({best_ssd_workers} workers)
• SSD Speedup: {speedup_text}
• Training Bottleneck: {'DataLoader' if max_ssd < 5600 else 'GPU Computation'}"""

    fig.text(
        0.02,
        0.02,
        summary_text,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        verticalalignment="bottom",
    )

    return fig


def main():
    """Run comprehensive dataloader benchmark"""

    print("=== Comprehensive DataLoader Benchmark ===\n")

    # Test configurations
    worker_counts = [1, 2, 4, 8, 16, 32, 64]
    batch_size = 1000

    # Original NFS path
    nfs_path = "/var/local/glang/activations/clt-activations.h5"
    if not os.path.exists(nfs_path):
        nfs_path = "/var/local/glang/activations/clt-activations-10M.h5"

    print(f"Testing with batch_size={batch_size}, 5x workers batches per test\n")

    # Test NFS performance
    print("1. Testing NFS Storage Performance:")
    nfs_results = {"batches_per_sec": [], "samples_per_sec": []}

    for workers in worker_counts:
        num_batches = 5 * workers  # 5x number of workers as requested
        batches_ps, samples_ps = test_dataloader_performance(
            nfs_path, workers, batch_size, num_batches
        )
        nfs_results["batches_per_sec"].append(batches_ps)
        nfs_results["samples_per_sec"].append(samples_ps)

    print(
        f"\nBest NFS performance: {max(nfs_results['samples_per_sec']):.0f} samples/s"
    )

    # Copy to SSD and test
    print(f"\n2. Copying dataset to SSD...")
    ssd_path, temp_dir = copy_file_to_ssd()

    print(f"\n3. Testing SSD Storage Performance:")
    ssd_results = {"batches_per_sec": [], "samples_per_sec": []}

    for workers in worker_counts:
        num_batches = 5 * workers  # 5x number of workers as requested
        batches_ps, samples_ps = test_dataloader_performance(
            ssd_path, workers, batch_size, num_batches
        )
        ssd_results["batches_per_sec"].append(batches_ps)
        ssd_results["samples_per_sec"].append(samples_ps)

    print(
        f"\nBest SSD performance: {max(ssd_results['samples_per_sec']):.0f} samples/s"
    )

    # Compile results
    results = {
        "worker_counts": worker_counts,
        "nfs_batches_per_sec": nfs_results["batches_per_sec"],
        "ssd_batches_per_sec": ssd_results["batches_per_sec"],
        "nfs_samples_per_sec": nfs_results["samples_per_sec"],
        "ssd_samples_per_sec": ssd_results["samples_per_sec"],
    }

    # Create beautiful plot
    print(f"\n4. Creating visualization...")
    fig = create_beautiful_plot(results)

    # Save plot
    output_path = "dataloader_benchmark_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Plot saved as: {output_path}")

    # Clean up
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temporary files")

    # Show summary
    print(f"\n=== Summary ===")
    max_nfs = max(results["nfs_samples_per_sec"])
    max_ssd = max(results["ssd_samples_per_sec"])
    speedup = max_ssd / max_nfs if max_nfs > 0 else 0

    print(f"NFS peak performance: {max_nfs:.0f} samples/s")
    print(f"SSD peak performance: {max_ssd:.0f} samples/s")
    print(f"SSD speedup: {speedup:.1f}x")
    print(f"Training speed (4-GPU): ~5,600 samples/s")

    if max_ssd > 10000:
        print("✓ DataLoader is NOT the bottleneck - GPU training is slower")
    else:
        print("⚠ DataLoader might be limiting training speed")


if __name__ == "__main__":
    main()
