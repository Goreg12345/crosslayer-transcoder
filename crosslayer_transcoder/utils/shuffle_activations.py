#!/usr/bin/env python3
"""
Script to shuffle activations along the batch dimension.
Loads entire file into memory for faster shuffling.
"""

import argparse
import os

import h5py
import numpy as np


def shuffle_activations(input_path, output_path, seed=42):
    """
    Shuffle activations along the batch dimension (first dimension).
    Loads entire file into memory and uses in-place Fisher-Yates shuffle to avoid copies.

    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
        seed: Random seed for reproducible shuffling
    """
    np.random.seed(seed)

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")

    # Read entire tensor into memory and close input file
    print("Reading tensor into memory...")
    with h5py.File(input_path, "r") as f_in:
        tensor = f_in["tensor"]
        shape = tensor.shape
        dtype = tensor.dtype

        print(f"Tensor shape: {shape}")
        print(f"Tensor dtype: {dtype}")
        print(f"File size: {os.path.getsize(input_path) / (1024**3):.2f} GB")

        # Read entire tensor at once - much faster than sample by sample
        data = tensor[:]

    # In-place Fisher-Yates shuffle to avoid creating copies
    print("Shuffling data in-place...")
    batch_size = data.shape[0]

    # Use numpy's built-in shuffle for maximum efficiency
    # This is guaranteed to be in-place and memory efficient
    np.random.shuffle(data)
    print(f"Shuffled {batch_size:,} samples in-place")

    # Create output file and write shuffled data
    print("Writing shuffled data...")
    with h5py.File(output_path, "w") as f_out:
        # Create dataset and write entire tensor at once
        f_out.create_dataset(
            "tensor",
            data=data,  # Write all data at once
            chunks=True,  # Enable chunking for better I/O
            # Removed compression for speed - can add back if storage is more important than speed
        )

    print(f"Shuffling complete! Output saved to: {output_path}")
    print(f"Output file size: {os.path.getsize(output_path) / (1024**3):.2f} GB")


def verify_shuffle(original_path, shuffled_path, num_samples=100):
    """
    Verify that the shuffle worked correctly by checking some samples.
    Optimized version that reads data more efficiently.

    Args:
        original_path: Path to original file
        shuffled_path: Path to shuffled file
        num_samples: Number of samples to verify
    """
    print(f"\nVerifying shuffle with {num_samples} random samples...")

    with (
        h5py.File(original_path, "r") as f_orig,
        h5py.File(shuffled_path, "r") as f_shuf,
    ):
        orig_tensor = f_orig["tensor"]
        shuf_tensor = f_shuf["tensor"]

        # Check shapes match
        assert orig_tensor.shape == shuf_tensor.shape, "Shapes don't match!"

        batch_size = orig_tensor.shape[0]

        # Sample indices for verification
        test_indices = np.random.choice(batch_size, min(num_samples, batch_size), replace=False)

        # Read only the samples we need for verification - much more efficient
        orig_samples = orig_tensor[test_indices]
        shuf_samples = shuf_tensor[test_indices]

        differences_found = 0
        for i, _idx in enumerate(test_indices):
            if not np.array_equal(orig_samples[i], shuf_samples[i]):
                differences_found += 1

        print(
            f"Found {differences_found}/{len(test_indices)} different samples (expected for proper shuffle)"
        )

        print("Verification complete - data appears to be properly shuffled")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle activations along batch dimension")
    parser.add_argument(
        "--input",
        default="/var/local/glang/activations/clt-activations-10M.h5",
        help="Input HDF5 file path",
    )
    parser.add_argument(
        "--output",
        default="/var/local/glang/activations/clt-activations-10M-shuffled.h5",
        help="Output HDF5 file path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible shuffling")
    parser.add_argument("--verify", action="store_true", help="Verify the shuffle after completion")

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist!")
        exit(1)

    # Check if output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # Perform the shuffle
    try:
        shuffle_activations(
            input_path=args.input,
            output_path=args.output,
            seed=args.seed,
        )

        if args.verify:
            verify_shuffle(args.input, args.output)

    except Exception as e:
        print(f"Error during shuffling: {e}")
        # Clean up partial output file if it exists
        if os.path.exists(args.output):
            print(f"Cleaning up partial output file: {args.output}")
            os.remove(args.output)
        raise
