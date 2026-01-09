#!/usr/bin/env python3
"""
Convert activation HDF5 file from float32 to float16.
Processes data in chunks to manage memory usage for large files.
"""

import os
import time
from pathlib import Path

import h5py
import numpy as np


def convert_h5_float32_to_float16(
    input_path: str,
    output_path: str = None,
    chunk_size: int = 100_000,
    accessor: str = "tensor",
):
    """
    Convert HDF5 file from float32 to float16.

    Args:
        input_path: Path to input HDF5 file (float32)
        output_path: Path to output HDF5 file (float16). If None, adds '_fp16' suffix
        chunk_size: Number of samples to process at once
        accessor: HDF5 dataset key to convert
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_fp16{input_path.suffix}"
    else:
        output_path = Path(output_path)

    print(f"Converting: {input_path}")
    print(f"Output: {output_path}")
    print(f"Chunk size: {chunk_size:,} samples")

    # Check input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Get input file info
    with h5py.File(input_path, "r") as f_in:
        if accessor not in f_in:
            raise KeyError(f"Dataset '{accessor}' not found in {input_path}")

        dataset = f_in[accessor]
        original_shape = dataset.shape
        original_dtype = dataset.dtype
        total_samples = original_shape[0]

        print(f"Input shape: {original_shape}")
        print(f"Input dtype: {original_dtype}")
        print(f"Total samples: {total_samples:,}")

        # Check if already float32
        if original_dtype != np.float32:
            print(f"Warning: Input dtype is {original_dtype}, not float32")

        # Calculate file size reduction
        original_size_gb = dataset.size * 4 / (1024**3)  # float32 = 4 bytes
        new_size_gb = dataset.size * 2 / (1024**3)  # float16 = 2 bytes
        print(f"Original size: {original_size_gb:.2f} GB")
        print(f"New size: {new_size_gb:.2f} GB")
        print(f"Size reduction: {(1 - new_size_gb / original_size_gb) * 100:.1f}%")

        # Create output file
        with h5py.File(output_path, "w") as f_out:
            # Create output dataset with float16 dtype
            output_dataset = f_out.create_dataset(
                accessor,
                shape=original_shape,
                dtype=np.float16,
            )

            # Process in chunks
            start_time = time.time()

            for start_idx in range(0, total_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, total_samples)
                end_idx - start_idx

                # Read chunk from input
                chunk_data = dataset[start_idx:end_idx]

                # Convert to float16
                chunk_fp16 = chunk_data.astype(np.float16)

                # Write to output
                output_dataset[start_idx:end_idx] = chunk_fp16

                # Progress reporting
                progress = (end_idx / total_samples) * 100
                elapsed = time.time() - start_time
                rate = end_idx / elapsed if elapsed > 0 else 0

                print(
                    f"Progress: {progress:5.1f}% ({end_idx:,}/{total_samples:,}) | "
                    f"Rate: {rate:,.0f} samples/s | "
                    f"Elapsed: {elapsed:.1f}s",
                    end="\r",
                    flush=True,
                )

            print()  # New line after progress

            # Copy any other datasets or attributes
            for key in f_in.keys():
                if key != accessor:
                    print(f"Copying additional dataset: {key}")
                    f_in.copy(key, f_out)

            # Copy file attributes
            for attr_name, attr_value in f_in.attrs.items():
                f_out.attrs[attr_name] = attr_value

            # Add conversion metadata
            f_out.attrs["converted_to_fp16"] = True
            f_out.attrs["original_dtype"] = str(original_dtype)
            f_out.attrs["conversion_time"] = time.time()

    total_time = time.time() - start_time
    final_rate = total_samples / total_time

    print("\n✅ Conversion complete!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average rate: {final_rate:,.0f} samples/s")
    print(f"Output file: {output_path}")

    # Verify the conversion
    print("\n🔍 Verification:")
    with h5py.File(output_path, "r") as f_verify:
        verify_dataset = f_verify[accessor]
        print(f"Output shape: {verify_dataset.shape}")
        print(f"Output dtype: {verify_dataset.dtype}")
        print(f"File size: {output_path.stat().st_size / (1024**3):.2f} GB")


def main():
    """Main function for command line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert HDF5 activations from float32 to float16")
    parser.add_argument("input_file", help="Path to input HDF5 file (float32)")
    parser.add_argument("-o", "--output", help="Path to output HDF5 file (default: input_fp16.h5)")
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=100_000,
        help="Chunk size for processing (default: 100,000)",
    )
    parser.add_argument(
        "-a",
        "--accessor",
        default="tensor",
        help="HDF5 dataset key (default: 'tensor')",
    )

    args = parser.parse_args()

    try:
        convert_h5_float32_to_float16(
            input_path=args.input_file,
            output_path=args.output,
            chunk_size=args.chunk_size,
            accessor=args.accessor,
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    # Default conversion for the specified file
    input_file = "/var/local/glang/activations/clt-activations-10M-shuffled.h5"

    if os.path.exists(input_file):
        print("🚀 Converting clt-activations-10M-shuffled.h5 to float16...")
        convert_h5_float32_to_float16(input_file)
    else:
        print(f"❌ File not found: {input_file}")
        print("Run with --help for usage information")
        exit(main())
