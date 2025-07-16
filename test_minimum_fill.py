#!/usr/bin/env python3
"""
Test script to verify minimum fill threshold functionality in SharedActivationBuffer.
"""

import threading
import time

import torch

from data.shared_memory import SharedActivationBuffer


def test_minimum_fill_threshold():
    """Test that get_activations respects minimum fill threshold."""

    # Create buffer with 20% minimum fill threshold
    buffer = SharedActivationBuffer(
        buffer_size=100,
        n_in_out=2,
        n_layers=3,
        activation_dim=4,
        dtype=torch.float32,
        minimum_fill_threshold=0.2,  # 20% minimum
    )

    print("=== Testing Minimum Fill Threshold ===")
    print(f"Buffer size: {buffer.buffer_size}")
    print(f"Minimum fill threshold: {buffer.minimum_fill_threshold * 100}%")
    print(f"Minimum samples needed: {int(buffer.buffer_size * buffer.minimum_fill_threshold)}")

    # Try to get activations when buffer is empty (should timeout quickly)
    print("\n1. Testing empty buffer (should timeout)...")
    try:
        start_time = time.time()
        activations = buffer.get_activations(batch_size=5, timeout=2.0)
        print("ERROR: Should not have returned activations!")
    except TimeoutError:
        elapsed = time.time() - start_time
        print(f"✓ Correctly timed out after {elapsed:.1f}s")

    # Add some activations but not enough to meet threshold
    print("\n2. Testing below threshold (should timeout)...")
    # Need 20 samples for 20% of 100, so add only 10 (10%)
    indices = torch.arange(10)
    fake_activations = torch.randn(10, 2, 3, 4, dtype=torch.float32)
    buffer.set_activations(indices, fake_activations)

    stats = buffer.get_stats()
    print(f"Valid samples: {stats['valid_samples']}")
    print(f"Fill percentage: {stats['valid_percentage']:.1f}%")
    print(f"Above threshold: {stats['above_minimum_threshold']}")

    try:
        start_time = time.time()
        activations = buffer.get_activations(batch_size=5, timeout=2.0)
        print("ERROR: Should not have returned activations!")
    except TimeoutError:
        elapsed = time.time() - start_time
        print(f"✓ Correctly timed out after {elapsed:.1f}s")

    # Add enough activations to meet threshold
    print("\n3. Testing above threshold (should succeed)...")
    # Add more samples to reach 25% (25 total samples)
    additional_indices = torch.arange(10, 25)
    additional_activations = torch.randn(15, 2, 3, 4, dtype=torch.float32)
    buffer.set_activations(additional_indices, additional_activations)

    stats = buffer.get_stats()
    print(f"Valid samples: {stats['valid_samples']}")
    print(f"Fill percentage: {stats['valid_percentage']:.1f}%")
    print(f"Above threshold: {stats['above_minimum_threshold']}")

    try:
        activations = buffer.get_activations(batch_size=5, timeout=2.0)
        print(f"✓ Successfully got activations: {activations.shape}")

        # Check that we got the right number of samples
        assert activations.shape == (5, 2, 3, 4), f"Wrong shape: {activations.shape}"
        print("✓ Activations have correct shape")

    except TimeoutError:
        print("ERROR: Should have returned activations!")

    # Test with batch size larger than available samples
    print("\n4. Testing batch size larger than available samples...")
    try:
        # We should have 20 valid samples left (25 - 5 consumed)
        activations = buffer.get_activations(batch_size=25, timeout=2.0)
        print("ERROR: Should not have returned activations!")
    except TimeoutError:
        print("✓ Correctly timed out when batch size > available samples")

    # Cleanup
    buffer.cleanup()
    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    test_minimum_fill_threshold()
