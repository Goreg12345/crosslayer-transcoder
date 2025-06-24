#!/usr/bin/env python3
"""
Example script showing how to run and interact with the activation server.
"""

import signal
import subprocess
import sys
import time
from multiprocessing import Process

from .client import ActivationClient
from .config import get_test_config


def run_server_background():
    """Run the server in the background using subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "activation_server.main",
        "--config",
        "test",
        "--buffer-size",
        "100",  # Very small for quick testing
        "--port",
        "8001",  # Use different port to avoid conflicts
    ]

    print("🚀 Starting server in background...")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Give server time to start
    time.sleep(5)

    return process


def demo_client():
    """Demonstrate client usage."""
    print("🧪 Activation Server Demo")

    server_process = None

    try:
        # Start server
        server_process = run_server_background()

        # Create client
        client = ActivationClient("http://localhost:8001")

        print("🏥 Checking server health...")
        health = client.health_check()
        print(f"   Status: {health['status']}")

        print("📊 Getting server stats...")
        stats = client.get_stats()
        print(f"   Buffer size: {stats.get('buffer_size', 'N/A'):,}")
        print(f"   Activation dim: {stats.get('activation_dim', 'N/A'):,}")
        print(f"   Memory usage: {stats.get('total_memory_gb', 0):.3f} GB")

        print("⏳ Waiting for data generation...")
        print(
            "   (This may take a while as the model needs to load and generate real activations)"
        )

        if client.wait_for_data(min_samples=5, timeout=120, check_interval=5):
            print("✅ Data is ready!")

            # Get updated stats
            stats = client.get_stats()
            print(f"   Valid samples: {stats.get('valid_samples', 0):,}")
            print(f"   Buffer utilization: {stats.get('valid_percentage', 0):.1f}%")

            print("📦 Getting activations as JSON...")
            json_data = client.get_activations(num_samples=2)
            print(f"   Retrieved {len(json_data['activations'])} samples")
            print(f"   Shape: {json_data['shape']}")
            print(f"   Data type: {json_data['dtype']}")

            print("🔢 Getting activations as tensor...")
            tensor, metadata = client.get_activations_tensor(num_samples=1)
            print(f"   Tensor shape: {tensor.shape}")
            print(f"   Tensor dtype: {tensor.dtype}")
            print(f"   Mean activation: {tensor.mean().item():.4f}")

            print("✨ Demo completed successfully!")

        else:
            print("❌ Timeout waiting for data generation")
            print("   This is normal for first run as models need to download")

    except Exception as e:
        print(f"❌ Demo error: {e}")
        if server_process:
            # Check server logs
            stdout, stderr = server_process.communicate(timeout=1)
            if stderr:
                print(f"Server error: {stderr}")

    finally:
        print("🛑 Stopping server...")
        if server_process:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()


def quick_config_test():
    """Quick test of configuration and imports."""
    print("🧪 Running quick configuration test...")

    try:
        from .config import get_production_config, get_test_config
        from .shared_memory import SharedActivationBuffer

        # Test config loading
        test_config = get_test_config()
        print(
            f"✅ Test config: {test_config.buffer_size} samples, {test_config.activation_dim} dims"
        )
        print(f"   Memory estimate: {test_config.get_memory_estimate_gb():.3f} GB")

        prod_config = get_production_config()
        print(
            f"✅ Production config: {prod_config.buffer_size:,} samples, {prod_config.activation_dim} dims"
        )
        print(f"   Memory estimate: {prod_config.get_memory_estimate_gb():.1f} GB")

        # Test shared memory creation (small)
        small_config = get_test_config()
        small_config.buffer_size = 10  # Very small

        print("🔧 Testing shared memory creation...")
        buffer = SharedActivationBuffer(
            buffer_size=small_config.buffer_size,
            n_in_out=small_config.n_in_out,
            n_layers=small_config.n_layers,
            activation_dim=small_config.activation_dim,
            dtype=small_config.dtype,
        )

        stats = buffer.get_stats()
        print(f"✅ Shared buffer created: {stats['buffer_size']} samples")
        print(f"   Memory: {stats['total_memory_gb']:.6f} GB")

        buffer.cleanup()
        print("✅ All tests passed!")

    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main function."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--config-test":
            quick_config_test()
        elif sys.argv[1] == "--demo":
            demo_client()
        else:
            print("Usage:")
            print("  python -m activation_server.run_example --config-test")
            print("  python -m activation_server.run_example --demo")
    else:
        print("🚀 Activation Server Example")
        print()
        print("Available commands:")
        print("  --config-test  : Test configuration and shared memory")
        print("  --demo        : Run full server demo (requires model download)")
        print()
        print("Choose an option:")
        print("1. Quick config test (fast)")
        print("2. Full demo (slow, downloads models)")

        try:
            choice = input("Enter choice (1 or 2): ").strip()
        except KeyboardInterrupt:
            print("\nBye!")
            return

        if choice == "1":
            quick_config_test()
        elif choice == "2":
            demo_client()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
