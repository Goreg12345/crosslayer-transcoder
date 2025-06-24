#!/usr/bin/env python3
"""
Test script for high-throughput streaming activation data.
Benchmarks the new streaming API vs traditional request-response vs fast FileResponse.
"""

import time

from client import ActivationClient


def test_traditional_api(client, duration=30, batch_size=5000):
    """Test traditional request-response API."""
    print(f"\n=== Traditional API (batch_size={batch_size}) ===")

    start_time = time.time()
    end_time = start_time + duration
    total_samples = 0
    total_batches = 0

    while time.time() < end_time:
        try:
            t, meta = client.get_activations_tensor(batch_size)
            total_samples += t.shape[0]
            total_batches += 1
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.1)
            continue

    actual_duration = time.time() - start_time
    samples_per_sec = total_samples / actual_duration
    batches_per_sec = total_batches / actual_duration

    print(f"Duration: {actual_duration:.1f}s")
    print(f"Total samples: {total_samples:,}")
    print(f"Total batches: {total_batches}")
    print(f"Samples/sec: {samples_per_sec:.1f}")
    print(f"Batches/sec: {batches_per_sec:.2f}")

    # Calculate bandwidth
    bandwidth_mbps = (samples_per_sec * 2 * 12 * 768 * 4) / (1024 * 1024)
    print(f"Bandwidth: {bandwidth_mbps:.1f} MB/s")

    return samples_per_sec, bandwidth_mbps


def test_traditional_api_fast(client, duration=30, batch_size=5000):
    """Test traditional request-response API with FileResponse optimization."""
    print(f"\n=== Traditional API FAST (FileResponse) (batch_size={batch_size}) ===")

    start_time = time.time()
    end_time = start_time + duration
    total_samples = 0
    total_batches = 0

    while time.time() < end_time:
        try:
            t, meta = client.get_activations_tensor_fast(batch_size)
            total_samples += t.shape[0]
            total_batches += 1
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.1)
            continue

    actual_duration = time.time() - start_time
    samples_per_sec = total_samples / actual_duration
    batches_per_sec = total_batches / actual_duration

    print(f"Duration: {actual_duration:.1f}s")
    print(f"Total samples: {total_samples:,}")
    print(f"Total batches: {total_batches}")
    print(f"Samples/sec: {samples_per_sec:.1f}")
    print(f"Batches/sec: {batches_per_sec:.2f}")

    # Calculate bandwidth
    bandwidth_mbps = (samples_per_sec * 2 * 12 * 768 * 4) / (1024 * 1024)
    print(f"Bandwidth: {bandwidth_mbps:.1f} MB/s")

    return samples_per_sec, bandwidth_mbps


def test_traditional_api_optimized(client, duration=30, batch_size=5000):
    """Test traditional request-response API with zero-copy optimization."""
    print(f"\n=== Traditional API OPTIMIZED (Zero-Copy) (batch_size={batch_size}) ===")

    start_time = time.time()
    end_time = start_time + duration
    total_samples = 0
    total_batches = 0

    while time.time() < end_time:
        try:
            t, meta = client.get_activations_tensor_optimized(batch_size)
            total_samples += t.shape[0]
            total_batches += 1
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.1)
            continue

    actual_duration = time.time() - start_time
    samples_per_sec = total_samples / actual_duration
    batches_per_sec = total_batches / actual_duration

    print(f"Duration: {actual_duration:.1f}s")
    print(f"Total samples: {total_samples:,}")
    print(f"Total batches: {total_batches}")
    print(f"Samples/sec: {samples_per_sec:.1f}")
    print(f"Batches/sec: {batches_per_sec:.2f}")

    # Calculate bandwidth
    bandwidth_mbps = (samples_per_sec * 2 * 12 * 768 * 4) / (1024 * 1024)
    print(f"Bandwidth: {bandwidth_mbps:.1f} MB/s")

    return samples_per_sec, bandwidth_mbps


def test_streaming_api(client, duration=30, batch_size=5000):
    """Test streaming API."""
    print(f"\n=== Streaming API (batch_size={batch_size}) ===")

    start_time = time.time()
    end_time = start_time + duration
    total_samples = 0
    total_batches = 0

    try:
        # Use timeout to prevent hanging
        stream = client.stream_activations(batch_size=batch_size)

        for batch in stream:
            current_time = time.time()
            if current_time >= end_time:
                print(f"Time limit reached after {current_time - start_time:.1f}s")
                break

            total_samples += batch.shape[0]
            total_batches += 1

            # Print progress every 10 batches
            if total_batches % 10 == 0:
                elapsed = time.time() - start_time
                current_rate = total_samples / elapsed if elapsed > 0 else 0
                print(f"  Batch {total_batches}: {current_rate:.0f} samples/sec")

            # Safety check to prevent infinite loops
            if total_batches > 1000:  # Reasonable upper limit
                print(f"Safety limit reached at {total_batches} batches")
                break

    except Exception as e:
        print(f"Stream error: {e}")
        import traceback

        traceback.print_exc()

    actual_duration = time.time() - start_time
    samples_per_sec = total_samples / actual_duration if actual_duration > 0 else 0
    batches_per_sec = total_batches / actual_duration if actual_duration > 0 else 0

    print(f"Duration: {actual_duration:.1f}s")
    print(f"Total samples: {total_samples:,}")
    print(f"Total batches: {total_batches}")
    print(f"Samples/sec: {samples_per_sec:.1f}")
    print(f"Batches/sec: {batches_per_sec:.2f}")

    # Calculate bandwidth
    bandwidth_mbps = (samples_per_sec * 2 * 12 * 768 * 4) / (1024 * 1024)
    print(f"Bandwidth: {bandwidth_mbps:.1f} MB/s")

    # Small delay between tests to let server cleanup
    print("Waiting for cleanup...")
    time.sleep(2)

    return samples_per_sec, bandwidth_mbps


def test_streaming_api_fast(client, duration=30, batch_size=5000):
    """Test fast streaming API with memory optimization."""
    print(f"\n=== Streaming API FAST (batch_size={batch_size}) ===")

    start_time = time.time()
    end_time = start_time + duration
    total_samples = 0
    total_batches = 0

    try:
        # Use timeout to prevent hanging
        stream = client.stream_activations_fast(batch_size=batch_size)

        for batch in stream:
            current_time = time.time()
            if current_time >= end_time:
                print(f"Time limit reached after {current_time - start_time:.1f}s")
                break

            total_samples += batch.shape[0]
            total_batches += 1

            # Print progress every 10 batches
            if total_batches % 10 == 0:
                elapsed = time.time() - start_time
                current_rate = total_samples / elapsed if elapsed > 0 else 0
                print(f"  Batch {total_batches}: {current_rate:.0f} samples/sec")

            # Safety check to prevent infinite loops
            if total_batches > 1000:  # Reasonable upper limit
                print(f"Safety limit reached at {total_batches} batches")
                break

    except Exception as e:
        print(f"Fast stream error: {e}")
        import traceback

        traceback.print_exc()

    actual_duration = time.time() - start_time
    samples_per_sec = total_samples / actual_duration if actual_duration > 0 else 0
    batches_per_sec = total_batches / actual_duration if actual_duration > 0 else 0

    print(f"Duration: {actual_duration:.1f}s")
    print(f"Total samples: {total_samples:,}")
    print(f"Total batches: {total_batches}")
    print(f"Samples/sec: {samples_per_sec:.1f}")
    print(f"Batches/sec: {batches_per_sec:.2f}")

    # Calculate bandwidth
    bandwidth_mbps = (samples_per_sec * 2 * 12 * 768 * 4) / (1024 * 1024)
    print(f"Bandwidth: {bandwidth_mbps:.1f} MB/s")

    # Small delay between tests to let server cleanup
    print("Waiting for cleanup...")
    time.sleep(2)

    return samples_per_sec, bandwidth_mbps


def test_streaming_api_optimized(client, duration=30, batch_size=5000):
    """Test streaming API with zero-copy optimization."""
    print(f"\n=== Streaming API OPTIMIZED (Zero-Copy) (batch_size={batch_size}) ===")

    start_time = time.time()
    end_time = start_time + duration
    total_samples = 0
    total_batches = 0

    try:
        # Use timeout to prevent hanging
        stream = client.stream_activations_optimized(batch_size=batch_size)

        for batch in stream:
            current_time = time.time()
            if current_time >= end_time:
                print(f"Time limit reached after {current_time - start_time:.1f}s")
                break

            total_samples += batch.shape[0]
            total_batches += 1

            # Print progress every 10 batches
            if total_batches % 10 == 0:
                elapsed = time.time() - start_time
                current_rate = total_samples / elapsed if elapsed > 0 else 0
                print(f"  Batch {total_batches}: {current_rate:.0f} samples/sec")

            # Safety check to prevent infinite loops
            if total_batches > 1000:  # Reasonable upper limit
                print(f"Safety limit reached at {total_batches} batches")
                break

    except Exception as e:
        print(f"Optimized stream error: {e}")
        import traceback

        traceback.print_exc()

    actual_duration = time.time() - start_time
    samples_per_sec = total_samples / actual_duration if actual_duration > 0 else 0
    batches_per_sec = total_batches / actual_duration if actual_duration > 0 else 0

    print(f"Duration: {actual_duration:.1f}s")
    print(f"Total samples: {total_samples:,}")
    print(f"Total batches: {total_batches}")
    print(f"Samples/sec: {samples_per_sec:.1f}")
    print(f"Batches/sec: {batches_per_sec:.2f}")

    # Calculate bandwidth
    bandwidth_mbps = (samples_per_sec * 2 * 12 * 768 * 4) / (1024 * 1024)
    print(f"Bandwidth: {bandwidth_mbps:.1f} MB/s")

    # Small delay between tests to let server cleanup
    print("Waiting for cleanup...")
    time.sleep(2)

    return samples_per_sec, bandwidth_mbps


def main():
    """Run comprehensive throughput benchmark."""
    print("🚀 Comprehensive High-Throughput Activation Benchmark")
    print("=" * 70)

    client = ActivationClient()

    # Wait for server to be ready
    print("Waiting for server...")
    # Simple health check instead of wait_for_server
    try:
        stats = client.get_server_stats()
        print(f"Server ready! Buffer: {stats.get('valid_samples', 0)} samples")
    except Exception as e:
        print(f"Warning: Server might not be ready: {e}")
        print("Continuing anyway...")

    # Test configurations
    test_duration = 20  # Shorter duration for more tests

    # Test scenarios - now including truly optimized versions
    scenarios = [
        ("Traditional API", "get_activations_tensor", [1000, 5000, 10000, 20000]),
        (
            "Traditional API FAST",
            "get_activations_tensor_fast",
            [1000, 5000, 10000, 20000],
        ),
        (
            "Traditional API OPTIMIZED",
            "get_activations_tensor_optimized",
            [1000, 5000, 10000, 20000],
        ),
        ("Streaming API", "stream_activations", [5000, 10000, 20000, 50000]),
        ("Streaming API FAST", "stream_activations_fast", [5000, 10000, 20000, 50000]),
        (
            "Streaming API OPTIMIZED",
            "stream_activations_optimized",
            [5000, 10000, 20000, 50000],
        ),
    ]

    all_results = {}

    for api_name, api_method, batch_sizes in scenarios:
        print(f"\n{'='*70}")
        print(f"🔥 TESTING {api_name.upper()}")
        print(f"{'='*70}")

        api_results = {}

        for batch_size in batch_sizes:
            print(f"\n{'-'*50}")
            print(f"Batch Size: {batch_size:,}")
            print(f"{'-'*50}")

            try:
                if api_method == "get_activations_tensor":
                    rate, bandwidth = test_traditional_api(
                        client, test_duration, batch_size
                    )
                elif api_method == "get_activations_tensor_fast":
                    rate, bandwidth = test_traditional_api_fast(
                        client, test_duration, batch_size
                    )
                elif api_method == "get_activations_tensor_optimized":
                    rate, bandwidth = test_traditional_api_optimized(
                        client, test_duration, batch_size
                    )
                elif api_method == "stream_activations":
                    rate, bandwidth = test_streaming_api(
                        client, test_duration, batch_size
                    )
                elif api_method == "stream_activations_fast":
                    rate, bandwidth = test_streaming_api_fast(
                        client, test_duration, batch_size
                    )
                elif api_method == "stream_activations_optimized":
                    rate, bandwidth = test_streaming_api_optimized(
                        client, test_duration, batch_size
                    )

                api_results[batch_size] = {"rate": rate, "bandwidth": bandwidth}

            except Exception as e:
                print(f"❌ Error testing {api_name} with batch size {batch_size}: {e}")
                api_results[batch_size] = {"rate": 0, "bandwidth": 0}

        all_results[api_name] = api_results

    # Final comparison
    print(f"\n{'='*70}")
    print("🎯 FINAL PERFORMANCE COMPARISON")
    print(f"{'='*70}")

    # Find the best performing configuration overall
    best_overall = {"api": "", "batch_size": 0, "rate": 0, "bandwidth": 0}

    for api_name, api_results in all_results.items():
        print(f"\n📊 {api_name}:")
        if api_results:  # Check if we have results
            best_for_api = max(api_results.items(), key=lambda x: x[1]["rate"])
            best_batch_size, best_result = best_for_api

            print(
                f"   Best: {best_result['rate']:>8.0f} samples/sec @ batch size {best_batch_size:,}"
            )
            print(f"   Bandwidth: {best_result['bandwidth']:>6.1f} MB/s")

            if best_result["rate"] > best_overall["rate"]:
                best_overall = {
                    "api": api_name,
                    "batch_size": best_batch_size,
                    "rate": best_result["rate"],
                    "bandwidth": best_result["bandwidth"],
                }

            # Show all results for this API
            for batch_size, result in sorted(api_results.items()):
                print(
                    f"     {batch_size:>6,}: {result['rate']:>8.0f}/s ({result['bandwidth']:>6.1f} MB/s)"
                )
        else:
            print("   No results (all tests failed)")

    # Overall winner
    print(f"\n{'='*70}")
    print("🏆 OVERALL WINNER")
    print(f"{'='*70}")
    print(f"API: {best_overall['api']}")
    print(f"Batch Size: {best_overall['batch_size']:,}")
    print(f"Rate: {best_overall['rate']:,.0f} samples/sec")
    print(f"Bandwidth: {best_overall['bandwidth']:,.1f} MB/s")

    # Performance comparison between traditional and fast versions
    print(f"\n{'='*70}")
    print("⚡ SPEED IMPROVEMENT ANALYSIS")
    print(f"{'='*70}")

    if "Traditional API" in all_results and "Traditional API FAST" in all_results:
        print("\n📈 Traditional API vs Traditional API FAST:")
        trad_results = all_results["Traditional API"]
        fast_results = all_results["Traditional API FAST"]

        for batch_size in sorted(set(trad_results.keys()) & set(fast_results.keys())):
            if trad_results[batch_size]["rate"] > 0:
                improvement = (
                    fast_results[batch_size]["rate"] / trad_results[batch_size]["rate"]
                    - 1
                ) * 100
                print(
                    f"   Batch {batch_size:>6,}: {improvement:+6.1f}% faster with FileResponse"
                )

    if "Streaming API" in all_results and "Streaming API FAST" in all_results:
        print("\n📈 Streaming API vs Streaming API FAST:")
        stream_results = all_results["Streaming API"]
        stream_fast_results = all_results["Streaming API FAST"]

        for batch_size in sorted(
            set(stream_results.keys()) & set(stream_fast_results.keys())
        ):
            if stream_results[batch_size]["rate"] > 0:
                improvement = (
                    stream_fast_results[batch_size]["rate"]
                    / stream_results[batch_size]["rate"]
                    - 1
                ) * 100
                print(
                    f"   Batch {batch_size:>6,}: {improvement:+6.1f}% faster with optimization"
                )

    # Check target achievement
    print(f"\n{'='*70}")
    print("🎯 TARGET ACHIEVEMENT")
    print(f"{'='*70}")
    target_rates = [30000, 50000]
    for target in target_rates:
        if best_overall["rate"] >= target:
            print(f"✅ Target {target:,}/s ACHIEVED!")
        else:
            shortage = target - best_overall["rate"]
            print(f"❌ Target {target:,}/s missed by {shortage:,.0f} samples/sec")

    # Calculate theoretical max
    print(f"\n📈 THEORETICAL ANALYSIS:")
    print(f"Current best: {best_overall['bandwidth']:,.1f} MB/s")

    # Network bandwidth often caps at 1-10 Gbps
    theoretical_max_10gig = 10 * 1000 / 8  # 10 Gbps in MB/s
    theoretical_samples_10gig = theoretical_max_10gig * 1024 * 1024 / (2 * 12 * 768 * 4)

    print(f"10 Gbps network limit: ~{theoretical_samples_10gig:,.0f} samples/sec")
    print(
        f"Your performance: {best_overall['rate']/theoretical_samples_10gig*100:.1f}% of 10 Gbps limit"
    )


if __name__ == "__main__":
    main()
