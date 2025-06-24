"""
Client utilities for interacting with the activation server.
"""

import io
import json
import logging
import struct
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import requests
import torch

logger = logging.getLogger(__name__)


class ActivationClient:
    """Client for accessing activation data from the server."""

    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip("/")
        self.session = requests.Session()

    def get_activations(self, batch_size: int = 1000) -> Dict:
        """
        Get a batch of activation data as JSON.

        Args:
            batch_size: Number of samples to retrieve

        Returns:
            Dictionary with activations list and metadata
        """
        response = self.session.get(
            f"{self.server_url}/activations", params={"batch_size": batch_size}
        )
        response.raise_for_status()
        return response.json()

    def get_activations_tensor(
        self, batch_size: int = 1000
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Get a batch of activation data as a tensor.

        Args:
            batch_size: Number of samples to retrieve

        Returns:
            Tuple of (activations_tensor, metadata)
        """
        response = self.session.get(
            f"{self.server_url}/activations/tensor", params={"batch_size": batch_size}
        )
        response.raise_for_status()

        # Get shape from headers
        shape_str = response.headers.get("X-Tensor-Shape")
        if not shape_str:
            raise ValueError("Missing tensor shape in response headers")

        shape = tuple(map(int, shape_str.split(",")))

        # Convert bytes to tensor
        tensor_data = np.frombuffer(response.content, dtype=np.float32)
        tensor = torch.from_numpy(tensor_data.reshape(shape).copy())

        # Get metadata from headers
        metadata = {
            "shape": shape,
            "dtype": response.headers.get("X-Tensor-Dtype", "float32"),
            "device": response.headers.get("X-Tensor-Device", "cpu"),
        }

        return tensor, metadata

    def stream_activations(
        self, batch_size: int = 5000, max_batches: int = 0
    ) -> Iterator[torch.Tensor]:
        """
        Stream activation data continuously for high throughput.
        Server sends raw tensor bytes directly.

        Args:
            batch_size: Number of samples per batch
            max_batches: Maximum number of batches to stream (0 = unlimited)

        Yields:
            torch.Tensor: Activation batches
        """
        params = {"batch_size": batch_size}
        if max_batches > 0:
            params["max_batches"] = max_batches

        response = self.session.get(
            f"{self.server_url}/activations/stream", params=params, stream=True
        )
        response.raise_for_status()

        # Get expected tensor shape from a test request
        try:
            test_tensor, _ = self.get_activations_tensor(batch_size=1)
            expected_shape = (batch_size,) + test_tensor.shape[1:]
            expected_bytes_per_batch = (
                batch_size * test_tensor.numel() * test_tensor.element_size()
            )
        except Exception as e:
            logger.error(f"Could not determine expected tensor shape: {e}")
            response.close()  # Clean up response
            raise RuntimeError(f"Cannot stream without knowing tensor shape: {e}")

        buffer = b""

        try:
            for chunk in response.iter_content(chunk_size=expected_bytes_per_batch):
                if not chunk:
                    break

                buffer += chunk

                # Process complete batches from buffer
                while len(buffer) >= expected_bytes_per_batch:
                    # Extract one batch worth of bytes
                    batch_bytes = buffer[:expected_bytes_per_batch]

                    # Convert to tensor (copy to make it writable)
                    tensor = (
                        torch.frombuffer(batch_bytes, dtype=torch.float32)
                        .reshape(expected_shape)
                        .clone()
                    )

                    yield tensor

                    # Remove processed batch from buffer
                    buffer = buffer[expected_bytes_per_batch:]

        except Exception as e:
            logger.error(f"Error in stream processing: {e}")
            raise
        finally:
            # Always close the response to clean up the connection
            response.close()

    def get_server_stats(self) -> Dict:
        """Get server statistics."""
        response = requests.get(f"{self.server_url}/stats")
        response.raise_for_status()
        return response.json()

    def refresh_buffer(self) -> Dict[str, Any]:
        """Manually trigger buffer refresh."""
        response = requests.post(f"{self.server_url}/refresh")
        response.raise_for_status()
        return response.json()

    def refill_from_file(self) -> Dict[str, Any]:
        """Manually trigger buffer refill from initialization file."""
        response = requests.post(f"{self.server_url}/refill")
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict[str, Any]:
        """Check server health."""
        response = requests.get(f"{self.server_url}/")
        response.raise_for_status()
        return response.json()

    def wait_for_data(
        self, min_samples: int = 100, timeout: int = 60, check_interval: float = 1.0
    ) -> bool:
        """
        Wait for the server to have sufficient data available.

        Args:
            min_samples: Minimum number of valid samples to wait for
            timeout: Maximum time to wait in seconds
            check_interval: How often to check in seconds

        Returns:
            True if sufficient data is available, False if timeout
        """
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                stats = self.get_server_stats()
                valid_samples = stats.get("valid_samples", 0)

                if valid_samples >= min_samples:
                    return True

                print(
                    f"Waiting for data... ({valid_samples}/{min_samples} samples ready)"
                )
                time.sleep(check_interval)

            except Exception as e:
                print(f"Error checking server: {e}")
                time.sleep(check_interval)

        return False


def test_client():
    """Test function to demonstrate client usage."""
    client = ActivationClient()

    print("Testing activation server client...")

    try:
        # Health check
        health = client.health_check()
        print(f"Server health: {health}")

        # Get stats
        stats = client.get_server_stats()
        print(f"Server stats: {json.dumps(stats, indent=2)}")

        # Try refilling from file if available
        try:
            refill_result = client.refill_from_file()
            print(f"Refill result: {json.dumps(refill_result, indent=2)}")
        except Exception as e:
            print(f"Refill failed (normal if no init file): {e}")

        # Wait for some data
        print("Waiting for data to be generated...")
        if client.wait_for_data(min_samples=10, timeout=30):
            print("Data is ready!")

            # Get activations as JSON
            activations_json = client.get_activations(batch_size=5)
            print(f"Got {len(activations_json['activations'])} samples")
            print(f"Shape: {activations_json['shape']}")

            # Get activations as tensor
            tensor, metadata = client.get_activations_tensor(batch_size=3)
            print(f"Tensor shape: {tensor.shape}")
            print(f"Tensor dtype: {tensor.dtype}")
            print(f"Tensor mean: {tensor.mean().item():.4f}")

        else:
            print("Timeout waiting for data")

    except Exception as e:
        print(f"Client test error: {e}")


if __name__ == "__main__":
    test_client()
