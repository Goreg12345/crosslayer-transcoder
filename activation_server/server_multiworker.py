"""
Multi-worker FastAPI server that connects to an existing shared memory buffer.
This server can run with multiple workers while sharing a single data generator.
"""

import json
import logging
import multiprocessing as mp
import os
import struct
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse

from activation_server.config import ServerConfig, get_production_config
from activation_server.shared_memory import SharedActivationBuffer

# Configure logging to prevent request logs from interfering with dashboard
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class MultiWorkerServer:
    """Multi-worker server that connects to existing shared memory."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.shared_buffer: Optional[SharedActivationBuffer] = None
        self.shm_info_file = Path("/tmp/activation_server_shm_info.json")

    def connect_to_shared_buffer(self):
        """Connect to existing shared memory buffer created by standalone generator."""
        if not self.shm_info_file.exists():
            raise RuntimeError(
                f"Shared memory info file not found: {self.shm_info_file}\n"
                "Make sure the standalone data generator is running first."
            )

        try:
            with open(self.shm_info_file, "r") as f:
                shm_info = json.load(f)

            logger.info(f"Connecting to shared memory buffer (PID: {shm_info['pid']})")

            # Recreate shared buffer with same parameters
            self.shared_buffer = SharedActivationBuffer(
                buffer_size=shm_info["buffer_size"],
                n_in_out=shm_info["n_in_out"],
                n_layers=shm_info["n_layers"],
                activation_dim=shm_info["activation_dim"],
                dtype=getattr(__import__("torch"), shm_info["dtype"].split(".")[-1]),
            )

            logger.info("Successfully connected to shared memory buffer")

        except Exception as e:
            raise RuntimeError(f"Failed to connect to shared memory: {e}")

    def disconnect(self):
        """Disconnect from shared memory (don't cleanup - that's the generator's job)."""
        if self.shared_buffer:
            # Don't call cleanup() - we don't own the shared memory
            self.shared_buffer = None
            logger.info("Disconnected from shared memory buffer")


# Global server instance
server_instance: Optional[MultiWorkerServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifecycle - connect/disconnect from shared memory."""
    global server_instance

    # Clean up any existing server instance (for hot-reload)
    if server_instance:
        server_instance.disconnect()

    config = get_production_config()
    server_instance = MultiWorkerServer(config)

    # Startup
    try:
        server_instance.connect_to_shared_buffer()
        yield
    finally:
        # Shutdown
        if server_instance:
            server_instance.disconnect()


# FastAPI app
app = FastAPI(
    title="Multi-Worker Activation Server",
    description="High-performance server for neural network activation data (multi-worker version)",
    version="1.0.0",
    lifespan=lifespan,
)


def get_buffer_stats_sync() -> Dict[str, Any]:
    """Get statistics about the shared buffer (synchronous version)."""
    global server_instance

    if not server_instance or not server_instance.shared_buffer:
        return {"error": "Buffer not connected"}

    try:
        return server_instance.shared_buffer.get_stats()
    except Exception as e:
        return {"error": f"Failed to get stats: {e}"}


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Multi-Worker Activation Server is running",
        "worker_pid": os.getpid(),
        "buffer_stats": get_buffer_stats_sync() if server_instance else None,
    }


@app.get("/activations")
async def get_activations(batch_size: int = 1) -> Dict[str, Any]:
    """
    Get activation data from the shared buffer.

    Args:
        batch_size: Number of activation samples to retrieve

    Returns:
        Dictionary containing activation tensor data and metadata
    """
    global server_instance

    if not server_instance or not server_instance.shared_buffer:
        raise HTTPException(
            status_code=503, detail="Server not connected to shared buffer"
        )

    if batch_size <= 0 or batch_size > server_instance.config.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"batch_size must be between 1 and {server_instance.config.max_batch_size}",
        )

    try:
        # Get activations from shared buffer
        activations = server_instance.shared_buffer.get_activations(batch_size)

        # Convert to serializable format
        activation_data = {
            "activations": activations.tolist(),  # Convert tensor to list for JSON
            "shape": list(activations.shape),
            "dtype": str(activations.dtype),
            "num_samples": activations.shape[0],  # Actual number returned
            "worker_pid": os.getpid(),
            "buffer_stats": get_buffer_stats_sync(),
        }

        return activation_data

    except Exception as e:
        logger.error(f"Error getting activations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving activations: {str(e)}"
        )


@app.get("/activations/tensor")
async def get_activations_tensor(batch_size: int = 1) -> Response:
    """
    Get activation data as raw tensor bytes.

    Args:
        batch_size: Number of activation samples to retrieve

    Returns:
        Raw tensor data as bytes
    """
    global server_instance

    if not server_instance or not server_instance.shared_buffer:
        raise HTTPException(
            status_code=503, detail="Server not connected to shared buffer"
        )

    try:
        # Get activations from shared buffer
        activations = server_instance.shared_buffer.get_activations(batch_size)

        # Convert tensor to bytes
        tensor_bytes = activations.numpy().tobytes()

        return Response(
            content=tensor_bytes,
            media_type="application/octet-stream",
            headers={
                "X-Tensor-Shape": ",".join(map(str, activations.shape)),
                "X-Tensor-Dtype": str(activations.dtype),
                "X-Tensor-Size": str(len(tensor_bytes)),
                "X-Worker-PID": str(os.getpid()),
            },
        )

    except Exception as e:
        logger.error(f"Error getting tensor activations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving tensor: {str(e)}"
        )


@app.get("/activations/stream")
async def stream_activations_tensor(
    batch_size: int = 5000, max_batches: int = 0
) -> Response:
    """
    Stream activation data continuously as raw tensor bytes.

    Args:
        batch_size: Number of activation samples per batch
        max_batches: Maximum number of batches to stream (0 = unlimited)

    Returns:
        Streaming tensor data as bytes with custom protocol
    """
    global server_instance

    if not server_instance or not server_instance.shared_buffer:
        raise HTTPException(
            status_code=503, detail="Server not connected to shared buffer"
        )

    if batch_size <= 0 or batch_size > server_instance.config.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"batch_size must be between 1 and {server_instance.config.max_batch_size}",
        )

    async def generate_stream():
        """Generate streaming data."""
        batches_sent = 0

        try:
            while True:
                # Check if we've reached max_batches limit
                if max_batches > 0 and batches_sent >= max_batches:
                    break

                # Get activations from shared buffer
                activations = server_instance.shared_buffer.get_activations(batch_size)

                # Convert to bytes
                tensor_bytes = activations.detach().cpu().numpy().tobytes()
                shape = activations.shape

                # Simple fixed-size header: 4 shape dimensions (4 bytes each) + data length (4 bytes) = 20 bytes total
                header = struct.pack(
                    "!IIIII", shape[0], shape[1], shape[2], shape[3], len(tensor_bytes)
                )

                # Send header + data
                yield header + tensor_bytes

                batches_sent += 1

                # Small delay to prevent overwhelming the client
                import asyncio

                await asyncio.sleep(0.001)  # 1ms delay

        except Exception as e:
            logger.error(f"Error in streaming generator: {e}")
            # Send error header (all zeros)
            error_header = struct.pack("!IIIII", 0, 0, 0, 0, 0)
            yield error_header

    return StreamingResponse(
        generate_stream(),
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Worker-PID": str(os.getpid()),
        },
    )


@app.get("/stats")
async def get_buffer_stats() -> Dict[str, Any]:
    """Get statistics about the shared buffer."""
    stats = get_buffer_stats_sync()
    stats["worker_pid"] = os.getpid()
    return stats


@app.post("/refresh")
async def refresh_buffer():
    """Manually trigger buffer refresh."""
    global server_instance

    if not server_instance or not server_instance.shared_buffer:
        raise HTTPException(status_code=503, detail="Buffer not connected")

    try:
        refreshed_count = server_instance.shared_buffer.force_refresh()
        return {"refreshed_indices": refreshed_count, "worker_pid": os.getpid()}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error refreshing buffer: {str(e)}"
        )


def run_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 1, **kwargs):
    """Run the multi-worker FastAPI server."""
    if workers > 1:
        logger.info(f"Starting server with {workers} workers")
        uvicorn.run(
            "activation_server.server_multiworker:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info",
            access_log=False,  # Disable access logging
            **kwargs,
        )
    else:
        logger.info("Starting server with single worker")
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=False,
            **kwargs,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Worker Activation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker processes"
    )

    args = parser.parse_args()

    run_server(host=args.host, port=args.port, workers=args.workers)
