"""
FastAPI server for serving neural network activations from shared memory buffer.
"""

import asyncio
import logging
import multiprocessing as mp
import struct
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from activation_server.config import ServerConfig
from activation_server.data_generator import DataGeneratorProcess
from activation_server.shared_memory import SharedActivationBuffer

logger = logging.getLogger(__name__)


class ActivationServer:
    """Main activation server class managing shared memory and processes."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.shared_buffer: Optional[SharedActivationBuffer] = None
        self.data_generator_process: Optional[mp.Process] = None
        self.running = False

    def start_background_processes(self):
        """Start the data generator process and initialize shared memory."""
        if self.running:
            logger.warning("Background processes already running")
            return

        # Clean up any existing processes first
        self.stop_background_processes()

        logger.info("Initializing shared memory buffer...")
        self.shared_buffer = SharedActivationBuffer(
            buffer_size=self.config.buffer_size,
            n_in_out=self.config.n_in_out,
            n_layers=self.config.n_layers,
            activation_dim=self.config.activation_dim,
            dtype=self.config.dtype,
        )

        logger.info("Starting data generator process...")
        self.data_generator_process = DataGeneratorProcess(
            shared_buffer=self.shared_buffer, config=self.config
        )
        self.data_generator_process.start()

        self.running = True
        logger.info("Background processes started successfully")

    def stop_background_processes(self):
        """Stop the data generator process and cleanup shared memory."""
        if not self.running:
            return

        logger.info("Stopping background processes...")

        if self.data_generator_process and self.data_generator_process.is_alive():
            self.data_generator_process.terminate()
            self.data_generator_process.join(timeout=5)
            if self.data_generator_process.is_alive():
                logger.warning("Force killing data generator process")
                self.data_generator_process.kill()

        if self.shared_buffer:
            self.shared_buffer.cleanup()

        self.running = False
        logger.info("Background processes stopped")


# Global server instance
server_instance: Optional[ActivationServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifecycle - start/stop background processes."""
    global server_instance

    # Clean up any existing server instance first (for hot-reload)
    if server_instance:
        server_instance.stop_background_processes()

    # config = ServerConfig()
    from activation_server.config import get_production_config, get_test_config

    config = get_production_config()
    server_instance = ActivationServer(config)

    # Startup
    try:
        server_instance.start_background_processes()
        yield
    finally:
        # Shutdown
        if server_instance:
            server_instance.stop_background_processes()


# FastAPI app
app = FastAPI(
    title="Activation Server",
    description="High-performance server for neural network activation data",
    version="1.0.0",
    lifespan=lifespan,
)


def get_buffer_stats_sync() -> Dict[str, Any]:
    """Get statistics about the shared buffer (synchronous version)."""
    global server_instance

    if not server_instance or not server_instance.shared_buffer:
        return {"error": "Buffer not initialized"}

    return server_instance.shared_buffer.get_stats()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "Activation Server is running",
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

    if not server_instance or not server_instance.running:
        raise HTTPException(status_code=503, detail="Server not ready")

    if not server_instance.shared_buffer:
        raise HTTPException(status_code=503, detail="Shared buffer not initialized")

    if num_samples <= 0 or num_samples > server_instance.config.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"num_samples must be between 1 and {server_instance.config.max_batch_size}",
        )

    try:
        # Get activations from shared buffer (automatically marks as invalid)
        activations = server_instance.shared_buffer.get_activations(batch_size)

        # Convert to serializable format
        activation_data = {
            "activations": activations.tolist(),  # Convert tensor to list for JSON
            "shape": list(activations.shape),
            "dtype": str(activations.dtype),
            "num_samples": activations.shape[0],  # Actual number returned
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

    if not server_instance or not server_instance.running:
        raise HTTPException(status_code=503, detail="Server not ready")

    if not server_instance.shared_buffer:
        raise HTTPException(status_code=503, detail="Shared buffer not initialized")

    try:
        # Get activations from shared buffer (automatically marks as invalid)
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
            },
        )

    except Exception as e:
        logger.error(f"Error getting tensor activations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving tensor: {str(e)}"
        )


@app.get("/activations/stream")
async def stream_activations_tensor(
    request: Request, batch_size: int = 5000, max_batches: int = 0
) -> Response:
    """
    Stream activation data continuously as raw tensor bytes.
    Returns multiple batches in a streaming format.

    Args:
        batch_size: Number of activation samples per batch
        max_batches: Maximum number of batches to stream (0 = unlimited)

    Returns:
        Streaming tensor data as bytes with custom protocol
    """
    global server_instance

    if not server_instance or not server_instance.running:
        raise HTTPException(status_code=503, detail="Server not ready")

    if not server_instance.shared_buffer:
        raise HTTPException(status_code=503, detail="Shared buffer not initialized")

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

                if await request.is_disconnected():
                    logger.warning("Client disconnected, stopping streaming")
                    print("Client disconnected, stopping streaming")
                    break

                # Get activations from shared buffer
                activations = server_instance.shared_buffer.get_activations(batch_size)

                yield activations.numpy().tobytes()
                batches_sent += 1

        except Exception as e:
            logger.error(f"Error in streaming generator: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error in streaming generator: {e}"
            )

    return StreamingResponse(
        generate_stream(),
        media_type="application/octet-stream",
    )


@app.get("/stats")
async def get_buffer_stats() -> Dict[str, Any]:
    """Get statistics about the shared buffer."""
    return get_buffer_stats_sync()


@app.post("/refresh")
async def refresh_buffer():
    """Manually trigger buffer refresh."""
    global server_instance

    if not server_instance or not server_instance.shared_buffer:
        raise HTTPException(status_code=503, detail="Buffer not initialized")

    try:
        refreshed_count = server_instance.shared_buffer.force_refresh()
        return {"refreshed_indices": refreshed_count}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error refreshing buffer: {str(e)}"
        )


@app.post("/refill")
async def refill_from_file():
    """Manually refill buffer from initialization file."""
    global server_instance

    if not server_instance or not server_instance.data_generator_process:
        raise HTTPException(status_code=503, detail="Server not ready")

    try:
        # We need to call the refill method on the data generator process
        # Since it's in a different process, we'll need to implement this through shared memory
        # For now, let's return info about what would be refilled

        stats = server_instance.shared_buffer.get_stats()
        invalid_count = stats["buffer_size"] - stats["valid_samples"]

        return {
            "message": "Refill requested",
            "invalid_indices_count": invalid_count,
            "note": "Refill will happen automatically in the background process",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error requesting refill: {str(e)}"
        )


def run_server(host: str = "0.0.0.0", port: int = 8000, **kwargs):
    """Run the FastAPI server."""
    uvicorn.run(
        "activation_server.server:app",
        host=host,
        port=port,
        log_level="info",
        access_log=False,  # Disable access logging to prevent dashboard interference
        **kwargs,
    )


if __name__ == "__main__":
    run_server()
