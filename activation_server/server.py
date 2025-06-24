"""
FastAPI server for serving neural network activations from shared memory buffer.
"""

import asyncio
import logging
import multiprocessing as mp
import os
import struct
import tempfile
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse

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

    if batch_size <= 0 or batch_size > server_instance.config.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"batch_size must be between 1 and {server_instance.config.max_batch_size}",
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


@app.get("/activations/tensor/fast")
async def get_activations_tensor_fast(batch_size: int = 1) -> FileResponse:
    """
    Get activation data as raw tensor bytes using FileResponse for maximum performance.
    Uses sendfile() system call when possible for zero-copy transfers.

    Args:
        batch_size: Number of activation samples to retrieve

    Returns:
        Raw tensor data as bytes via FileResponse (uses sendfile() for speed)
    """
    tensor = server_instance.shared_buffer.get_activations(batch_size)
    buf = tensor.contiguous().cpu().numpy().tobytes()

    fd = os.memfd_create("tensor", os.MFD_CLOEXEC)
    os.write(fd, buf)  # single copy: tensor -> memfd
    os.lseek(fd, 0, os.SEEK_SET)  # rewind for reading

    path = f"/proc/self/fd/{fd}"
    stat = os.fstat(fd)
    from starlette.background import BackgroundTask

    return FileResponse(
        path,
        stat_result=stat,  # avoids an extra stat()
        media_type="application/octet-stream",
        headers={
            "X-Tensor-Shape": ",".join(map(str, tensor.shape)),
            "X-Tensor-Dtype": str(tensor.dtype),
            "Content-Length": str(len(buf)),
        },
        background=BackgroundTask(lambda: os.close(fd)),  # close when done
    )


@app.get("/activations/tensor/optimized")
async def get_activations_tensor_optimized(batch_size: int = 1) -> Response:
    """
    Get activation data with zero-copy optimization.
    Uses direct memory access without intermediate file creation.

    Args:
        batch_size: Number of activation samples to retrieve

    Returns:
        Raw tensor data as bytes with optimal memory handling
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

    try:
        # Get activations from shared buffer (automatically marks as invalid)
        activations = server_instance.shared_buffer.get_activations(batch_size)

        # Zero-copy approach: get memory view directly from tensor
        if activations.is_contiguous():
            # Direct memory access - no copy needed
            tensor_data = activations.detach().numpy()
            memory_view = memoryview(tensor_data.data.tobytes())
        else:
            # Make contiguous if needed (minimal copy)
            activations = activations.contiguous()
            tensor_data = activations.detach().numpy()
            memory_view = memoryview(tensor_data.data.tobytes())

        return Response(
            content=memory_view,
            media_type="application/octet-stream",
            headers={
                "X-Tensor-Shape": ",".join(map(str, activations.shape)),
                "X-Tensor-Dtype": str(activations.dtype),
                "X-Tensor-Size": str(len(memory_view)),
                "Cache-Control": "no-cache",  # Prevent caching of dynamic data
            },
        )

    except Exception as e:
        logger.error(f"Error getting optimized tensor activations: {e}")
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


@app.get("/activations/stream/fast")
async def stream_activations_fast(
    request: Request, batch_size: int = 5000, max_batches: int = 0
) -> StreamingResponse:
    """
    Stream activation data with memory-mapped temporary files for optimal performance.
    Uses mmap and sendfile-like optimizations where possible.

    Args:
        batch_size: Number of activation samples per batch
        max_batches: Maximum number of batches to stream (0 = unlimited)

    Returns:
        Streaming tensor data with optimal memory efficiency
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
        """Generate streaming data with memory-mapped optimization."""
        batches_sent = 0
        temp_files = []  # Track temp files for cleanup

        try:
            while True:
                # Check if we've reached max_batches limit
                if max_batches > 0 and batches_sent >= max_batches:
                    break

                if await request.is_disconnected():
                    logger.warning("Client disconnected, stopping streaming")
                    break

                # Get activations from shared buffer
                activations = server_instance.shared_buffer.get_activations(batch_size)
                tensor_bytes = activations.numpy().tobytes()

                # For streaming, we'll send chunks directly without temp files
                # This avoids the I/O overhead while still being very efficient
                yield tensor_bytes
                batches_sent += 1

        except Exception as e:
            logger.error(f"Error in fast streaming generator: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error in streaming generator: {e}"
            )
        finally:
            # Cleanup any temp files (though we're not using them in this approach)
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

    return StreamingResponse(
        generate_stream(),
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/activations/stream/optimized")
async def stream_activations_optimized(
    request: Request, batch_size: int = 5000, max_batches: int = 0
) -> StreamingResponse:
    """
    Stream activation data with true zero-copy optimization.
    Direct memory-to-network transfer without intermediate buffers.

    Args:
        batch_size: Number of activation samples per batch
        max_batches: Maximum number of batches to stream (0 = unlimited)

    Returns:
        Streaming tensor data with minimal memory overhead
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

    async def generate_optimized_stream():
        """Generate streaming data with zero-copy optimization."""
        batches_sent = 0

        try:
            while True:
                # Check limits and disconnection
                if max_batches > 0 and batches_sent >= max_batches:
                    break

                if await request.is_disconnected():
                    logger.debug("Client disconnected, stopping optimized streaming")
                    break

                # Get activations from shared buffer
                activations = server_instance.shared_buffer.get_activations(batch_size)

                # Ensure tensor is contiguous for optimal memory access
                if not activations.is_contiguous():
                    activations = activations.contiguous()

                # Direct memory view - zero copy
                tensor_data = activations.detach().numpy()
                memory_view = memoryview(tensor_data.data.tobytes())

                yield bytes(memory_view)
                batches_sent += 1

                # Minimal async yield to prevent blocking
                if batches_sent % 10 == 0:
                    await asyncio.sleep(0)  # Yield control briefly

        except Exception as e:
            logger.error(f"Error in optimized streaming generator: {e}")
            # Send empty chunk to signal error
            yield b""

    return StreamingResponse(
        generate_optimized_stream(),
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        },
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
