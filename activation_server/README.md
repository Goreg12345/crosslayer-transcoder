# Activation Server

A high-performance FastAPI server for serving neural network activations with shared memory buffers and multiprocessing.

## Features

- **FastAPI Server**: RESTful API for retrieving activation data
- **Shared Memory**: Large PyTorch tensors stored in shared memory (up to 1TB+)
- **Multiprocessing**: Background process continuously generates new activation data
- **Queue-based Communication**: Efficient inter-process communication for data refresh
- **Multiple Data Formats**: JSON and raw tensor byte responses
- **Configurable**: Multiple configuration presets for different use cases

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │  Shared Memory   │    │ Data Generator  │
│   Server        │◄──►│    Buffer        │◄──►│   Process       │
│   (Main)        │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                        │
        │                       │                        │
        ▼                       ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   HTTP          │    │ PyTorch Tensor   │    │   LLM Model     │
│   Requests      │    │ (1TB+ in RAM)    │    │   + Dataset     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Using Test Configuration (Recommended for first run)

```bash
# Install dependencies
uv sync

# Test configuration and shared memory (fast)
python -m activation_server.run_example --config-test

# Run with test config (uses small model, small buffer)
python -m activation_server.main --config test --buffer-size 100

# Test the server (in another terminal)
curl http://localhost:8000/
curl http://localhost:8000/stats
curl "http://localhost:8000/activations?num_samples=5"
```

### Using Production Configuration

```bash
# Run with production config (real model, large buffer)
python -m activation_server.main --config production

# Or with custom settings
python -m activation_server.main \
    --config production \
    --buffer-size 1000000 \
    --activation-dim 4096 \
    --host 0.0.0.0 \
    --port 8000
```

## API Endpoints

### GET `/`
Health check and server status.

### GET `/activations?num_samples=N`
Get activation data as JSON.
- `num_samples`: Number of samples to retrieve (default: 1)

**Response:**
```json
{
    "activations": [[...], [...]],
    "shape": [2, 4096],
    "dtype": "torch.float32",
    "num_samples": 2,
    "indices": [123, 456],
    "buffer_stats": {...}
}
```

### GET `/activations/tensor?num_samples=N`
Get activation data as raw tensor bytes.
- Headers contain tensor metadata (`X-Tensor-Shape`, `X-Tensor-Dtype`, etc.)

### GET `/stats`
Get buffer statistics and server info.

### POST `/refresh`
Manually trigger buffer refresh.

## Configuration

### Presets

- `test`: Small buffer, mock data, fast startup
- `production`: 1M samples, real models, balanced performance  
- `large-scale`: 10M samples, half precision, maximum scale
- `env`: Load from environment variables

### Environment Variables

```bash
export BUFFER_SIZE=1000000
export ACTIVATION_DIM=4096
export MODEL_NAME="microsoft/DialoGPT-medium"
export DATASET_NAME="openwebtext"

export HOST="0.0.0.0"
export PORT=8000
```

### Memory Requirements

| Configuration | Buffer Size | Activation Dim | Memory (float32) | Memory (float16) |
|---------------|-------------|----------------|------------------|-------------------|
| Test          | 1,000       | 768           | ~3 MB            | ~1.5 MB          |
| Production    | 1,000,000   | 4,096         | ~16 GB           | ~8 GB            |
| Large Scale   | 10,000,000  | 4,096         | ~160 GB          | ~80 GB           |

## Client Usage

```python
from activation_server.client import ActivationClient

client = ActivationClient("http://localhost:8000")

# Wait for data to be ready
client.wait_for_data(min_samples=100)

# Get activations as tensor
tensor, metadata = client.get_activations_tensor(num_samples=10)
print(f"Shape: {tensor.shape}")

# Get server stats
stats = client.get_stats()
print(f"Valid samples: {stats['valid_samples']}")
```

## Development

### Running Tests

```bash
# Quick configuration test (no model loading)
python -m activation_server.run_example --config-test

# Full demo (downloads models, generates real activations)
python -m activation_server.run_example --demo

# Or run server manually
python -m activation_server.main --config test --buffer-size 100 &
python -m activation_server.client
pkill -f activation_server
```

### Adding New Models

Modify `config.py` to add new model configurations:

```python
def get_custom_config() -> ServerConfig:
    return ServerConfig(
        model_name="your-model-name",
        activation_dim=your_activation_dim,
        target_layer=-1,  # Which layer to extract
        # ... other settings
    )
```

### Model Requirements

The server generates real LLM activations, which requires:
- Model downloads (several GB depending on model)
- GPU memory for larger models (recommended)
- CPU fallback supported but slower

## Troubleshooting

### Memory Issues
- Reduce `buffer_size` or use `torch.float16` dtype
- Monitor with `GET /stats` endpoint
- Check available RAM with `free -h`

### Model Loading Issues
- Ensure sufficient GPU memory for the model
- Use smaller models (e.g., `distilbert-base-uncased`) for testing
- Check internet connection for model downloads
- Use `--buffer-size` to reduce memory usage for testing

### Process Communication Issues
- Check that multiprocessing is supported on your system
- Look for shared memory cleanup errors in logs
- Restart server if processes become unresponsive

### Performance Tuning
- Adjust `generation_batch_size` based on your GPU
- Increase `buffer_size` for better cache hit rates
- Use half precision (`torch.float16`) to reduce memory usage

## Logging

Logs are written to both console and `activation_server.log`:

```bash
# View real-time logs
tail -f activation_server.log

# Change log level
python -m activation_server.main --log-level DEBUG
``` 