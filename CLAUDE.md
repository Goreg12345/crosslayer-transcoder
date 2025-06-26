# Crosslayer Transcoder - Claude Code Context

## Project Overview
This project implements Anthropic's crosslayer transcoders for neural network interpretability research. The system learns to encode activations from one layer and decode them to reconstruct activations in the same or later layers of a neural network.

## Architecture Components

### Core Model (model/clt.py)
- **CrossLayerTranscoder**: Main PyTorch Lightning module implementing the transcoder
  - Encoder: `W_enc` maps activations to sparse features
  - Decoder: `W_dec` reconstructs activations from features 
  - Uses triangular mask to ensure causal dependencies (features only contribute to same/later layers)
  - Loss: MSE reconstruction + L1 sparsity penalty with learnable thresholds

### Data Pipeline (data/)
- **ActivationDataModule**: Lightning DataModule with two modes:
  - **Shared Memory Mode**: High-performance streaming with background processes (always try to use this)
  - **Simple Buffer Mode**: Direct HDF5 file loading for reliability
- **SharedMemoryDataLoader**: Zero-copy data sharing between processes
- **ActivationComputer**: Generates activations via GPT-2 forward passes
- **DiskActivationSource**: Reads pre-computed activations from HDF5 files

### Custom Components
- **JumpReLU** (model/jumprelu.py): Custom activation function with learnable thresholds
- **Callbacks** (utils/callbacks.py): Training checkpoints and monitoring

## Training Setup
- **Lightning CLI** (`cli.py`) for training orchestration
- **YAML configuration** (`config/default.yaml`) for reproducible experiments
- **Wandb** integration for experiment tracking
- **Mixed precision** (16-bit) training support
- **Multiprocessing** data generation with adaptive CPU/GPU switching

## File Organization
```
├── cli.py                  # Lightning CLI entry point
├── config/default.yaml     # Training configuration
├── model/
│   ├── clt.py             # Main transcoder model
│   └── jumprelu.py        # Custom activation function
├── data/                  # High-performance data loading system
│   ├── datamodule.py      # Lightning DataModule
│   ├── shared_memory.py   # Shared memory buffer
│   └── activation_sources.py # Data sources
├── utils/                 # Utilities and callbacks
├── test/                  # Benchmarks and tests
└── *.ipynb               # Jupyter notebooks for experiments
```

## Dependencies
Uses **uv** for dependency management via `pyproject.toml`. Key packages:
- PyTorch + Lightning, nnsight, transformers, wandb, h5py

## Usage Patterns
1. **Training**: `python cli.py fit --config config/default.yaml`
2. **VSCode**: Use Run/Debug configurations for easy development
3. **Benchmarking**: `python test/benchmark_data_loader.py`
4. **Interactive**: Jupyter notebooks for experiments

## Lightning CLI Commands
```bash
# Basic training
python cli.py fit --config config/default.yaml

# Override parameters
python cli.py fit --config config/default.yaml \
  --data.batch_size 8000 --model.learning_rate 0.001

# Different modes
python cli.py fit --config config/default.yaml \
  --data.use_shared_memory true --data.buffer_size 10000000
```

## Refactoring Notes
- Large monolithic files could be split into smaller modules
- Configuration management could be centralized
- Error handling and logging could be standardized
- Type hints could be added throughout
- Testing infrastructure could be improved