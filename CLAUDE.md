# Crosslayer Transcoder - Claude Code Context

## Project Overview
This project implements Anthropic's crosslayer transcoders for neural network interpretability research. The system learns to encode activations from one layer and decode them to reconstruct activations in the same or later layers of a neural network.

## Architecture Components

### Core Model (clt.py)
- **CrossLayerTranscoder**: Main PyTorch Lightning module implementing the transcoder
  - Encoder: `W_enc` maps activations to sparse features
  - Decoder: `W_dec` reconstructs activations from features 
  - Uses triangular mask to ensure causal dependencies (features only contribute to same/later layers)
  - Loss: MSE reconstruction + L1 sparsity penalty with learnable thresholds

### Activation Server (activation_server/)
- **FastAPI-based server** for efficient activation data serving
- **Multiple endpoints**:
  - `/activations`: JSON format batches
  - `/activations/tensor`: Raw tensor format
  - `/activations/stream`: High-throughput streaming
- **Configuration system** with test/production presets
- **Shared memory support** for multi-worker scenarios

### Data Pipeline
- **DiscBuffer** (buffer.py): HDF5-based dataset for efficient random access to large activation files
- **ActivationClient** (client.py): Client library for connecting to activation server
- **WebText dataloader** utilities for language model data

### Custom Components
- **JumpReLU** (jumprelu.py): Custom activation function with learnable thresholds and rectangular gradients
- **ReplacementModel** (replacement_model.py): Testing framework that replaces GPT-2 MLP layers with transcoder reconstructions

## Training Setup
- **PyTorch Lightning** for training orchestration
- **Wandb** for experiment tracking
- **Mixed precision** (16-bit) training
- **GPU compilation** with `torch.compile`
- **Multi-worker data loading** with prefetching

## File Organization
```
├── clt.py                  # Main transcoder model
├── train.py               # Training script
├── client.py              # Activation server client
├── buffer.py              # HDF5 data buffer
├── jumprelu.py            # Custom activation function
├── replacement_model.py   # Model replacement testing
├── activation_server/     # FastAPI server components
├── metrics/              # Evaluation metrics
├── checkpoints/          # Model checkpoints
└── *.ipynb              # Jupyter notebooks for experiments
```

## Key Dependencies
- PyTorch + PyTorch Lightning
- nnsight (for neural network inspection)
- FastAPI + uvicorn (server)
- HDF5 (data storage)
- Wandb (experiment tracking)
- Transformers (language models)

## Usage Patterns
1. **Training**: Run `train.py` with activation data in HDF5 format
2. **Serving**: Use `activation_server/main.py` to serve activation data
3. **Evaluation**: Use `replacement_model.py` to test transcoder quality
4. **Interactive**: Jupyter notebooks for experiments and analysis

## Common Commands
```bash
# Setup environment
./setup.sh

# Train model
python train.py

# Run activation server
python -m activation_server.main --config production

# Test client
python client.py
```

## Refactoring Notes
- Large monolithic files could be split into smaller modules
- Configuration management could be centralized
- Error handling and logging could be standardized
- Type hints could be added throughout
- Testing infrastructure could be improved