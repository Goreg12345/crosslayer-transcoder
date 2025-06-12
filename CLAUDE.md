# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# One-command setup (installs uv, Python 3.12, creates .venv, installs dependencies)
./setup.sh

# Activate environment
source .venv/bin/activate

# Run with activated environment
python main.py
jupyter lab

# Or run directly without activation
.venv/bin/python main.py
.venv/bin/jupyter lab
```

### Dependency Management (using uv)
```bash
# Add new dependencies
uv add torch transformers

# Add dev dependencies  
uv add --dev pytest black

# Update all dependencies
uv sync
```

### Training Commands
```bash
# Main training script (multi-GPU with DDP)
python train.py

# Monitor training with TensorBoard profiling
tensorboard --logdir=log/ddp
```

## Architecture Overview

This project implements **crosslayer transcoders** - sparse autoencoders that learn interpretable features across multiple transformer layers simultaneously, rather than analyzing each layer independently.

### Core Components

**CrossLayerTranscoder (`clt.py`)**
- Main Lightning module implementing the sparse autoencoder
- Key tensors:
  - `W_enc`: (n_layers, d_acts, d_features) - encoder weights
  - `W_dec`: (n_layers, n_layers, d_features, d_acts) - decoder weights  
  - `mask`: Upper triangular mask ensuring causal reconstruction (features only contribute to same/later layers)
- Loss: MSE reconstruction + L1 sparsity with tanh smoothing
- Training normalizes activations per-layer to handle varying scales

**JumpReLU Activation (`jumprelu.py`)**
- Custom activation function with learnable per-layer, per-feature thresholds
- Only activates when input > threshold AND input > 0
- Includes custom autograd function for proper gradient computation
- Shape: `theta` parameter is (1, n_layers, d_features)

**Data Pipeline**
- `DiscBuffer` (`buffer.py`): HDF5-based dataset for efficient random access to pre-computed MLP activations
- Training data: 10M activation pairs from GPT-2's MLP layers stored at `/var/local/glang/activations/clt-activations-10M.h5`
- `utils.py`: WebText dataset loading and tokenization utilities

**Evaluation System**
- `ReplacementModelAccuracy` (`metrics/replacement_model_accuracy.py`): Tests transcoder quality by replacing GPT-2 MLP layers
- Uses nnsight library to hook into GPT-2 and substitute MLP outputs with transcoder reconstructions
- Measures next-token prediction accuracy as functional validation

### Training Configuration

**Default Hyperparameters (in `train.py`):**
- Architecture: 768 → 6,144 features (8x expansion) across 12 layers
- Batch size: 4,000
- Learning rate: 1e-3
- Sparsity coefficient (λ): 0.0002
- Tanh smoothing (c): 0.1
- JumpReLU threshold: 0.03
- Training: 2,000 steps with validation every 1,000 steps
- Multi-GPU: 4 GPUs with DDP strategy, mixed precision

**Key Training Details:**
- Activations are normalized per-layer during forward pass: `(batch - mean) / std`
- Input format: `batch[:, 0]` = residual stream, `batch[:, 1]` = MLP output
- Reconstruction target is the original MLP output, not the residual
- Upper triangular masking in decoder ensures causal structure

### Logging and Monitoring

**W&B Integration:**
- Project: "wandb_clt"
- Tracks: train_loss, train_mse, train_sparsity, L0 sparsity, replacement_model_accuracy

**TensorBoard Profiling:**
- GPU utilization traces stored in `log/ddp/`
- Enable by uncommenting `TBProfilerCallback` in trainer

### File Organization

- `train.py`: Main training script with Lightning trainer setup
- `clt.py`: CrossLayerTranscoder model definition
- `jumprelu.py`: Custom activation function
- `buffer.py`: HDF5 dataset interface
- `metrics/replacement_model_accuracy.py`: Functional evaluation metric
- `utils.py`: Data loading utilities
- Notebooks: `train.ipynb`, `replacement_model.ipynb` for experimentation