# Crosslayer Transcoder


This repository trains Crosslayer Transcoders and variants with PyTorch/Lightning on multi‑GPU via tensor parallelism. It implements Anthropic’s [crosslayer transcoders](https://transformer-circuits.pub/2024/crosscoders/index.html) and related architectures (per‑layer transcoders, MOLTs, SAEs, Matryoshka CLTs) and supports losses such as ReLU, JumpReLU, TopK, and BatchTopK, for learning human‑interpretable features from LLM activations and building replacement models for [circuit tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).

We want to understand the “brain” of LLMs: what their representations encode and what algorithms emerged from circuits. To start, we can learn feature dictionaries with [Sparse Autoencoders](https://transformer-circuits.pub/2023/monosemantic-features) to break activations into human‑interpretable features. They tell us _what_ features representations contain but not _how_ they interact to make circuits and algorithms. For that, we need to sparsify the entire model (we call this a sparse replacement model), not just representations of a single layer. One approach are [Transcoders](https://arxiv.org/abs/2406.11944), which learn features that approximate MLP components, which lets us swap in a replacement model and trace circuits end to end. [Crosslayer transcoders](https://transformer-circuits.pub/2024/crosscoders/index.html) allow features to affect all subsequent layers, essentially letting features live across layers. This yields smaller and more interpretable circuits and enables [circuit tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) and studies of [LLM biology](https://transformer-circuits.pub/2025/attribution-graphs/biology.html).


## Implemented and Planned Features


> **⚠️ Early Development Disclaimer**  
> This repository is still in very early development and under active development. It's not yet a stable, production-ready package. There will likely be many breaking changes in the future as the codebase evolves. Use at your own risk and expect API changes between commits.

### Architectures
- ✅ Per-Layer Transcoder (PLT)
- ✅ Crosslayer Transcoder (CLT)
- ✅ Sparse Mixture of Linear Transforms (MOLT)
- ⏳ Matryoshka CLTs
- ⏳ SAEs (by tweaking the activation data extractor)

### Nonlinearities and Loss Functions
- ✅ ReLU and JumpReLU (via straight-through estimators)
- ✅ TopK
- ✅ BatchTopK (per layer and across layers)
- ✅ Pre-activation loss (reduces dead features for (Jump-)ReLU CLTs)
- ✅ TopK auxiliary loss (reduces dead features for (Batch-)TopK CLTs)
- ✅ Tanh sparsity penalty
- ✅ Activation standardization

### Training
- ✅ On-demand activation extraction and streaming using a shared-memory activation buffer
- ⚠️ Tensor parallelism using PyTorch DTensor API (requires PyTorch 2.8; comms optimization in progress)
- ⏳ Sparse Kernels
- ✅ Distributed data parallelism
- ✅ WandB logging
- ✅ Mixed precision (float16 + gradient scaler or bfloat16), gradient accumulation, checkpointing, profiling

### Metrics (logged to WandB during training)
- ✅ Replacement Model Accuracy and KL divergence
- ✅ Dead Features
- ✅ Feature activation frequency and other statistics
- ✅ L0
- ⏳ Replacement Model Score


## Installation

Recommended: use the setup script (it installs uv if needed and creates the venv).

```bash
# Clone the repository and enter the directory
git clone https://github.com/Goreg12345/crosslayer-transcoder.git
cd crosslayer-transcoder

# Run the setup script (installs uv if needed and creates .venv)
./setup.sh
```

Notes:
- This will create `.venv/` and install from `pyproject.toml`, using `uv.lock` for reproducibility.
- For GPU installs, ensure you have a compatible PyTorch build for your CUDA setup. If needed, follow the official PyTorch instructions to select the right wheel for your CUDA version.




## How to Use (Configure and Customize with Lightning CLI)

You can customize almost everything: datasets and activation extraction, model architecture, loss functions, and all training hyperparameters. This works by using PyTorch Lightning’s CLI to read a YAML config that defines which classes to use and how to compose them. By editing a single `config.yaml`, you control the entire run and keep every parameter in one place; you can still override any field from the command line for quick experiments.

- Why this is great
  - Single source of truth for all settings → easy to reproduce and share
  - Composable: swap architectures, losses, data pipelines by changing class entries in YAML
  - Discoverable and explicit: every knob is visible in one file, with sane defaults
  - Fast iteration: override any value from the CLI without touching code

- How it works
  - The CLI loads the YAML, instantiates the classes declared there, and wires them together (data → model → trainer)
  - You can keep multiple configs (e.g., under `config/`) for different experiments and hardware setups
  - Any setting can be overridden via dot notation flags on the command line

### Examples

```bash
# Train with a config file
python main.py fit --config ./config/default.yaml

# Override anything at runtime (examples)
python main.py fit --config ./config/default.yaml \
  --trainer.max_epochs=10 \
  --data.num_workers=8 \
  --model.nonlinearity=topk \
  --model.num_features=32768
```

You can also switch components by editing the YAML to point to different classes and init args. A minimal schematic (names are illustrative):

```yaml
model:
  class_path: crosslayer_transcoder.model.CrossLayerTranscoderModule
  init_args:
    d_features: 32768
    nonlinearity_theta: 0.03
    lambda_sparsity: 0.0004
data:
  class_path: crosslayer_transcoder.data.ActivationDataModule
  init_args:
    dataset_name: "Skylion007/openwebtext"
    buffer_size: 1000000
    batch_size: 4000
trainer:
  max_epochs: 10
  accelerator: gpu
  devices: 4
```

### Example Configurations

The `config` folder contains example configuration files for different architectures. These serve as good starting points and can be easily customized for your needs:

- `default.yaml` and `jumprelu-clt.yaml` trains a small JumpReLU Crosslayer Transcoder with **120k features** on **gpt2-small** using the **openwebtext** dataset. It uses a small pre-activation loss to reduce the number of dead features.
- `topk-clt.yaml` trains a small TopK Crosslayer Transcoder with **240k features** and **16 active** features per token and layer. It uses auxilliary loss to reduce the number of dead features.
- `jumprelu-plt.yaml` trains an ordinary Transcoder ("Per-Layer Transcoder").

### Extending with your own components

1. **Implement your new architecture/loss/data class** following the same interface as existing components
2. **Reference it in the YAML** via its import path and init args
3. **It plugs into the same training loop** - multi‑GPU (DDP) works out of the box
4. **Tensor parallelism works automatically** because PyTorch Lightning handles the distributed setup and PyTorch's Distributed Tensor API shards your model across GPUs without requiring changes to your component code


## Development

Format and lint the codebase:

```bash
uv run ruff format .
uv run ruff check --fix .
```

## Testing

Run the test suite to ensure everything is working correctly:

```bash
# Run all tests
uv run pytest

# Run tests excluding GPU-dependent ones
uv run pytest -m "not gpu"

# Run only fast tests (exclude slow ones)
uv run pytest -m "not slow"
```
