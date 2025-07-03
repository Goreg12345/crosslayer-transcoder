# Crosslayer Transcoder

An implementation of Anthropic's crosslayer transcoders.

## Quick Setup

**One command to set up everything:**

```bash
./setup.sh
```

This will:
- ✅ Install `uv` (fast Python package manager) if not present
- ✅ Install Python 3.12+ if needed
- ✅ Create a `.venv` virtual environment locally
- ✅ Install all dependencies (including dev dependencies)

No need to have Poetry, pipx, or even the right Python version installed beforehand!

## Usage

After running setup:

```bash
# Activate the environment
source .venv/bin/activate

# Run Jupyter Lab
.venv/bin/jupyter lab

# Or start training via Lightning CLI
.venv/bin/python cli.py fit --help
```

## Project Structure

- **`data/`** - High-performance data pipeline with shared memory streaming, background data generation, and activation loading from HDF5 files
- **`model/`** - Core transcoder implementation (`clt.py`) and PyTorch Lightning wrapper (`clt_lightning.py`) with training logic  
- **`metrics/`** - Model evaluation metrics including replacement model accuracy for interpretability analysis
- **`config/`** - YAML configuration files for reproducible training runs via Lightning CLI
- **`utils/`** - Utilities, callbacks, and helper functions for training and data processing
- **`test/`** - Benchmarks and performance tests for the data loading pipeline
- **`cli.py`** - Lightning CLI entry point for training with flexible configuration options

## Managing Dependencies

```bash
# Add new dependencies
uv add torch transformers

# Add dev dependencies  
uv add --dev pytest black

# Update all dependencies
uv sync
```
