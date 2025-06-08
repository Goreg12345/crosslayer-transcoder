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

# Or run Python scripts directly
.venv/bin/python main.py
```

## Managing Dependencies

```bash
# Add new dependencies
uv add torch transformers

# Add dev dependencies  
uv add --dev pytest black

# Update all dependencies
uv sync
```

## Why uv?

- 🚀 **10-100x faster** than pip/poetry
- 🐍 **Manages Python versions** automatically
- 📦 **No global installs** needed
- 🔒 **Lockfile included** for reproducible builds