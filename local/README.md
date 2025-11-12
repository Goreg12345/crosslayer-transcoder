# Local Experiments Directory

This directory is for your personal experiments and is **not committed to git**.

## Structure

```
local/
├── configs/           # Your personal config files
├── checkpoints/       # Model checkpoints and weights
├── wandb/            # WandB experiment logs
├── lightning_logs/    # Lightning training logs
└── experiments/       # Your experiment scripts
```

## Usage

### Running Experiments

```bash
# From project root, use your local configs
python main.py fit --config local/configs/my_experiment.yaml

# Or with overrides
python main.py fit --config local/configs/my_experiment.yaml \
  --trainer.max_steps=1000 \
  --model.d_features=8192
```

### Example Local Config

Create `local/configs/my_experiment.yaml`:

```yaml
seed_everything: 42
trainer:
  max_steps: 1000
  accelerator: gpu
  devices: [0]
  precision: 16-mixed
  logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args:
      project: "my-crosslayer-experiments"
      name: "experiment-1"
      save_dir: "./local/wandb"

model:
  class_path: crosslayer_transcoder.model.CrossLayerTranscoderModule
  init_args:
    d_acts: 768
    d_features: 8192
    learning_rate: 0.001

data:
  class_path: crosslayer_transcoder.data.ActivationDataModule
  init_args:
    buffer_size: 1000000
    batch_size: 4000
```

### Benefits

- **Clean package**: Only example configs in `config/`
- **Personal experiments**: All your work stays local
- **Easy sharing**: Package repo stays focused on code
- **Flexible**: Can have multiple experiment setups

## Notes

- This directory is gitignored
- Use relative paths in configs (e.g., `./local/checkpoints`)
- WandB and Lightning will automatically use these directories
- You can organize experiments by date, model type, etc.
