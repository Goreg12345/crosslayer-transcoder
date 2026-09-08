# MOLT campaign order

All layer numbers are zero-based. Every run gets a distinct W&B name, shared
memory name, and checkpoint directory of the form:

`checkpoints/molt-gemma3-4b-it-rtx6000/layer-<layer>/<variant>/training.ckpt`

## Gates

1. Finish the existing layer-22 baseline.
2. Run the matched layer-22 `preact-1e-6` control. Its only optimization
   difference is `model.init_args.pre_actv_loss: 1e-6`.
3. Compare baseline/control dead fraction, L0, MSE, and reconstruction quality.
   Do not start pilots until the control has completed and been reviewed.
4. Before the multi-layer campaign, enable ECC on the RTX PRO 6000 and
   reboot/reset the GPU. Verify that `nvidia-smi -i 1 -q -d ECC` reports
   `Current: Enabled` before continuing.
5. Run four pilots sequentially: layers 0, 1, 11, and 32. These sample the
   embedding-adjacent, early, middle, and late network while retaining layer 22
   as the initial experiment.
6. Review all pilots, then schedule the remaining 29 layers sequentially.

The matched control command is:

```bash
set -a
source .env
set +a
export HF_TOKEN="$HUGGINGFACE_API_KEY"
uv run clt fit \
  --config config/molt-gemma3-4b-it-5090-extractor-rtx6000.yaml \
  --config config/molt-gemma3-4b-it-preact-control.yaml
```

The current live process loaded its original checkpoint path before the base
config was updated. After it exits successfully, move its final checkpoint
from `checkpoints/molt-gemma3-4b-it-rtx6000/clt.ckpt` to
`checkpoints/molt-gemma3-4b-it-rtx6000/layer-22/baseline/training.ckpt` before
starting the control. Do not move a checkpoint while the writer is live.

`tools/run_layer22_control_after_baseline.sh` performs that validation and
migration, and refuses to launch the control unless the baseline checkpoint's
global step is at least 100,000.
