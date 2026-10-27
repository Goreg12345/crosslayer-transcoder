# Current campaign

Shared config: [molt-qwen3-all-mlps.yaml](molt-qwen3-all-mlps.yaml).
Qwen3-4B layers 22, 0, 1, 10, 30 first, then the remaining 31.
N=80, ranks 512–32, 1.055B parameters per layer, LR 4e-5, peak sparsity 7.5e-5.
W&B project: `molt`; activation generation batch size: 4.
Training uses `cuda:0` (RTX PRO 6000); activation generation uses `cuda:1`
(RTX 5090). These are PyTorch device indices, not `nvidia-smi` indices.
Sparsity reaches its peak at 80%; LR declines linearly to zero over the final 20%.

Only the target layer is buffered: 8.8M entries, about 90.1 GB BF16 storage.
Maximum sequence length is 1024. The native Qwen chat template yields
183,838,055 non-padding training tokens after truncation (207,865 conversations).
No BOS is manually added. Count: `results/qwen3-campaign/token-count.json`.

Start or resume the first five:

```bash
tools/run_qwen_molt_campaign.sh --first-five
```

Start or resume all layers:

```bash
tools/run_qwen_molt_campaign.sh
```

Use `--dry-run` to print commands. Ctrl-C stops the training process and its
workers. The same command resumes an incomplete checkpoint or skips a completed
100,000-step (100M-token) run. Each layer has one `training.ckpt`, atomically
overwritten every 90 minutes and at completion. Completed files stay in place.
Progress since the last saved checkpoint is lost when stopping.

The campaign is prepared, not launched. The older notes below are superseded.

---

## Historical Gemma campaign notes

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
