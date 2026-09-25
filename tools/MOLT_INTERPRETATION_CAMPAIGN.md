# Every campaign layer: shared 10M tokens and four Delphi scores

Georg authorized the other layers after layer 22: collect their 10M-token caches
on the local RTX 5090 now, then generate explanations, titles, detection, fuzz,
active-fuzz, and intruder scores after the current layer-22 run finishes.
Simulation remains excluded.

Progress: <http://100.76.58.1:8765/layers/>

## Checkpoints and collection

The queue follows the existing 36-layer campaign order: 22 first, then 0, 1, 10,
30, and the remaining layers in increasing order. It only uses completed
100,000-step `N80-rank512to32-lr4e-5-sp7.5e-5-native-ramp80` checkpoints.
In-progress checkpoints remain queued until training finishes. The initially
ready cohort is layers 0–7, 10, and 30.

`tools/collect_molt_layers.py` replays the exact token IDs and conversation
boundaries from the existing layer-22 10M-token cache. It captures pre-norm MLP
inputs and applies each checkpoint's own encoder, standardizer, and thresholds.
All ready layers share one Qwen forward pass through the deepest requested
layer. Gates are sparsified and copied to CPU in each layer hook so multiple
large dense activation tensors do not remain on the GPU.

Collection uses batch size 16, BF16, and atomic 128-conversation shards. Its CUDA
allocator is capped at 10 GiB on GPU
`GPU-2ada3848-a497-abbf-c0da-c053a9680677` (RTX 5090). Driver/context overhead is
additional. The initial ten-layer run reserves roughly 9 GiB in practice and
leaves roughly 11 GiB free alongside the training workers. It does not stop or
reconfigure training, the activation workers, or the PRO 6000 steering server.

Resume preserves original batch boundaries and can recover if an interruption
commits some layers' shards before others. Each store skips its already committed
conversations. Reference token/offset hashes, checkpoint path/size/mtime, layer,
precision, and sampling settings are recorded. Configuration changes are rejected.

A 14,696-token validation across layers 0, 22, and 30 matched every layer-22 token,
sparse position, feature ID, and gate value against the original cache exactly.

## Interpretation and publication

The API coordinator waits for the current `molt-autointerp-layer22.service` to
exit and for layer 22 to be complete, resuming incomplete work if necessary.
It then processes one cached layer at a time. The global endpoint load remains
32 concurrent requests, with a 10,000-token context budget and the same endpoint
model. No additional API requests are launched while the original layer-22
worker is active.

The [full-layer pipeline](MOLT_LAYER_EXPLANATIONS.md) now accepts other completed
layers. It uses the approved layer-22 method settings but each layer's own cache,
checkpoint, feature count, and evidence. A local pilot is optional: only layer
22 reuses its pilot results. Its existing fingerprint remains unchanged.
Titles and explanations are generated before the four scoring methods. Sparse
evidence, unobserved transforms, invalid responses, and unsupported scores remain
explicit, with the same retry and audit policy as layer 22.

Recovery was hardened on September 20, 2026 after a layer-0 explanation repeatedly
hit the output limit while greedily repeating one example word. Empty or truncated
explanations now retry with a more constrained answer format and the same evidence;
the actual revised prompt is audited and gets its own cache key. Truncated output
is never accepted as an explanation. Scoring prompts remain unchanged.

After three unsuccessful worker resumptions, the coordinator preserves the layer's
incomplete status and proceeds to another ready layer. A six-hour retry cooldown
is persisted in `.api-retries.json` under the campaign portal and survives service
restarts. The dashboard shows the deferred retry time. A different interpretation
fingerprint does not inherit the old cooldown, and deferred work is revisited after
the cooldown when the current layer finishes. The queue reports completion only
when every layer is actually complete.

The collection queue publishes activation explorers as caches become ready.
The API coordinator publishes each layer's live title/explanation/score pages.
Per-directory renderer locks prevent overlapping publishers from corrupting
temporary HTML files. Pages identify their actual layer. Only layer 22 links to
the existing layer-22 steering engine.

## Services, files, and resume

- `molt-collect-other-layers.service`: the first ten-layer collection cohort.
- `molt-collection-queue.service`: adopts that cohort, publishes its caches, and
  collects later completed checkpoints.
- `molt-interpretation-queue.service`: waits for layer 22, then processes the
  remaining cached layers sequentially.

The two queue units are enabled user services under
`~/.config/systemd/user/`. They resume from persisted artifacts when user services
restart. They use a 12 GiB cgroup memory cap, lower CPU priority, and bounded
systemd restart attempts. The API unit loads `.env` through `EnvironmentFile`;
keys are absent from command arguments, dashboards, and prompt logs.

Queue plan/status files are under `results/molt-qwen-campaign-interpretation/`:
`plan.json`, `collection-queue.json`, and `api-queue.json`. Each `layer-N` symlink
there points to its independent cache at
`results/molt-qwen-dashboard-campaign-layerN-10m/`. The existing MOLT server
serves this directory through its `/layers/` route.

Logs:

- `logs/molt-collect-other-layers.log`
- `logs/molt-collection-queue.log`
- `logs/molt-collection-queue-workers.log`
- `logs/molt-collection-queue-render.log`
- `logs/molt-interpretation-queue.log`
- `logs/molt-autointerp-layerN-queued.log`
- `logs/molt-autointerp-layerN-queued-review.log`

Normally no manual handoff is needed. To resume a stopped queue after addressing
an error, restart its user service. Do not start a duplicate collector or API
worker for a directory already being processed. Queue and per-cache locks reject
duplicate owners; completed cache and method results are reused.
