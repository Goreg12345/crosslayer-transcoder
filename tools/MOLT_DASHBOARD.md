# Qwen MOLT feature dashboard

Open `results/molt-qwen-dashboard/index.html`, or serve the complete directory:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory results/molt-qwen-dashboard
```

Visit http://127.0.0.1:8765. All browser assets are local; viewing requires no model, GPU, or network. Keep the `features/` and `assets/` directories beside the index when copying it.

The dashboard uses [SAE-Vis 0.3.7](https://github.com/callummcdougall/sae_vis), which provides activation histograms, token highlights and hover values. Its author no longer actively maintains it; its rendering API remains useful here. We feed it precomputed MOLT gate activations and use only the activation components. MOLT does not have the fixed decoder direction assumed by ordinary SAE logit-lens panels.

Each of the 2,480 features is one scalar transform gate, in checkpoint encoder-row order. Ranks are 512 (IDs 0–79), 256 (80–239), 128 (240–559), 64 (560–1199), and 32 (1200–2479). Gate strength is `JumpReLU(e((x - saved_mean) / saved_std))`; active means strictly greater than zero. The sparsity loss's norm/tanh weighting is not part of this activation. Strength does not equal the transform's output norm or causal importance.

The default checkpoint is `checkpoints/molt-qwen3-4b/layer-22/chat-ultrachat-lr4e-5-sparsity3e-5/training.ckpt`, step 100,000. Inputs are captured before Qwen layer 22's `post_attention_layernorm`, matching the checkpoint's `pre_norm` setting, in BF16 with BF16 encoder autocast as in training. Only the encoder, saved standardizer and JumpReLU thresholds are loaded onto the GPU. Later Qwen layers are skipped after input capture.

The previous L0≈3 dashboard is preserved in `results/molt-qwen-dashboard-l0-3`.

The current higher-density sample contains the first 128 held-out UltraChat `test_sft` conversations, individually truncated at 1,024 tokens using the model's chat template: 110,066 tokens, average L0 11.2165. Chat and special tokens count; padding does not. Feature firing rates use all sampled tokens as the denominator. The strength histogram includes positive activations only; binary histograms include both inactive and active token counts. Top examples use one peak per conversation with up to 16 surrounding tokens on each side; medium/lower groups are selected from remaining conversation peaks. Binary mode keeps the same examples and replaces positive strengths by one.

No-activation features remain browsable and mean only “not observed in this finite sample.” This is a descriptive sample, not a new MSE evaluation or a semantic feature annotation.

## Regenerate

The renderer needs older dependencies, installed in `.venv-dashboard` as a separate overlay over the existing project `.venv`. It leaves the training environment unchanged:

```bash
bash tools/setup_dashboard.sh
PYTHONPATH=. .venv/bin/python tools/collect_molt_dashboard.py --device cuda:0
.venv-dashboard/bin/python tools/render_molt_dashboard.py
```

Collection defaults to the checkpoint above, 128 conversations and 1,024 tokens. Increase `--conversations` for greater feature coverage; specify `--output` for a separate sample. Rendering accepts the output directory as a positional argument. `metadata.json` records the checkpoint path, size, modification timestamp, model, layer, sampling and precision. `activations.npz` stores sparse gate observations and token IDs, and `feature_stats.json` stores all feature statistics. Rendering can be repeated without running Qwen again.

Validation: `PYTHONPATH=. .venv/bin/pytest -q tests/test_dashboard_gates.py` checks gate-only/full-MOLT equivalence and exact threshold semantics.

## Positive and negative logits

The feature page's **Direct logit projection** selector chooses the average over all active sampled tokens or any displayed example's peak token. Clicking an example selects its projection and outlines the example. Positive and negative tables rank vocabulary tokens at the **next-token prediction** after that position. Binary mode changes activation highlights, not the logit calculation.

Unlike a rank-one SAE with a fixed decoder vector, MOLT's output direction depends on the input. In the code's row-vector convention, for feature `i`:

```
x_norm = (pre_norm_input - saved_input_mean) / saved_input_std
contribution_i = gate_i * ((x_norm @ V_i) @ U_i) * saved_output_std
readout_i = contribution_i * final_norm_weight / baseline_final_rms
projected_logits_i = readout_i @ unembedding_weight.T
```

The output mean is shared across features and cancels when switching one transform off. The final RMS denominator comes from the original Qwen forward pass at the same token and remains frozen. Transform products and the readout use FP32 weights/arithmetic on the same BF16 standardized inputs and gates used by the activation dashboard. The collector verifies every gate against the cached observations before computing projections.

This is a **direct-path approximation**: it excludes the response of the remaining transformer layers and changes in final normalization. A positive score is not a guaranteed probability increase; probabilities also depend on the other logits. Actual causal effects would require subtracting the transform's raw output at layer 22, keeping the reconstruction error fixed, running the downstream layers again, and comparing the resulting logits with the baseline. The aggregate is an arithmetic mean over *all active tokens*, including observed gate magnitudes; context-dependent effects can cancel, so inspect individual examples as well.

This distinction follows the [SAE logit-readout discussion](https://www.transformer-circuits.pub/2023/monosemantic-features/index.html) and the use of frozen normalization in [attribution-graph methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).

To compute or refresh the projections, then render:

```bash
PYTHONPATH=. .venv/bin/python tools/compute_molt_dashboard_logits.py
.venv-dashboard/bin/python tools/render_molt_dashboard.py
```

`logit_projections.json` contains the method, formula, aggregate and example-level top/bottom vocabulary IDs and scores, plus decoded tokens. The default is ten vocabulary tokens per sign. No-activation features have no projected logit table. Tests in `tests/test_dashboard_logits.py` verify the raw-space on-minus-off contribution, the frozen-normalization logit difference, and sign filtering.

## Token steering

Run the inference server instead of the static-only server to enable steering:

```bash
PYTHONPATH=. .venv/bin/python tools/serve_molt_dashboard.py \
  --bind 100.76.58.1 --port 8765 --device cuda:0
```

Open http://100.76.58.1:8765/steer, or choose **Steer this transform** in the feature explorer. The server holds Qwen3-4B in GPU memory and handles one inference operation at a time. Its default bind address is loopback; use this machine's Tailscale address for tailnet access. The page requires this backend; a static file server cannot run generation.

1. Construct a conversation with system, user, assistant and thinking segments (or choose the single-prompt or raw-text editor), then inspect its tokenization/natural gates.
2. Select tokens individually or with Shift-click, choose a strength, and assign it. Different tokens can have different amounts. Chat-template tokens are visible and selectable. Editing the input invalidates the selection to prevent stale token positions.
3. Optionally specify a zero-based inclusive range of **consumed generated tokens**. Generated index 0 influences the prediction after the first response token. To steer the first response token, select the last prompt token. The last emitted token is not processed if generation ends there.
4. Compare the baseline and steered responses. Temperature 0 is greedy; sampling runs use independent generators initialized to the same seed. Stop cancels after the current forward pass. Download JSON to retain the request, checkpoint provenance, resolved strengths, output token IDs and intervention trace.

Strength semantics:

- **Set**, mean-active reference: `1×` sets the gate to the transform's mean positive gate over the held-out dashboard sample; `2×` doubles that reference. This can activate a naturally closed gate.
- **Add**, mean-active reference: `1×` adds that mean to the token's natural gate. Negative values suppress; the target is clamped at zero.
- **Multiply**: `1×` preserves the natural gate at this token, `2×` doubles it, and `0×` suppresses it. Multiplication cannot open a zero gate.
- Set/add also support **raw gate units**. Transforms never observed active require raw units because no active-mean reference is available.

For each selected input position, the backend computes:

```
delta_raw = (target_gate - natural_gate) * ((x_normalized @ V_i) @ U_i) * saved_output_std
Qwen_layer22_mlp_output += delta_raw
```

This preserves the original Qwen MLP output and the MOLT reconstruction error. It does not replace the complete MLP with MOLT. The output mean cancels in the difference. Transform arithmetic uses FP32 and the modified output is cast back to Qwen's BF16. Zero deltas bypass rewriting so a no-op is exact.

The backend runs all subsequent Qwen layers with the intervention and maintains the altered KV cache during decoding. **What changed** shows requested gate changes, raw-vector norms, and actual first-response-token logits through all downstream layers. Those measurements differ from the feature explorer's frozen-normalization direct projections. Once completions diverge, their later token contexts differ; the UI therefore reports a paired logit comparison only for the first prediction.

API: `POST /api/inspect`, `POST /api/generate`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel`. Generation requests include the tokenization fingerprint returned by inspection, the transform ID, mode, strength reference, token assignments and optional generated range. Inputs are limited to 2,048 tokens without silent truncation; outputs to 256 tokens. Browser POSTs must use same-origin JSON. Completed jobs are kept only in memory, with the most recent 16 retained.

Validation covers exact zero-addition generation, gate units, prompt-position isolation, generated-position alignment, hook cleanup, and the previous gate/logit calculations. Live Qwen tests additionally confirmed that `2×` the active-mean reference on a naturally closed transform changes final logits, while zero addition reproduces baseline token IDs and logits exactly.

When another browser or an automated sweep owns the steering server, `/api/inspect` and `/api/generate` return HTTP 409. This means **another steering operation holds the inference lock**, not that CUDA ran out of memory. The browser now displays a waiting state and retries those requests, preserving token assignments and previous results until a new generation is accepted. Stop cancels the local pending request; if cancellation races with acceptance, it cancels only that browser's newly accepted job. Validation errors are not retried. The page also polls server availability while idle.

`tests/browser/steering_busy.cjs` exercises busy inspection/generation, request preservation, cancellation and acceptance races using mocked API responses, without running any GPU inference. It requires Playwright and can use `STEERING_URL` to select the server URL.


### Conversation builder

The steering page supports arbitrary ordered system, user, assistant and raw blocks. Each turn contains editable text and/or explicit `<think>` segments. Turns can be reordered, duplicated and deleted; segments can be reordered or deleted. The top-level **+ Think** adds thinking to the final assistant turn or creates an assistant turn if necessary.

Choose where generation starts: a new assistant answer with an empty thinking section, a new assistant turn with no forced thinking choice, a new thinking segment, continuation of the final turn, continuation of an unfinished final assistant thought, or immediately after the final closed turn. For unfinished thinking, the last segment must be thinking inside an assistant turn. Use **Exact rendered input** to review the complete ChatML; **Edit this as raw text** allows unrestricted literal changes.

The browser explicitly serializes the conversation and submits it through the existing `format: "raw"` API. Historical thinking is preserved, even before later user turns; Qwen's ordinary chat template would filter some of that history. Raw blocks have no automatic role wrapper. The existing 2,048-token input limit still applies. Token indices cover the complete input, including role and thinking markers; any edit clears token assignments and requires inspection again.

Conversation JSON can be imported, exported or downloaded. The result download also includes the structured conversation alongside the exact rendered prompt. Browser tests in `tests/browser/steering_conversation.cjs` cover conversation editing, serialization, continuation, import/export, selection invalidation, and the submitted steering payload with all inference mocked.
