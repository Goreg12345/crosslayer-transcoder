# Layer 22: fixed 32-transform explanation pilot

This completed **32-transform pilot is archived**. Georg reviewed it and approved
[expansion to all transforms in layer 22](MOLT_LAYER_EXPLANATIONS.md), with titles
and detection, fuzz, active fuzz, and intruder scores. The expanded run excludes
simulation. The [remaining campaign layers](MOLT_INTERPRETATION_CAMPAIGN.md) have
also been authorized and are queued after the full layer-22 run.

- Review: <http://100.76.58.1:8765/sample-10m/pilot32/>
- Shared activation explorer: <http://100.76.58.1:8765/sample-10m/>
- Data: `results/molt-qwen-dashboard-campaign-layer22-10m/pilot32/`

## Evidence and selection

The shared cache contains exactly 10,000,000 non-padding tokens from 11,345
UltraChat `test_sft` conversations, capped at 1,024 tokens per conversation.
It uses the campaign layer-22 checkpoint at step 100,000. Both the explorer and
the explainer read this cache; scoring does not run Qwen on a local GPU.

Selection was frozen before LLM calls: seven transforms of rank 512, seven of
rank 256, and six each of ranks 128, 64, and 32. Within each rank, frequency
quantile bins and seed 42 spread the selection across supported firing rates.
Transforms need sufficient distinct conversations; this pilot does not evaluate
very rare or unobserved transforms. `selection.json` records eligibility and IDs.

Each transform has 32 explanation windows, 50 held-out positive windows, and
50 held-out zero-gate windows. A seeded 75/25 conversation split separates
explanation and evaluation. Windows contain 16–64 tokens, with exact token-window
deduplication across all three sets. Positive windows end at a conversation's
peak; training mixes eight strong examples with weaker examples. No hypothesis
or cohort is adjusted in response to evaluation scores.

## Explanation and scoring

The endpoint is `https://geor.taild80f17.ts.net/v1`, model
`qwen3.8-27b-nvfp4`, with 32 concurrent requests and a 10,000-token total context
budget. Credentials come from `vLLM_API_KEY` through systemd's `EnvironmentFile`.
Delphi's pinned revision is recorded in `manifest.json`. Its DefaultExplainer
is asked for a hypothesis of up to 80 words; a separate short label is for browsing.
Outputs are retained without truncation, including occasional over-length answers.
The scorers receive the full hypothesis where applicable.

| Method | Per transform | What it measures |
| --- | --- | --- |
| Detection | 50 positive + 50 negative decisions | Can the explanation distinguish firing contexts? |
| Fuzz | 50 correct + 50 incorrect highlights | Does the explanation identify the firing tokens? |
| Active fuzz | 50 correct + 50 incorrect highlights | Same test with incorrect highlights inside positive contexts |
| Intruder | 20 groups, four positive contexts and one negative | Context coherence, without using the explanation; chance 20% |
| Simulation | 10 positive + 10 negative windows | Token activation prediction, normalized to 0–10 using the explanation set's maximum |

Detection, classification prompts, intruder scoring, and the JSON simulator use
Delphi. `AuditedFuzzingScorer` replaces its distractor sampler with seeded,
strictly zero-gate positions, preserving the classification prompt. This avoids
fixed-position, weakly active, and unrelated fallback distractors.
`StrictNeuronSimulator` rejects invalid token counts, identities, and values
instead of accepting the upstream wrapper's silent all-zero fallback. Literal
thinking delimiters are displayed with Unicode brackets one-to-one to prevent
the endpoint's reasoning parser from consuming them. A single JSON code fence
is accepted; misaligned outputs are not repaired.

Binary scores include confusion matrices, accuracy, balanced accuracy, F1, and
approximate Wilson intervals. Simulation reports Pearson correlation, MSE,
R-squared, and improvement over an all-zero predictor. Invalid responses remain
visible and are excluded from valid-only metrics; a method is incomplete until
all its cases have valid responses. Negative scores and poor explanations are
retained. Classification is scheduled before the longer simulation requests.

Limitations: the same model explains and judges; random negatives are not
semantically matched; supported-feature selection omits rare gates. Positive
windows end at their peak, but token judgments can see later tokens within that
prefix. Paired active-fuzz cases and reused intruder contexts make decisions
dependent, so confidence intervals are approximate. These are gate-trigger
explanations, not causal claims about transform outputs. Embedding and surprisal
scorers require an embedding backend or teacher-forced model scoring and are
not included in this endpoint-only pilot.

Because positive windows end at their peak, highlight-based scorers can exploit
token-position cues. The unmodified Delphi intruder sampler also includes a
fixed three-quarter-position distractor highlight. Treat these scores as
exploratory, alongside detection and manual case inspection, rather than a
clean causal comparison of explanation quality.

## Review, audit, and resume

The review pages show hypotheses, training evidence, every scored case,
incorrect predictions, and simulation heatmaps. Notes and Good/Mixed/Poor
verdicts persist in the browser; **Export review JSON** saves all 32 verdicts.
Exporting a review does not launch anything.

`features/<id>.json` checkpoints each completed method. `audit/<id>.jsonl`
contains actual adapted prompts, responses, completion reasons, and request
usage where available, without API keys. Exact task/prompt/output-budget keys
allow validated responses from interrupted methods to be replayed. The first
valid response is retained regardless of accuracy. Checkpoint, cache hash,
cohort, model, and settings are fingerprinted; changed configurations fail.
`validation.json` records the completed audit of all 4,224 windows against the
10M cache, including token/value equality, boundaries, disjoint conversations,
deduplication, peak placement, and zero-gate negatives.

Services: `molt-pilot32-layer22.service` performs remote scoring and
`molt-pilot32-review.service` refreshes the review pages. These transient units
survive terminal disconnects but do not restart across a reboot. The main
`molt-dashboard-layer22.service` serves on the RTX PRO 6000 with a 10 GiB
allocator cap. Local training remains active.

`molt-pilot32-finish.service` runs `finish_molt_pilot.sh`: it waits for the active
scorer to exit, makes at most two resume passes for invalid/missing responses,
then renders both the review and activation pages. It preserves all completed
scores and exits with a failure status if any cases are still incomplete.

To resume this same pilot with the key already in the environment:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. local/autointerp-venv/bin/python \
  tools/run_molt_pilot.py results/molt-qwen-dashboard-campaign-layer22-10m
```

Do not launch a duplicate process while the pilot service is active. The worker
publishes only these 32 labels into `autointerp/summary.json` after scoring.
Render the review and shared explorer after it exits:

```bash
PYTHONPATH=. .venv/bin/python tools/render_molt_pilot.py \
  results/molt-qwen-dashboard-campaign-layer22-10m
PYTHONPATH=. .venv-dashboard/bin/python tools/render_molt_dashboard.py \
  results/molt-qwen-dashboard-campaign-layer22-10m
```

Tests cover strict simulation validation, response replay, audited fuzz sampling,
sampling/deduplication, metrics, endpoint context limits, and concurrency. The
browser check is `tests/browser/pilot_review.cjs` and exercises navigation,
filters, evidence, saved review notes, and JSON export without inference calls.
