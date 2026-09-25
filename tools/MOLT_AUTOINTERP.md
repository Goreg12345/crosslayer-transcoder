# MOLT gate explanations with Delphi

The label target is the scalar JumpReLU gate of each transform: when it fires.
A MOLT transform has an input-dependent output, so these descriptions do not
establish its downstream causal effect. Direct logit projections and steering
can support a separate analysis of output effects.

The pipeline uses EleutherAI [Delphi](https://github.com/EleutherAI/delphi), pinned
to `4fea06e6e8b68eeaf302474325fca13df95c5d6f`, with its `DefaultExplainer` and
`DetectionScorer`. It adapts our cached gates directly into `LatentRecord`s;
neither Qwen inference nor checkpoint conversion is needed for explanation.

Setup creates `local/autointerp-venv` and `local/delphi`. The environment reads
the project's existing packages but installs additional dependencies only into
the overlay. Three upstream package `__init__.py` files are changed to lazy
re-exports to avoid importing vLLM, FAISS, and embedding models for remote API
work. Explainer and scorer implementations remain unchanged. Our API adapter
adds literal chat-token escaping, concise gate-label instructions, strict
one-decision scorer output validation, retries, and bounded concurrency.

```bash
bash tools/setup_molt_autointerp.sh
set -a
source .env
set +a
CUDA_VISIBLE_DEVICES= PYTHONPATH=. local/autointerp-venv/bin/python \
  tools/explain_molt_dashboard.py results/molt-qwen-dashboard-campaign-layer22 \
  --base-url https://geor.taild80f17.ts.net/v1 \
  --model qwen3.8-27b-nvfp4 --api-key-env vLLM_API_KEY \
  --limit 0 --concurrency 32 --context-limit 10000
.venv-dashboard/bin/python tools/render_molt_dashboard.py \
  results/molt-qwen-dashboard-campaign-layer22
```

The endpoint must accept OpenAI-compatible chat completions and the vLLM
`chat_template_kwargs.enable_thinking=false` extension. Credentials are read
from the named environment variable and never saved in outputs.

Use `--limit 0` for all eligible transforms. Existing successful feature files
are reused only if the configuration fingerprint matches. Change `--output`
for a new experiment; the renderer reads the default `autointerp/summary.json`.
Use `--limit 20` for a pilot. Progress summaries are saved every 32 completed
transforms, and each individual result is saved as soon as its scoring finishes.
The summary's `usage` counts the current invocation, including retries with
reported usage; existing labels are reused without additional endpoint calls.

The HTTP pool and shared semaphore permit at most 32 concurrent requests by
default, with a 600-second response timeout for batch serving. The 10,000-token
context budget includes the reserved completion. Prompts are screened using a
conservative UTF-8 byte bound with chat framing allowance. For borderline
prompts, the server's exact token count is obtained from a one-token completion
because the endpoint proxy does not expose tokenization routes. Over-budget
prompts fail without silently removing evidence. Runtime request concurrency,
maximum observed prompt size, context probes, and retries are recorded.

The full-layer summary also reports live `in_flight`, `waiting_for_slot`, and
`concurrency_limit`, plus cumulative `in_flight_seconds_total` and
`measurement_seconds`. Divide the differences of those last two values between
snapshots to measure average outstanding HTTP requests during a window; the
lifetime peak alone does not demonstrate sustained occupancy. Completed-request
latency totals/counts and HTTP failure counts are included. These describe client
requests, including any time queued at the proxy or inference server, rather than
the number of sequences actually executing on remote GPUs. Usage counters reset
when the worker resumes, so compare snapshots from the same invocation. Input
throughput counts submitted prompt tokens, including prefixes the backend may cache.

## Evidence and evaluation

- Split conversations 75/25, with seed 42, before selecting examples. No
  conversation appears in both the explanation and evaluation partitions.
- Use up to twelve positive examples, one peak per conversation, combining
  strong and weaker examples. Each example contains at most 64 tokens and ends
  at its peak; no subsequent tokens are visible to the explainer.
- Normalize displayed gate strengths using the explanation examples only.
  Keep original gate values in the saved evidence.
- Score each explanation on five held-out positive windows and five zero-gate
  windows, with one classification per request. These are random negatives,
  not semantically matched hard negatives. Scores from ten examples are pilot
  measurements, not well-calibrated confidence estimates.
- Preserve chat/special tokens as evidence, displayed as `⟪im_end⟫` etc. so they
  remain literal text rather than endpoint conversation delimiters.
- Transforms with no activation are `not_observed`; those without enough
  independent examples are `insufficient_evidence`. Neither status means a
  transform is permanently dead. Others remain `pending` until processed.

Outputs include checkpoint/cache provenance, model, Delphi revision, prompts'
sampling settings, per-transform evidence and predictions, detection metrics,
and token usage. The dashboard displays generated labels as provisional, with
their held-out detection counts, and supports searching explanation text.

The initial layer-22 dashboard cache has 110,066 tokens in 128 conversations:
1,137/2,480 transforms fire at least once; 1,343 are unobserved. This is enough
for a pilot, but a larger and more diverse corpus is needed for broader coverage.
Delphi's README recommends at least 10 million cached tokens for its workflow.

The September 18, 2026 layer-22 batch completed all 826 eligible transforms
(806 new labels plus the 20 pilot labels), using 32 concurrent remote requests.
All 8,866 new requests succeeded without retries; the largest observed prompt
was 2,646 tokens. The remaining 311 observed transforms lack sufficient examples
for this scoring setup, and 1,343 were unobserved. Held-out detection was correct
on 5,621/8,260 examples (68.1%); these provisional hypotheses vary in quality.
The full run statistics are saved in the dashboard's
`autointerp/batch32-report.json`.

## Alternatives considered

- [Neuronpedia's max-activation/logit explainer](https://www.neuronpedia.org/blog/circuit-tracer)
  produces concise descriptions using activation examples, top logits, and
  following tokens. Its approach is useful for a separate output-effect label;
  MOLT requires context-dependent projections rather than a fixed decoder vector.
- [SAEBench autointerp](https://github.com/adamkarvonen/SAEBench/tree/main/sae_bench/evals/autointerp)
  is useful when standardized evaluation across SAE configurations is the aim.
- [OpenAI's original neuron explainer](https://github.com/openai/automated-interpretability)
  implements explanation and activation simulation. Its original repository is
  archived; Delphi is the more direct fit for this new pipeline.
- A small custom explainer can consume the same cache and endpoint, but would
  require maintaining the prompting and scoring methodology ourselves.

Validation: `PYTHONPATH=. .venv/bin/pytest -q tests/test_explanation_samples.py`.
Transport validation (mocked; no endpoint calls):
`PYTHONPATH=. local/autointerp-venv/bin/python -m pytest -q tests/test_autointerp_endpoint.py`.

## Ten-million-token sample

The expanded layer-22 sample is collected separately in
`results/molt-qwen-dashboard-campaign-layer22-10m`, from the same 100,000-step
campaign checkpoint and the held-out `test_sft` split. Its target is 10,000,000
non-padding tokens (90.9 times the initial sample). Conversations remain separate,
with a 1,024-token cap; only the last conversation may be shortened further to
meet the exact total. Dataset chat/special tokens remain included.

The collector removes transformer layers after layer 22 and its unused MLP
before transferring weights to the GPU. The completed job used batches of 16,
a 7.5 GiB PyTorch allocator cap, and the RTX 5090 UUID. It saves atomic shards
every 128 conversations. Repeating the identical command resumes from the last
committed shard; changed checkpoint or sampling settings are rejected. The GPU
cap covers the allocator, not driver/context overhead. Observed reserved CUDA
memory during collection is about 6.84 GiB, plus driver/context overhead.

```bash
CUDA_VISIBLE_DEVICES=GPU-2ada3848-a497-abbf-c0da-c053a9680677 PYTHONPATH=. \
  .venv/bin/python tools/collect_molt_dashboard.py \
  --checkpoint checkpoints/molt-qwen3-4b/layer-22/N80-rank512to32-lr4e-5-sp7.5e-5-native-ramp80/training.ckpt \
  --output results/molt-qwen-dashboard-campaign-layer22-10m \
  --tokens 10000000 --batch-size 16 --shard-conversations 128 --gpu-memory-gib 7.5
```

Batch size is recorded in collection provenance. BF16 batched extraction can
change gates near their thresholds compared with single-conversation extraction;
the new cache and its labels have their own fingerprint. On a 54,828-token check,
token IDs and conversation boundaries exactly matched the original sample,
active-position/feature sets had 99.49% Jaccard overlap, and shared active gates
had mean absolute difference 0.006. No old labels are reused for the new cache.

The CPU explanation preparation indexes sparse activations once, then selects
conversation peaks before decoding windows. It materializes only the chosen
examples and uses binary searches for window activations. This avoids scanning
the whole cache and constructing thousands of unused text windows per feature.

Collection is complete: 10,000,000 tokens across 11,345 conversations.
The [32-transform quality pilot](MOLT_PILOT.md) is complete and Georg approved
[all transforms in layer 22](MOLT_LAYER_EXPLANATIONS.md). The current run generates
titles and explanations, then detection, fuzz, active fuzz, and intruder scores,
with simulation excluded. Use `run_molt_layer_explanations.py` and its full-layer
review renderer; the older detection-only wrapper below is a legacy workflow.
The [remaining campaign layers](MOLT_INTERPRETATION_CAMPAIGN.md) are now also
authorized, with their API work queued behind layer 22. Credentials are loaded through systemd's
`EnvironmentFile` and do not appear in command lines or logs.

View collection progress, then the expanded explorer, at
[the 10M sample route](http://100.76.58.1:8765/sample-10m/).
The original dashboard and steering server remain running on the original
sample. Its direct logit projections remain available there. The expanded view
contains gate statistics and regenerated labels; new direct logit projections
are a separate computation and are explicitly marked unavailable. Steering
links use the existing engine and its original sample's mean-active reference.
No training processes or dependencies were changed for collection.

Progress files and logs:

- `results/molt-qwen-dashboard-campaign-layer22-10m/collection-progress.json`
- `results/molt-qwen-dashboard-campaign-layer22-10m/pilot32/summary.json`
- `results/molt-qwen-dashboard-campaign-layer22-10m/autointerp/summary.json` (published after the pilot)
- `logs/molt-collect-layer22-10m.log`
- `logs/molt-pilot32-layer22.log`

Validation includes restart after an interrupted shard write, provenance
mismatch rejection, indexed sampling equivalence, bounded example decoding,
request concurrency/context tests, and a complete local render/publish pipeline
check on eight conversations. The latter has no eligible held-out scoring
features and makes zero endpoint requests.
