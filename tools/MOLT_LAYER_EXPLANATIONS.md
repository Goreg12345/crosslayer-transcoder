# Layer 22: all-transform Delphi run

Georg approved expansion from the completed 32-transform pilot to **every
transform in layer 22**, using the existing **10,000,000-token** sample. The
requested methods are detection, fuzzing, active fuzzing, and intruder.
**Simulation is excluded.** Georg subsequently approved the
[other campaign layers](MOLT_INTERPRETATION_CAMPAIGN.md): their caches are
collected now and their API work is queued behind this layer.

Results: <http://100.76.58.1:8765/sample-10m/autointerp-layer/>

The shared cache already contains activations for all 2,480 transforms across
11,345 UltraChat `test_sft` conversations. It is the campaign layer-22 checkpoint
at step 100,000. There is no new GPU extraction pass: the activation explorer and
the Delphi pipeline use the same cache.

- 1,203 transforms were observed; 1,277 have no positive evidence.
- 1,087 observed transforms have the full evidence budget.
- 116 observed transforms have fewer examples or short contexts.
- The 32 approved pilot explanations, LLM titles, and four relevant scores are
  reused with provenance; the original pilot, including its simulations, remains
  archived under `pilot32/`.

## Evidence and scores

The pipeline requests an explanation of up to 80 words through Delphi's
DefaultExplainer, then asks the same endpoint for a concise 3–8 word title based
on that explanation. The separate title does not add claims. Both are saved in
the feature result; scorers use the full explanation, not the title.

Supported transforms use 32 explanation windows, 50 held-out positive windows,
and 50 held-out zero-gate windows. Detection, fuzz, and active fuzz normally have
100 decisions each. Intruder uses 20 groups of four positive contexts and one
negative. All use the pilot's prompts and scorer adapters. Intruder is a
coherence measure that does not receive the explanation.

The seed-42 conversation split, 64-token context cap, peak-ending windows, and
exact token-window deduplication remain in use. For rare transforms, the
sampler retains available short or fully active windows rather than inventing
evidence. Detection and ordinary fuzz use balanced positive/negative counts;
active fuzz requires zero-gate distractor tokens in the positive context.
Intruder requires at least four distinct held-out positive contexts and always
keeps five alternatives. Unsupported methods are `unavailable`, not zero scores.

Transform 1336 has one observed token, only in the evaluation partition. Its
conversation is reassigned exclusively to explanation, with no evaluation on
that conversation. This exception is recorded. Every observed transform has
some explanation evidence; every unobserved transform has an explicit record
without an inferred explanation.

`complete` means the full evidence budget and four valid scores;
`limited_evidence` means all supported work is done but sample counts are smaller
or a method is unavailable. API/parse failures remain `incomplete` or `error`.
The overall run can finish while retaining honest limited-evidence and
not-observed records.

Interpretation limitations from the pilot still apply: the same model explains
and judges; random negative passages are not semantically matched; peak-ending
windows and upstream intruder highlights provide possible position cues; paired
active-fuzz and reused intruder contexts make confidence intervals approximate.
These labels describe gate activation, not causal transform output effects.

## Operation and resume

The endpoint is `https://geor.taild80f17.ts.net/v1`, model
`qwen3.8-27b-nvfp4`, with 32 concurrent requests and a 10,000-token total context
limit. `vLLM_API_KEY` is loaded through systemd's `EnvironmentFile` from `.env`;
credentials are never placed in command arguments or prompt audits.

- `molt-autointerp-layer22.service`: explanation/title phase, then four scorers.
- `molt-autointerp-layer22-review.service`: live review pages and explorer
  publication initially, after titles, and after scoring.
- Logs: `logs/molt-autointerp-layer22.log` and
  `logs/molt-autointerp-layer22-review.log`.
- Output: `results/molt-qwen-dashboard-campaign-layer22-10m/autointerp-layer/`.

The worker and renderer use no local CUDA devices and each has an 8 GiB process
memory cap. Training and the PRO 6000 steering server continue separately.
Transient services survive terminal disconnects, but not a machine reboot.

The manifest pins cache, checkpoint, model, sample policy, and methods. Evidence
is prepared before remote calls. Each feature and method is checkpointed; a
compact progress file updates every 15 seconds. A process lock prevents duplicate
workers. Valid responses to identical prompts can be replayed. Up to three
passes retry only missing/invalid work; valid poor scores are never replaced.

To resume with the API key already in the environment, after checking that no
worker service is active:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. local/autointerp-venv/bin/python \
  tools/run_molt_layer_explanations.py \
  results/molt-qwen-dashboard-campaign-layer22-10m
```

Render once or watch live progress:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python tools/render_molt_layer.py \
  results/molt-qwen-dashboard-campaign-layer22-10m --watch --publish-explorer
```

`features/<id>.json` holds the title, hypothesis, all four methods' metrics and
individual decisions. `audit/<id>.jsonl` records exact prompts and responses
without credentials. `summary.json` omits the large per-case arrays for efficient
live progress. The explorer consumes shared `autointerp/summary.json` annotations
and refreshes sidebar titles without reloading model weights.

Validation completed for all **153,866 evidence windows**: 37,349 explanation,
56,417 positive evaluation, and 60,100 negative evaluation windows. The audit
checked exact cached token IDs, display tokens, gate values, conversation bounds,
peak positions, disjoint conversations, deduplication, and zero-gate negatives.
The report is `autointerp-layer/validation.json`. Browser checks cover all 2,480
rows, observed/unobserved/limited filters, the four-score detail view, saved
review notes, and the shared explorer's 1,203 searchable titles and review links.
