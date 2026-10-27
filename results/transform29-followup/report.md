# Transform 29: steering strength, reconstruction importance, and input formatting

**The concern is partly supported, and cannot be ruled out.** In this checkpoint, transform 29 contributes **30.5% of reconstruction benefit**, and rank-512 transforms together contribute **44.0%**. Transform 29 uses many output dimensions. However, the remaining transforms provide 69.5% of benefit; the results do not show that the MOLT's reconstruction is entirely carried by large transforms. We have not established a superiority claim over a matched transcoder or SAE baseline.

**Primary evaluation.** Qwen3-4B layer 22, checkpoint step 100,000, W&B `g2abz94y`. Random seed 20260908 selects 64 UltraChat `test_sft` conversations from indices 192 onward, avoiding the earlier exploratory sample. The primary evaluation matches the old training prefix and effective maximum length: **58,381 tokens**, max 1,023 tokens, including the prepended `<|endoftext|>`. Saved input/output standardizers and BF16 gates are used; transform products and error accounting use FP32. The original Qwen MLP's raw output is the target.

- Standardized MSE: **0.260319**; final logged training MSE: 0.259388.
- Saved-mean-only predictor MSE: **1.014554**; explained reduction relative to that baseline: **74.34%**.
- Gate L0: **2.9557**; mean sum of active transform ranks: **699.88**.
- Raw-output MSE: **0.063204**. Standardized and raw MSE are distinct quantities.

The rank sum measures activated factor capacity, not an observed number of independent dimensions. Nevertheless, an active rank-512 MOLT gate is not the same object as one active scalar SAE/transcoder feature: the former multiplies an input-dependent vector with potentially hundreds of degrees of freedom. Comparing gate L0 alone misses this difference. No fair cross-architecture performance comparison was performed here.

**Importance definitions.** For standardized target `y`, transform contributions `c_i`, prediction `p = sum_i c_i`, and residual `r = p - y`:

```
individual removal loss increase = ||c_i||² - 2 <r, c_i>
additive benefit allocation_i    = <c_i, 2y - p>
```

The allocations sum exactly, up to rounding, to `||y||² - ||y-p||²`. This is the Shapley allocation for squared-error improvement in a game where the fixed transform contributions are switched on or off; pairwise interactions are divided equally. It is an accounting choice, not a measurement of semantics or causal language-model importance. Group removal uses the summed group output and includes interactions within that group. Individual removal scores are not additive, so their percentages must not be interpreted as shares of one total.

| Rank | Transforms | Transform parameters | Benefit share (95% conversation bootstrap) | MSE increase if group removed |
|---|---:|---:|---:|---:|
| 512 | 80 | 20% | 44.0% (42.9%–45.2%) | +0.3145 |
| 256 | 160 | 20% | 9.5% (8.7%–10.5%) | +0.0605 |
| 128 | 320 | 20% | 9.5% (9.0%–10.0%) | +0.0587 |
| 64 | 640 | 20% | 15.7% (15.0%–16.4%) | +0.0997 |
| 32 | 1280 | 20% | 21.3% (20.7%–21.9%) | +0.1386 |


Each rank group has the same factor-parameter budget because its transform count doubles as rank halves. Encoder parameters are excluded from the 20% figures. The bootstrap resamples conversations, not individual correlated tokens; it describes uncertainty within this dataset/sample scheme, not across training seeds.

**Transform 29 specifically.** It activates on **75.31%** of tokens in the new random sample. Its additive benefit share is **30.4945%**. Removing it raises total MSE from **0.2603 to 0.4818**, an increase of **0.2215**. It holds approximately **0.25% of the transform-factor parameters** (one of 80 transforms in a group holding 20%), so this is a frequently used, parameter-efficient common computation as well as a potential interpretability bottleneck.

The next largest transform, ID 23, contributes only **3.03%**. The top ten together contribute **41.53%**. Subtracting transform 29's allocation from the accounting, the remaining rank-512 transforms contribute **19.45% of the remaining benefit**. This is an accounting renormalization, not a new coalition/Shapley evaluation after physically removing transform 29. Thus the rank-512 concentration is largely one exceptional transform, not evidence that every large transform dominates. Ranks 64 and 32 together provide **36.98%** of benefit.

**Does transform 29 really need many dimensions?** On the first eight primary-evaluation conversations (7,134 tokens), I truncated the SVD of its learned `V29 @ U29` matrix in standardized output coordinates, keeping the gate and every other transform fixed. This is post-hoc truncation without retraining. It is not optimized for the empirical input covariance, and a retrained lower-rank model could perform differently.

| Retained rank of transform 29 | Total MSE | Retained transform-output energy |
|---|---:|---:|
| 0 | 0.4844 | 0.0% |
| 1 | 0.4836 | 0.3% |
| 8 | 0.4745 | 4.5% |
| 32 | 0.4521 | 14.6% |
| 64 | 0.4249 | 26.9% |
| 128 | 0.3860 | 44.6% |
| 256 | 0.3337 | 68.4% |
| 512 | 0.2639 | 100.0% |


Rank 32 retains only **14.6%** of observed output energy under this matrix-SVD truncation and raises total MSE to **0.4521**. Rank 1 is almost as damaging as removing the transform entirely. Separately, the uncentered SVD of 1,024 observed active output vectors requires **209 components** to capture 90% of energy; the first component captures **7.0%**, and mean pairwise output cosine is **0.055**. These observations support a multidimensional computation. Neither rank nor output dimensionality alone proves that the computation is uninterpretable; it could still implement an intelligible algorithm that we have not identified.

**Steering: 53 comparisons.** There are 39 native-format runs across seven prompts and 14 checks across three prompts with the training BOS prefix. Each run compares greedy baseline and intervention, up to 64 new tokens. All complete outputs and token-level intervention traces are in the JSON artifacts and interactive viewer.

For naturally active prompts, doses are multipliers of the current natural gate: `0, 0.25, 0.5, 1, 2`. For inactive prompts, doses set the gate to `dose × 1.63174`, the earlier sample's mean positive activation; the sweep reaches 4× that mean, or **6.527**. That exceeds the previous sample maximum of 2.3125. Strong-dose effects therefore describe extrapolation, not naturally observed feature operation.

Prompt assignments target only naturally active positions for attenuation or naturally inactive positions for forced activation. Generated-token assignments apply to all consumed generated positions: multiplication preserves naturally zero gates, while set-mode can also change any positions that would subsequently be active. This distinction matters when interpreting low set-mode doses. The chat-boundary experiment changes only the final prompt position and leaves generated tokens unmodified. All interventions preserve the original MLP reconstruction error and run the remaining Qwen layers; they are not frozen direct logit projections.

- **Active prose/reasoning:** reducing or removing the feature generally preserves coherent continuation and changes wording or formatting. Some 64-token outputs are identical even after removal. One story stays identical at 0× through 1× and changes plot details at 2×. There is no reliable semantic switch.
- **Inactive code:** at low doses the continuation often stays identical. Around the natural mean it can add a division-by-zero guard. At 4×, it degenerates into repetitive comments in native formatting, or repeated blank lines in the training-prefix check. This does not consistently turn code into meaningful prose.
- **Inactive counting:** 0–1× preserves the ascending sequence. At 2×, counting repeats or reverses; with the training prefix it produces `11 10 9 8 ...`. At 4× it becomes mixed/repetitive text, including repeated `10` with the training prefix.
- **Inactive repeated “dog”:** the baseline is already degenerate, making this a weak interpretability probe. Strong forcing changes it into repetitive parentheses rather than restoring natural language.
- **A single chat-boundary token:** on `17 + 25`, a 2× setting changes the answer to `32`, but 4× returns `42`. The response is nonmonotonic and uses a first-context convention different from MOLT training. It demonstrates disruption, not a clean arithmetic or meaning feature.

The training-prefix checks reproduce the broad conclusion: strong forced activation disrupts structured continuations, while active prose does not exhibit a stable, interpretable semantic dose response. These are small qualitative samples, not capability benchmarks; generated outputs often stop at the length cap.

**The first-token discrepancy and its resolution.** The original native-template 512-token evaluation gave MSE **0.5019**. Its first positions averaged **123.47** MSE and contributed **48.9% of all squared error**, although the first-token mean-only baseline was only 0.0968. Excluding those 64 positions gave MSE **0.2568**. BF16 versus FP32 product arithmetic did not explain the discrepancy: on a four-conversation diagnostic, the MSEs were 0.48676 and 0.48680, respectively.

The old activation generator tokenized the chat, allocated `max_sequence_length - 1` slots, rolled the batch right by one, and wrote the model-config BOS into position zero. It left the validity mask unchanged. Consequently Qwen training began with `<|endoftext|><|im_start|>...`, while dashboard samples began with `<|im_start|>...`. It did not simply exclude position zero. Matching that training preprocessing restored MSE **0.2603 with position zero included**, so the misleading negative rank-group results from the native-prefix outlier are not used for the main conclusions. The boundary result is primarily evidence of a preprocessing mismatch, not evidence that the rank-128 group is intrinsically harmful.

**Correct token formatting and implemented fix.** Locally verified tokenizer outputs are:

| Model | Native chat start | Manual BOS insertion |
|---|---|---|
| Qwen3-4B | `<|im_start|>user...` | None |
| Gemma-3-4B-IT | `<bos><start_of_turn>user...` | None; BOS is already emitted |

For plain text, Gemma's tokenizer also adds BOS by default; Qwen's does not. Use the tokenizer/template output directly. This follows the [official Qwen workflow](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/quickstart.md) and [Gemma formatting guidance](https://ai.google.dev/gemma/docs/core/prompt-structure), and was checked against the locally cached tokenizers for these exact model IDs.

At the user's request, `generation_loop.py` now passes token IDs through without rolling or injecting BOS and uses the full configured length. The restart path reuses the same loader setup. `TextDataset` documents authoritative tokenizer output and rejects non-callable chat formatters. Plain-text tokenizer output is also preserved. Tests cover both cached model templates, padding, truncation, the last retained token, and loader restart. **24 tests passed** across tokenization regression tests, MOLT tests, and datamodule teardown. Running training processes were not restarted, and no checkpoint weights were changed. The old-checkpoint evaluation scripts explicitly reproduce historical preprocessing; they do not depend on the refactored generator.

**What is established, and what remains open.** There is a real concentration of reconstruction benefit in a broad, multidimensional transform, and raw gate L0 understates its internal capacity. These results do not justify dismissing MOLT reconstruction quality: most benefit still comes from other transforms, and lower-rank groups contribute materially. They also do not establish that the large transform lacks an interpretable algorithm, or that an SAE/transcoder would expose its computation more clearly.

To evaluate the claimed advantage over other architectures, we still need trained baselines on the same model, layer, input/output locations, tokenizer formatting, data, and evaluation sample, with parameter, compute, and active-rank budgets reported alongside gate L0. A useful next controlled training comparison is mixed-rank MOLT versus a rank-capped MOLT and a scalar transcoder, plus a shared linear/background component with a sparse residual model. Reconstruction, downstream causal fidelity, and independent interpretability evaluation should be reported separately. No comparable scalar checkpoint or retrained rank-capped baseline was evaluated here; post-hoc truncation is not a substitute for that comparison.

**Artifacts.** `index.html` provides dose selection and a searchable importance table; `feature_importance.csv` ranks all 2,480 transforms. Primary results: `bos_matched/reconstruction.json`; sensitivity analyses: `reconstruction.json`, `reconstruction_interior.json`, `diagnostics.json`; rank cap: `rank_cap.json`; steering: `steering_sweep.json`, `bos_steering_sweep.json`. Scripts and extracted `xy.pt` tensors are retained for reproduction. Run scripts from the repository root with `PYTHONPATH=.` and `.venv/bin/python`. The steering script uses the existing local service at port 8765. The primary reconstruction command sets `MOLT_INV_OUTPUT=results/transform29-followup/bos_matched MOLT_INV_LENGTH=1023 MOLT_INV_BOS=1`.
