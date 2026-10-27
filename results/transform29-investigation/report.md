# Transform 29 investigation

**Best working label: “varied word-like text, away from boundaries and repetitive/structured sequences.”** This is a moderately interpretable routing gate for a broad transform. The evidence rules out an assistant-role feature, a specific topic, and a detector of semantic coherence. It does not establish a unique computational purpose for the transform's output.

**Run identification.** W&B run `g2abz94y`, named `molt-qwen3-4b-chat-layer22-N80-lr4e-5`, checkpoint `checkpoints/molt-qwen3-4b/layer-22/chat-ultrachat-lr4e-5/training.ckpt`, step 100,000, saved 2026-09-07 23:03:42 UTC (September 8 00:03:42 London). Model: Qwen/Qwen3-4B; layer index 22. Transform 29 is the zero-based encoder row used in the dashboard and has rank 512.

The cached held-out L0 is **2.9527**, with transform 29 active on **87,892 / 110,066 tokens = 79.8539%**. This matches the identifying feature frequency exactly. The final available training log entry is step 99,949, L0 **2.9650**, `training/mse` **0.25939**; the last ten logged MSE values average **0.25715**. Thus the MSE is close to, but not exactly, the remembered 0.24. I extracted these values directly from the local W&B binary history; I did not rerun full-model MSE evaluation.

**Dataset evidence.** Exploratory data: first 128 UltraChat `test_sft` conversations, each truncated to 1,024 tokens. Replication: the next 64 conversations, 57,744 tokens, collected after forming the hypotheses. Total: **192 conversations / 167,810 tokens**. Counts below are token counts, not independent statistical observations. Active means gate strictly greater than zero; padding is excluded.

| Token category | Original 128 conversations | Fresh 64 conversations |
|---|---:|---:|
| Assistant content | 83.58% (80,480) | 81.93% (43,879) |
| User content | 78.39% (26,311) | 78.88% (12,092) |
| Role headers | 0.00% (1,362) | 0.00% (740) |
| Chat delimiters | 0.00% (1,297) | 0.00% (702) |
| Content tokens containing letters | 92.42% (90,681) | 91.89% (47,024) |
| Punctuation-only content tokens | 33.06% (12,126) | 33.35% (6,813) |
| Whitespace-only content tokens | 0.00% (1,460) | 0.12% (850) |


The fresh overall firing rate is **45,490 / 57,744 = 78.7787%**. The original sample's 681 observed turns begin with activation rates of **1.03%, 16.89%, 34.07%, 44.35%, 59.32%** at content positions 0–4; by position 10 it is **85.84%**. Boundary suppression is strong, but not an exact universal rule. A simple distance measure after any period, exclamation mark, question mark, or newline shows the same recovery pattern; this is a character heuristic, not a syntactic sentence parser.

In the original sample, standalone `.` activates on only **5 / 3,403** occurrences, `?` on **0 / 287**, `!` on **0 / 192**, and `:` on **0 / 515**. Punctuation is not uniformly excluded: commas, for example, can activate. Headers and inter-turn separator newlines are off. The fresh sample has one active whitespace token, so “never active on whitespace” would overstate the result.

One actual user question begins:

`How[0] does[0] the[0.96] author[0.98] propose[1.42] to[1.70] fix[1.65] ... system[1.62] ?[0] What[0] changes[1.66] ...`

The feature is clearly processing words inside a user question. Additional randomly sampled inactive-word examples include turn starts such as `How` and `Write`, sentence starts such as `Here`, and code identifiers. There are also exceptions deep inside ordinary prose: this is not a perfect punctuation/position classifier. See `cache_analysis.json` for the reproducibly sampled contexts.

**Controlled hypothesis tests.** Identical prose under user, assistant, and system headers has identical binary activation masks: **22 / 26 = 84.6%**. Identical question text gives **22 / 27 = 81.5%** in all three roles. Gate strengths change slightly. Bare text also activates strongly (88.5% for the prose probe), so a chat wrapper is unnecessary. The role experiment is small and excludes subtle role sensitivity; it decisively contradicts “only assistant text.”

| Controlled input | Active content tokens |
|---|---:|
| prose | 84.6% |
| question | 81.5% |
| code | 0.0% |
| python2 | 0.0% |
| python3 | 0.0% |
| sql | 0.0% |
| javascript | 0.0% |
| code comment | 30.6% |
| code string | 65.2% |
| word salad | 90.0% |
| gibberish | 91.9% |
| repeated | 2.9% |
| copy cycle | 4.2% |
| number words | 0.0% |
| number digits | 0.0% |
| chinese | 87.5% |
| french | 88.9% |
| poem | 86.2% |


These are small hand-written probes, not benchmarks. The most diagnostic comparisons are:

- **Code versus its embedded language:** five pure code probes across Python, SQL, and JavaScript have zero activation. In `code_comment`, the Python stays off while 15 word tokens inside the English comment activate. The same happens inside a quoted English string. This is local sensitivity, not simply classifying the entire document as code or prose.
- **Meaning versus word-like form:** shuffled words activate at 90%; invented words such as `flarn gribble zork plimble` activate at 91.9%. Semantic coherence is unnecessary.
- **Vocabulary versus context:** ` dog` is strongly active in prose but only 1 of 35 repetitions activates. A repeating six-animal cycle activates on 2 of 48 tokens. Spelled-out counting from one to twenty is entirely off, while number words inside a normal explanatory sentence can activate. Thus it is more than “letters rather than digits.”
- **Language:** French and Chinese probes activate strongly, ruling out an English-only gate. Two examples cannot establish language universality.

The repetition/counting results suggest suppression during predictable continuation or copying, but I did not measure next-token entropy or identify an induction circuit. Treat that as a follow-up hypothesis, not a demonstrated mechanism.

**Causal experiments.** I used the original Qwen forward pass and added `(new_gate - natural_gate) × ((standardized_input @ V_29) @ U_29) × output_std` to the raw layer-22 MLP output. This preserves the reconstruction error and all other transforms, then runs the remaining Qwen layers normally. No frozen-logit projection is used for these causal results. The intervention applies at every consumed token in each tested sequence. Effects at inactive positions can therefore arrive through downstream attention to earlier intervened positions.

Fixed-token evaluation used the first 384 tokens of eight specified cached conversations (IDs 0, 2, 5, 9, 16, 25, 40, 55), **3,072 prediction positions**, with **3,064 available next-token targets**. These conversations were deliberately varied rather than randomly selected.

| Intervention | Mean KL, baseline to intervention | Top prediction changed | Mean next-token loss change |
|---|---:|---:|---:|
| Remove transform 29 | 0.01555 nats | 4.59% | +0.02051 nats |
| Double its natural gate | 0.01504 nats | 4.88% | +0.00375 nats |

Among naturally active positions, ablation changes **5.35%** of top predictions. The removed vector is substantial: the ratio of mean removed-vector norm to mean original MLP-output norm is **0.431** across these equal-length sequences. That is not “43% of model computation” or a variance fraction. The modest downstream effect shows why gate frequency and output-vector magnitude alone do not measure causal importance.

Four greedy-generation comparisons (up to 64 new tokens, gate ablated on prompt and generated positions) showed wording/content variation in an autumn explanation and a dog story, but identical outputs for `17 + 25 = 42` and the tested Python-function response. The prose remained coherent. For example, the story changed “cold, gray morning” to “cold, misty morning,” and changed incidental names and setting. These examples do not establish a reliable style, topic, or capability controlled by the transform. Generation samples are short and often end at the token limit.

**Numerical checks.** Replaying four full-length cached conversations reproduced all **4,096 gates exactly**, including zero binary disagreements. A multiply-by-one intervention produced identical generated token IDs and zero first-prediction logit change. Shortening the causal sequences to 384 tokens changes numerical execution shape: gate values differ slightly, and a few threshold crossings can create differences around 0.65. The causal experiments use newly computed natural gates for both branches, not stale cached gates. This is why I verified full-length cache reproduction separately.

**Interpretation and limits.** I would annotate the dashboard with the working label above, with **high confidence in the observed activation profile** and **moderate confidence in the broad routing interpretation**. I would not label it “assistant,” “semantic meaning,” “intelligence,” or a specific topic. It is also not completely uninterpretable: its negative examples and context dependence form a clear pattern. However, a MOLT rank-512 transform has an input-dependent output vector, so an interpretable gate does not imply one interpretable output direction. The exact computation implemented within that subspace remains unresolved. These experiments identify a useful behavioral description, not unique mechanistic ground truth.

**Artifacts and reproduction.** `index.html` contains searchable token highlights for every controlled probe and baseline/ablated generations. `analyze_cache.py` computes corpus categories and examples; `experiments.py` runs direct GPU probes and causal experiments; `validation.py` uses the existing local steering service for the additional probes and fresh conversations; `check_service.py` performs replay/no-op checks; `build_report.py` builds this report. Run Python from the repository root with `PYTHONPATH=.` and the project `.venv`. The service scripts use `http://100.76.58.1:8766` and require that local service to be running. GPU memory is shared with training; validation reused the already-loaded service after a second model copy could not fit. All result JSON/NPZ files are saved alongside the scripts. No model weights or training settings were changed.
