import json,html
from pathlib import Path
import numpy as np
O=Path('results/transform29-investigation');read=lambda p:json.loads((O/p).read_text())
a=read('cache_analysis.json');fresh=read('fresh/cache_analysis.json');e=read('experiments.json');v=read('validation.json');checks=read('numerical_checks.json')
agg={}
for m in [0,2]:
 rows=[x for x in e['causal'] if x['multiplier']==m];agg[m]={}
 for group in rows[0]['groups']:
  totals={k:sum(x['groups'][group][k] for x in rows) for k in rows[0]['groups'][group]}
  agg[m][group]=dict(n=totals['n'],kl=totals['kl_sum']/max(1,totals['n']),flip=totals['flips']/max(1,totals['n']),nll=totals['nll_delta_sum']/max(1,totals['nll_n']))
(O/'causal_summary.json').write_text(json.dumps(agg,indent=2))
table='| Token category | Original 128 conversations | Fresh 64 conversations |\n|---|---:|---:|\n'
for key,label in [('assistant','Assistant content'),('user','User content'),('header','Role headers'),('special','Chat delimiters'),('content_alpha','Content tokens containing letters'),('content_punctuation','Punctuation-only content tokens'),('content_whitespace','Whitespace-only content tokens')]:
 table+=f"| {label} | {a['groups'][key]['rate']:.2%} ({a['groups'][key]['n']:,}) | {fresh['groups'][key]['rate']:.2%} ({fresh['groups'][key]['n']:,}) |\n"
probes=e['probes']+v['probes']
probe_table='| Controlled input | Active content tokens |\n|---|---:|\n'
for name in ['prose','question','code','python2','python3','sql','javascript','code_comment','code_string','word_salad','gibberish','repeated','copy_cycle','number_words','number_digits','chinese','french','poem']:
 p=next(x for x in probes if x['name']==name and x.get('role','user')=='user');probe_table+=f"| {name.replace('_',' ')} | {p['rate']:.1%} |\n"
report=f'''# Transform 29 investigation

**Best working label: “varied word-like text, away from boundaries and repetitive/structured sequences.”** This is a moderately interpretable routing gate for a broad transform. The evidence rules out an assistant-role feature, a specific topic, and a detector of semantic coherence. It does not establish a unique computational purpose for the transform's output.

**Run identification.** W&B run `g2abz94y`, named `molt-qwen3-4b-chat-layer22-N80-lr4e-5`, checkpoint `checkpoints/molt-qwen3-4b/layer-22/chat-ultrachat-lr4e-5/training.ckpt`, step 100,000, saved 2026-09-07 23:03:42 UTC (September 8 00:03:42 London). Model: Qwen/Qwen3-4B; layer index 22. Transform 29 is the zero-based encoder row used in the dashboard and has rank 512.

The cached held-out L0 is **2.9527**, with transform 29 active on **87,892 / 110,066 tokens = 79.8539%**. This matches the identifying feature frequency exactly. The final available training log entry is step 99,949, L0 **2.9650**, `training/mse` **0.25939**; the last ten logged MSE values average **0.25715**. Thus the MSE is close to, but not exactly, the remembered 0.24. I extracted these values directly from the local W&B binary history; I did not rerun full-model MSE evaluation.

**Dataset evidence.** Exploratory data: first 128 UltraChat `test_sft` conversations, each truncated to 1,024 tokens. Replication: the next 64 conversations, 57,744 tokens, collected after forming the hypotheses. Total: **192 conversations / 167,810 tokens**. Counts below are token counts, not independent statistical observations. Active means gate strictly greater than zero; padding is excluded.

{table}

The fresh overall firing rate is **45,490 / 57,744 = 78.7787%**. The original sample's 681 observed turns begin with activation rates of **1.03%, 16.89%, 34.07%, 44.35%, 59.32%** at content positions 0–4; by position 10 it is **85.84%**. Boundary suppression is strong, but not an exact universal rule. A simple distance measure after any period, exclamation mark, question mark, or newline shows the same recovery pattern; this is a character heuristic, not a syntactic sentence parser.

In the original sample, standalone `.` activates on only **5 / 3,403** occurrences, `?` on **0 / 287**, `!` on **0 / 192**, and `:` on **0 / 515**. Punctuation is not uniformly excluded: commas, for example, can activate. Headers and inter-turn separator newlines are off. The fresh sample has one active whitespace token, so “never active on whitespace” would overstate the result.

One actual user question begins:

`How[0] does[0] the[0.96] author[0.98] propose[1.42] to[1.70] fix[1.65] ... system[1.62] ?[0] What[0] changes[1.66] ...`

The feature is clearly processing words inside a user question. Additional randomly sampled inactive-word examples include turn starts such as `How` and `Write`, sentence starts such as `Here`, and code identifiers. There are also exceptions deep inside ordinary prose: this is not a perfect punctuation/position classifier. See `cache_analysis.json` for the reproducibly sampled contexts.

**Controlled hypothesis tests.** Identical prose under user, assistant, and system headers has identical binary activation masks: **22 / 26 = 84.6%**. Identical question text gives **22 / 27 = 81.5%** in all three roles. Gate strengths change slightly. Bare text also activates strongly (88.5% for the prose probe), so a chat wrapper is unnecessary. The role experiment is small and excludes subtle role sensitivity; it decisively contradicts “only assistant text.”

{probe_table}

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
| Remove transform 29 | {agg[0]['all']['kl']:.5f} nats | {agg[0]['all']['flip']:.2%} | +{agg[0]['all']['nll']:.5f} nats |
| Double its natural gate | {agg[2]['all']['kl']:.5f} nats | {agg[2]['all']['flip']:.2%} | +{agg[2]['all']['nll']:.5f} nats |

Among naturally active positions, ablation changes **5.35%** of top predictions. The removed vector is substantial: the ratio of mean removed-vector norm to mean original MLP-output norm is **0.431** across these equal-length sequences. That is not “43% of model computation” or a variance fraction. The modest downstream effect shows why gate frequency and output-vector magnitude alone do not measure causal importance.

Four greedy-generation comparisons (up to 64 new tokens, gate ablated on prompt and generated positions) showed wording/content variation in an autumn explanation and a dog story, but identical outputs for `17 + 25 = 42` and the tested Python-function response. The prose remained coherent. For example, the story changed “cold, gray morning” to “cold, misty morning,” and changed incidental names and setting. These examples do not establish a reliable style, topic, or capability controlled by the transform. Generation samples are short and often end at the token limit.

**Numerical checks.** Replaying four full-length cached conversations reproduced all **4,096 gates exactly**, including zero binary disagreements. A multiply-by-one intervention produced identical generated token IDs and zero first-prediction logit change. Shortening the causal sequences to 384 tokens changes numerical execution shape: gate values differ slightly, and a few threshold crossings can create differences around 0.65. The causal experiments use newly computed natural gates for both branches, not stale cached gates. This is why I verified full-length cache reproduction separately.

**Interpretation and limits.** I would annotate the dashboard with the working label above, with **high confidence in the observed activation profile** and **moderate confidence in the broad routing interpretation**. I would not label it “assistant,” “semantic meaning,” “intelligence,” or a specific topic. It is also not completely uninterpretable: its negative examples and context dependence form a clear pattern. However, a MOLT rank-512 transform has an input-dependent output vector, so an interpretable gate does not imply one interpretable output direction. The exact computation implemented within that subspace remains unresolved. These experiments identify a useful behavioral description, not unique mechanistic ground truth.

**Artifacts and reproduction.** `index.html` contains searchable token highlights for every controlled probe and baseline/ablated generations. `analyze_cache.py` computes corpus categories and examples; `experiments.py` runs direct GPU probes and causal experiments; `validation.py` uses the existing local steering service for the additional probes and fresh conversations; `check_service.py` performs replay/no-op checks; `build_report.py` builds this report. Run Python from the repository root with `PYTHONPATH=.` and the project `.venv`. The service scripts use `http://100.76.58.1:8766` and require that local service to be running. GPU memory is shared with training; validation reused the already-loaded service after a second model copy could not fit. All result JSON/NPZ files are saved alongside the scripts. No model weights or training settings were changed.
'''
(O/'report.md').write_text(report)
# Standalone interactive token explorer; no external scripts, network calls, or model needed.
esc=html.escape
cards=[]
for p in probes:
 name=p['name'].replace('_',' ');role=p.get('role','user')
 spans=''.join(f'<span class="token {"on" if z["gate"]>0 else "off"}" style="--a:{min(.85,.18+z["gate"]*.25):.3f}" title="position {z["position"]}; gate {z["gate"]:.6f}">{esc(z["text"])}</span>' for z in p['tokens'])
 cards.append(f'<article data-name="{esc(name+" "+role)}"><h3>{esc(name)} <small>{esc(role)} · {p["rate"]:.1%} content active</small></h3><div class="tokens">{spans}</div></article>')
gens=''.join(f'<article><h3>{esc(x["prompt"])}</h3><div class="cols"><div><b>Baseline</b><pre>{esc(x["baseline"]["text"])}</pre></div><div><b>Transform 29 removed</b><pre>{esc(x["ablated"]["text"])}</pre></div></div></article>' for x in e['generations'])
page='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Transform 29 investigation</title><style>
body{max-width:1100px;margin:40px auto;padding:0 22px;color:#192d36;background:#f7f9fa;font:16px/1.6 system-ui}h1{font-size:32px;line-height:1.2}h2{margin-top:36px}article{background:white;border:1px solid #dce3e7;border-radius:10px;padding:18px;margin:16px 0}h3{font-size:17px;margin:0 0 15px}small{font-weight:400;color:#516771;margin-left:12px}.tokens{font:15px/2.1 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}.token{border-radius:3px;padding:2px 0}.on{background:rgba(50,175,139,var(--a))}.off{background:#f1dfe0}body.binary .on{background:#6dceb0}body.inactive .on{background:transparent;color:#aaa}input,select{font:inherit;padding:9px;border:1px solid #c4d1d7;border-radius:5px}.controls{position:sticky;top:0;background:#f7f9fa;padding:10px 0;z-index:2;display:flex;gap:12px}.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}pre{white-space:pre-wrap;font:14px/1.6 ui-monospace,monospace}a{color:#007d68}.stats{display:flex;gap:25px;flex-wrap:wrap}.stats b{font-size:24px;display:block}@media(max-width:650px){.cols{grid-template-columns:1fr}.controls{flex-direction:column}}
</style></head><body><h1>Transform 29: varied word-like text</h1><p>A broad gate that favors ongoing word-like content and suppresses boundaries, code, counting, and repetition. This describes where it activates; it does not identify a unique semantic output.</p><div class="stats"><div><b>192</b>conversations</div><div><b>167,810</b>corpus tokens</div><div><b>4.6%</b>top predictions changed by ablation</div></div><p>Qwen3-4B · layer 22 · checkpoint step 100,000 · rank 512. <a href="report.md">Full report and limitations</a> · <a href="experiments.json">Raw causal results</a></p><h2>Controlled probes</h2><p>Green = active; pink = inactive. Hover a token for its gate strength. Percentages exclude chat wrappers. The same word can switch on or off with context.</p><div class="controls"><input id="search" placeholder="Filter: code, prose, repeated…"><select id="mode"><option value="">Gate strength</option><option value="binary">Binary activation</option><option value="inactive">Emphasize inactive tokens</option></select></div><section id="probes">'''+''.join(cards)+'''</section><h2>Greedy generation with transform 29 removed</h2><p>64-token limit; samples may be incomplete. Both branches retain the original Qwen MLP and reconstruction error. These four examples are exploratory.</p>'''+gens+'''<script>document.querySelector('#search').oninput=e=>document.querySelectorAll('#probes article').forEach(x=>x.hidden=!x.dataset.name.includes(e.target.value.toLowerCase()));document.querySelector('#mode').onchange=e=>document.body.className=e.target.value;</script></body></html>'''
(O/'index.html').write_text(page)
print('Saved report.md and index.html')
