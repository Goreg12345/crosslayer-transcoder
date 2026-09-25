#!/usr/bin/env python3
"""Render a focused human-review workspace for the fixed MOLT pilot cohort."""

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import time


def esc(value):
    return html.escape(str(value), quote=True)


def js(value):
    return (
        json.dumps(value, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def write(path, text):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


STYLE = """
*{box-sizing:border-box}body{margin:0;background:#f4f7fa;color:#203545;font:14px system-ui;line-height:1.55}header,main{max-width:1550px;margin:auto;padding:24px 30px}header{background:#fff;border-bottom:1px solid #dbe4e9}h1{font-size:28px;margin:8px 0}h2{font-size:20px}h3{font-size:16px;margin:0 0 8px}.eyebrow{font-size:12px;letter-spacing:1px;color:#087969;font-weight:700}p{max-width:1050px;color:#526576}a{color:#087969}button,select,textarea,input{font:inherit;border:1px solid #c6d5de;border-radius:6px;padding:7px;background:#fff;color:inherit}button{cursor:pointer}button:hover{background:#e7f5f1}textarea{width:100%;min-height:95px}input{min-width:260px}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.card,details{background:white;border:1px solid #dbe4e9;border-radius:9px;padding:16px;margin:12px 0}summary{cursor:pointer;font-weight:650}table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px 12px;border-bottom:1px solid #e0e8ed;text-align:left;vertical-align:top}th{font-size:12px;background:#edf4f6;position:sticky;top:0}.table-wrap{overflow:auto;border:1px solid #dbe4e9;border-radius:8px}.label{min-width:200px;max-width:300px}.score{white-space:nowrap;font-variant-numeric:tabular-nums}.sub{display:block;font-size:11px;color:#667985}.tokens{white-space:pre-wrap;overflow-wrap:anywhere;line-height:2.05;font-family:ui-monospace,monospace;background:#fafcfd;padding:12px;border-radius:6px}.tok{border-radius:2px;padding:1px 0}.marked{outline:1px solid #247bc2;outline-offset:1px}.metric-grid{display:flex;gap:12px;flex-wrap:wrap}.metric-grid>.card{flex:1;min-width:180px}.value{font-size:24px;font-weight:650}.status{font-size:12px;border-radius:12px;padding:3px 9px;background:#e8eff4}.wrong{border-left:4px solid #cb8144}.right{border-left:4px solid #80afa4}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}.explanation{font-size:18px;color:#203545}.review{border-color:#92b7ab;background:#f5fbf8}.legend{font-size:12px}.compact{margin:5px 0;font-size:12px}.method-note{font-size:13px}nav{display:flex;gap:16px;flex-wrap:wrap}.empty{color:#83939d}
"""


REVIEW_JS = """
const STORE='molt-pilot-review:'+CONFIG.fingerprint;
function reviews(){try{return JSON.parse(localStorage.getItem(STORE)||'{}')}catch{return {}}}
function saveReview(id,field,value){const all=reviews();all[id]={...(all[id]||{}),[field]:value,updated_at:new Date().toISOString()};localStorage.setItem(STORE,JSON.stringify(all));}
function wireReviews(){for(const input of document.querySelectorAll('[data-review-id]')){const id=input.dataset.reviewId,field=input.dataset.field;input.value=reviews()[id]?.[field]||(field==='verdict'?'unreviewed':'');input.addEventListener('input',()=>saveReview(id,field,input.value));}}
function exportReviews(){const all=reviews(),data={fingerprint:CONFIG.fingerprint,checkpoint:CONFIG.checkpoint,cache_sha256:CONFIG.cache_sha256,created_at:new Date().toISOString(),reviews:CONFIG.selected.map(f=>({id:f.id,rank:f.rank,...(all[f.id]||{verdict:'unreviewed'})}))};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download=CONFIG.review_filename||'molt-layer22-pilot32-reviews.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
wireReviews();document.querySelectorAll('[data-export]').forEach(b=>b.onclick=exportReviews);
"""


def verdict(feature):
    return f'<select aria-label="Review verdict for transform {feature}" data-review-id="{feature}" data-field="verdict"><option value="unreviewed">Unreviewed</option><option value="good">Good</option><option value="mixed">Mixed</option><option value="poor">Poor</option></select>'


def shell(title, body, config, extra="", refresh=False):
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{STYLE}</style></head><body>{body}<script>const CONFIG={js(config)};{REVIEW_JS}{extra}</script></body></html>'


def percent(value):
    return "—" if value is None else f"{100 * value:.0f}%"


def metric_cell(method):
    if not method:
        return '<span class="empty">Pending</span>'
    metrics = method.get("metrics", {})
    status = method.get("status", "pending")
    if "pearson" in metrics:
        score = (
            "undefined" if metrics["pearson"] is None else f"{metrics['pearson']:.2f}"
        )
        skill = metrics.get("mse_skill_vs_zero")
        return f'<span class="score">r {score}</span><span class="sub">{percent(skill)} skill vs zero · {metrics["valid"]}/{metrics["total"]} windows</span>'
    if "accuracy" in metrics:
        interval = metrics.get("accuracy_ci95")
        ci = (
            f"95% CI {percent(interval[0])}–{percent(interval[1])}"
            if interval
            else "No valid decisions"
        )
        return f'<span class="score">{percent(metrics["accuracy"])}</span><span class="sub">{ci}<br>{metrics["valid"]}/{metrics["total"]} valid</span>'
    return esc(status)


def tokens_html(tokens, acts, marked=(), ids=None, maximum=None):
    maximum = maximum or max(acts, default=0) or 1
    marked = set(marked)
    spans = []
    for i, (token, act) in enumerate(zip(tokens, acts)):
        color = (
            f"rgba(239,151,57,{0.15 + 0.55 * act / maximum:.3f})"
            if act > 0
            else "transparent"
        )
        title = f"Position {i} · activation {act:.6g}" + (
            f" · token ID {ids[i]}" if ids is not None else ""
        )
        spans.append(
            f'<span class="tok{" marked" if i in marked else ""}" style="background:{color}" title="{esc(title)}">{esc(token)}</span>'
        )
    return '<div class="tokens">' + "".join(spans) + "</div>"


DESCRIPTIONS = {
    "detection": "Does the explanation identify an activating window? 50 positive and 50 zero-gate windows; chance accuracy is 50%.",
    "fuzz": "Does the explanation justify every marked token? Correctly marked positive windows versus zero-gate windows with artificial marks; chance is 50%.",
    "active_fuzz": "A harder marking test within positive contexts: real activating tokens versus genuinely zero-gate tokens in those same contexts; chance is 50%.",
    "intruder": "Find the zero-gate intruder among four positive contexts. This tests feature coherence without using the explanation; chance is 20%. Groups reuse positive contexts.",
    "simulation": "Predict 0–10 activation at every token from the explanation, on 10 positive and 10 negative windows. Pearson r measures alignment; skill compares MSE with always predicting zero. Undefined correlation is shown explicitly.",
}


def render_feature(root, output, row, selected, config):
    feature = row["id"]
    methods = row.get("methods", {})
    method_names = config.get("methods", list(DESCRIPTIONS))
    full_layer = config.get("scope") == "layer"
    layer = config.get("layer", 22)
    steer_link = f'<a href="/steer?feature={feature}">Steer</a>' if layer == 22 else ""
    ids = [s["id"] for s in selected]
    index = ids.index(feature)
    evidence_path = output / "evidence" / f"{feature}.json"
    evidence = (
        json.loads(evidence_path.read_text())
        if evidence_path.exists()
        else {"train": [], "test": [], "negatives": []}
    )
    audience = "TRANSFORM" if full_layer else "PILOT TRANSFORM"
    header = f'<header><div class="eyebrow">MOLT · LAYER {layer} · {audience} {feature}</div><h1>{esc(row.get("label", f"Transform {feature}"))}</h1><nav><a href="../">All {len(selected):,} transforms</a><a href="{ids[(index - 1) % len(ids)]}.html">← Previous</a><a href="{ids[(index + 1) % len(ids)]}.html">Next →</a><a href="../../features/{feature}.html">Activation dashboard</a>{steer_link}<a href="{feature}.json">Raw results</a><a href="../audit/{feature}.jsonl">Prompt audit</a></nav><p>Rank {row["rank"]} · fires on {100 * row["frequency"]:.4f}% of 10M tokens · {row["count"]:,} active observations · <span class="status">{esc(row.get("status", "pending"))}</span></p></header>'
    body = header + "<main>"
    body += f'<section class="card"><h2>Gate-activation hypothesis</h2><p class="explanation">{esc(row.get("explanation") or ("No activation evidence in this sample." if row.get("status") == "not_observed" else "Explanation pending."))}</p><p class="compact">Describes when the gate fires. It does not establish the transform’s causal output effect.</p></section>'
    body += f'<section class="card review"><h2>Your assessment</h2><div class="toolbar">{verdict(feature)}<button data-export>Export all reviews</button></div><p class="compact">Saved in this browser. {"Completed campaign layers are queued for interpretation." if full_layer else "The archived 32-transform pilot."}</p><textarea aria-label="Review notes" placeholder="Does the hypothesis fit the examples? What does it miss?" data-review-id="{feature}" data-field="notes"></textarea></section>'
    body += (
        '<div class="metric-grid">'
        + "".join(
            f'<div class="card"><h3>{esc(name.replace("_", " ").title())}</h3>{metric_cell(methods.get(name))}</div>'
            for name in method_names
        )
        + "</div>"
    )
    body += f"<p>Evidence: {len(evidence['train'])} explanation windows, {len(evidence['test'])} held-out positive windows, {len(evidence['negatives'])} negative windows.</p><details><summary>Explanation evidence · {len(evidence['train'])} separate conversations</summary><p>Only these examples were shown to the explainer. Windows end at their peak; orange intensity shows the actual gate. Token tooltips show exact values.</p>"
    for i, ex in enumerate(evidence["train"]):
        body += (
            f'<div class="card"><h3>Example {i + 1} · conversation {ex["conversation"]}</h3>'
            + tokens_html(ex["str_tokens"], ex["activations"], ids=ex["tokens"])
            + "</div>"
        )
    body += "</details>"
    for name in method_names:
        description = DESCRIPTIONS[name]
        if full_layer and name == "detection":
            description = "Does the explanation identify activating windows? Positive and zero-gate cases are balanced; chance accuracy is 50%. Rare transforms have fewer cases."
        method = methods.get(name, {})
        body += f'<details><summary>{esc(name.replace("_", " ").title())} · {esc(method.get("status", "pending"))}</summary><p class="method-note">{esc(description)}</p>'
        if method.get("reason"):
            body += f"<p>{esc(method['reason'])}</p>"
        if method.get("error"):
            body += f"<p>Scoring error: {esc(method['error'])}</p>"
        rows = method.get("rows", [])
        if name in ("detection", "fuzz", "active_fuzz"):
            body += '<p class="legend">Errors first. Orange = measured gate; blue outline = tokens judged by the fuzzing scorer.</p>'
            for i, case in sorted(
                enumerate(rows), key=lambda item: (item[1]["correct"] is True, item[0])
            ):
                prediction = (
                    "unscored"
                    if case["prediction"] is None
                    else ("yes" if case["prediction"] else "no")
                )
                expected = "yes" if case["activating"] else "no"
                question = "Gate present?" if name == "detection" else "Marks correct?"
                body += (
                    f'<details class="{"right" if case["correct"] else "wrong"}"><summary>Case {i + 1} · {question} Expected {expected}; predicted {prediction}</summary>'
                    + tokens_html(
                        case["str_tokens"],
                        case["activations"],
                        case.get("highlighted_positions", []),
                    )
                    + "</details>"
                )
        elif name == "intruder":
            for i, case in sorted(
                enumerate(rows), key=lambda item: (item[1]["correct"], item[0])
            ):
                sample = case["sample"]
                if sample is None:
                    body += f"<p>Case {i + 1}: unscored parsing failure.</p>"
                    continue
                body += f'<details class="{"right" if case["correct"] else "wrong"}"><summary>Case {i + 1} · intruder {case["correct_index"]}; predicted {case["prediction"]}</summary>'
                for j, text in enumerate(sample["examples"]):
                    body += f"<p>Example {j}{' · actual intruder' if j == case['correct_index'] else ''}</p><pre>{esc(text)}</pre>"
                body += "</details>"
        else:
            for i, case in enumerate(rows):
                body += (
                    f"<details><summary>Window {i + 1} · {'positive' if case['activating'] else 'zero-gate negative'}{' · unscored' if case.get('prediction') is None else ''}</summary><p>Measured, normalized 0–10</p>"
                    + tokens_html(
                        case["str_tokens"],
                        case["actual"],
                        ids=case["tokens"],
                        maximum=10,
                    )
                )
                if case.get("prediction") is not None:
                    body += "<p>Predicted, 0–10</p>" + tokens_html(
                        case["str_tokens"], case["prediction"], maximum=10
                    )
                else:
                    body += f"<p>{esc(case.get('error', 'Unavailable'))}</p>"
                body += "</details>"
        body += "</details>"
    body += '<p>All evaluation conversations are held out from the explainer. Exact duplicate windows are removed. The same LLM generates and judges these hypotheses; manual review remains essential. Peak-ending windows can give highlight-based scorers token-position cues; Delphi intruder distractors also include a fixed three-quarter-position highlight. Treat these scores as exploratory. Full method settings and sampling exclusions are in <a href="../manifest.json">the manifest</a>.</p></main>'
    write(
        output / "features" / f"{feature}.html",
        shell(f"MOLT transform {feature} review", body, config),
    )


def render(root):
    output = root / "pilot32"
    selection = json.loads((output / "selection.json").read_text())
    settings = selection["settings"]
    summary = (
        json.loads((output / "summary.json").read_text())
        if (output / "summary.json").exists()
        else {}
    )
    selected = selection["selected"]
    features = summary.get("features", {})
    fingerprint = summary.get("fingerprint", settings["cache_sha256"])
    config = dict(
        fingerprint=fingerprint,
        checkpoint=settings["checkpoint"],
        cache_sha256=settings["cache_sha256"],
        selected=selected,
    )
    counts = Counter(row.get("status", "pending") for row in features.values())
    methods = sum(
        m.get("status") == "complete"
        for row in features.values()
        for m in row.get("methods", {}).values()
    )
    state = summary.get("run", {}).get("state", "preparing")
    body = f'<header><div class="eyebrow">MOLT · QWEN3-4B LAYER 22 · HUMAN REVIEW PILOT</div><h1>32 transforms, five evaluation methods</h1><p>10,000,000 held-out tokens · 11,345 conversations · checkpoint step 100,000<br><strong>{counts["complete"]}/32 transforms complete · {methods}/160 method evaluations complete</strong> · {esc(state)}</p><p>The cohort was fixed across all five ranks and activation-frequency bins before explanations were generated. Each hypothesis uses 32 distinct examples. Detection and both fuzzing tests each use 100 held-out decisions.</p><nav><a href="../">All activation pages</a><a href="selection.json">Fixed cohort and sampling</a><a href="manifest.json">Method settings</a><a href="validation.json">Evidence audit</a><button data-export>Export your reviews</button></nav></header><main>'
    if (root / "autointerp-layer" / "manifest.json").exists():
        body += '<section class="card review"><strong>Archived pilot.</strong> Expansion to all transforms in layer 22 has been approved. <a href="../autointerp-layer/">Open full-layer explanations and four scores →</a> The pilot results and your review notes remain available here.</section>'
    else:
        body += '<section class="card review"><strong>Review first, expand later.</strong> Inspect hypotheses, evidence, and scored cases, then save and export your notes.</section>'
    body += '<div class="toolbar"><input id="search" placeholder="Find ID or explanation" aria-label="Find transform"><label>Rank <select id="rank"><option value="">All</option><option>512</option><option>256</option><option>128</option><option>64</option><option>32</option></select></label></div><p class="compact">Scores are provisional. Detection/fuzz chance: 50%; intruder chance: 20%. Intruder measures coherence, not explanation quality. Simulation reports correlation and MSE improvement over a zero predictor. Confidence intervals are approximate, especially for paired active-fuzz cases and reused intruder contexts. Peak-ending windows and the upstream intruder highlight sampler can introduce position cues; inspect cases alongside scores.</p><div class="table-wrap"><table><thead><tr><th>Transform / rank</th><th>Firing rate</th><th>Hypothesis label</th><th>Detection</th><th>Fuzz</th><th>Active fuzz</th><th>Intruder</th><th>Simulation</th><th>Your verdict</th></tr></thead><tbody>'
    for selected_row in selected:
        feature = selected_row["id"]
        row = features.get(str(feature), dict(**selected_row, status="pending"))
        label = row.get("label", "Explanation pending")
        body += f'<tr data-rank="{row["rank"]}" data-search="{esc(str(feature) + " " + label.lower())}"><td><a href="features/{feature}.html"><strong>{feature}</strong></a><span class="sub">rank {row["rank"]}</span></td><td>{100 * row["frequency"]:.4f}%</td><td class="label"><a href="features/{feature}.html">{esc(label)}</a><span class="sub">{esc(row["status"])}</span></td>'
        body += "".join(
            "<td>" + metric_cell(row.get("methods", {}).get(name)) + "</td>"
            for name in DESCRIPTIONS
        )
        body += "<td>" + verdict(feature) + "</td></tr>"
        result_path = output / "features" / f"{feature}.json"
        page = output / "features" / f"{feature}.html"
        if (
            not page.exists()
            or max(
                Path(__file__).stat().st_mtime,
                result_path.stat().st_mtime if result_path.exists() else 0,
            )
            > page.stat().st_mtime
        ):
            render_feature(root, output, row, selected, config)
    body += "</tbody></table></div><p>Live scores use the same model as the explainer. Low scores remain in this fixed cohort. No explanations or thresholds are tuned using these evaluation results.</p></main>"
    extra = """function filter(){const q=document.getElementById('search').value.toLowerCase(),rank=document.getElementById('rank').value;for(const row of document.querySelectorAll('tr[data-search]'))row.hidden=!row.dataset.search.includes(q)||(rank&&row.dataset.rank!==rank)}document.getElementById('search').oninput=filter;document.getElementById('rank').onchange=filter;"""
    if state not in ("complete", "incomplete"):
        extra += 'setTimeout(()=>{if(document.activeElement?.tagName!=="INPUT")location.reload()},30000);'
    write(
        output / "index.html",
        shell("MOLT · 32-transform quality pilot", body, config, extra),
    )
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    while True:
        state = render(args.directory)
        print(f"Rendered pilot review workspace ({state})", flush=True)
        if not args.watch or state in ("complete", "incomplete"):
            break
        time.sleep(15)
