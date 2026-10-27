#!/usr/bin/env python3
"""Publish live full-layer explanations, titles, and four Delphi scores."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

from crosslayer_transcoder.dashboard.layer_explanations import METHODS
from tools.render_molt_pilot import (
    esc,
    metric_cell,
    render_feature,
    shell,
    verdict,
    write,
)


def render(root):
    output = root / "autointerp-layer"
    manifest_path = output / "manifest.json"
    summary_path = output / "summary.json"
    if not manifest_path.exists() or not summary_path.exists():
        return "preparing", "preparing"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    settings, selected = manifest["settings"], manifest["selected"]
    metadata = json.loads((root / "metadata.json").read_text())
    layer = settings["layer"]
    pilot_link = (
        '<a href="../pilot32/">Archived 32-transform pilot</a>'
        if (root / "pilot32" / "index.html").exists()
        else ""
    )
    features = summary["features"]
    counts = Counter(row["status"] for row in features.values())
    run = summary["run"]
    config = dict(
        fingerprint=manifest["fingerprint"],
        checkpoint=settings["checkpoint"],
        cache_sha256=settings["cache_sha256"],
        scope="layer",
        layer=layer,
        methods=list(METHODS),
        selected=[dict(id=f["id"], rank=f["rank"]) for f in selected],
        review_filename=f"molt-layer{layer}-all-transform-reviews.json",
    )
    scored = sum(
        all(
            row.get("methods", {}).get(m, {}).get("status") == "complete"
            for m in METHODS
        )
        for row in features.values()
    )
    pending = sum(
        not row.get("explanation") and row["status"] not in ("not_observed", "discarded")
        for row in features.values()
    )
    totals = manifest["preparation_counts"]
    concurrency = summary.get("usage", {}).get("concurrency_limit", 32)
    backend_names = [row["name"] for row in summary.get("usage", {}).get("backends", [])]
    serving = f"{concurrency} concurrent endpoint requests"
    if backend_names:
        serving += " across " + ", ".join(backend_names)
    body = f'<header><div class="eyebrow">MOLT · QWEN3-4B · LAYER {layer} · FULL LAYER</div><h1>All {len(selected):,} transforms</h1><p>{metadata["tokens"]:,} tokens · {metadata["conversations"]:,} conversations · checkpoint step {metadata["global_step"]:,}<br><strong>{run["explained"]:,}/{totals["observed"]:,} observed transforms explained · {scored:,} with all four scores</strong><br>{counts["not_observed"]:,} unobserved · {totals["limited_evidence_budget"]} with limited sample sizes · {pending:,} explanations pending</p><p><span class="status">{esc(run["state"])} · {esc(run["phase"])}</span> · Updated {esc(run["updated_at"])} · {esc(serving)}</p><nav><a href="../">Activation explorer</a>{pilot_link}<a href="/layers/">All layers</a><a href="manifest.json">Methods and provenance</a><a href="summary.json">Live progress JSON</a><button data-export>Export your reviews</button></nav></header><main>'
    body += "<p>Titles and explanations are generated first, then detection, fuzzing, active fuzzing, and intruder scores. This run does not request simulation. Rare transforms show actual sample sizes; transforms without evidence have no inferred explanation. Click a title to inspect the evidence and every scored case.</p>"
    if counts["discarded"]:
        body += f'<p>{counts["discarded"]:,} transforms have discarded work after repeated output-limit failures. Completed scores and audit records are preserved.</p>'
    body += '<div class="toolbar"><input id="search" placeholder="Find ID, title, or explanation" aria-label="Find transform"><label>Rank <select id="rank"><option value="">All</option><option>512</option><option>256</option><option>128</option><option>64</option><option>32</option></select></label><label>Show <select id="status"><option value="observed">Observed transforms</option><option value="all">All transforms</option><option value="explained">Explained</option><option value="scored">Four scores complete</option><option value="limited">Limited evidence</option><option value="not_observed">Unobserved</option><option value="incomplete">Errors / incomplete</option></select></label><span id="visible-count"></span></div>'
    body += '<p class="compact">Detection and fuzzing chance: 50%; intruder chance: 20%. Intruder measures feature coherence without the explanation. The same model explains and judges. Peak-ending windows and upstream intruder highlighting can introduce position cues. Confidence intervals are approximate for paired or reused contexts.</p><div class="table-wrap"><table><thead><tr><th>Transform / rank</th><th>Firing rate</th><th>Title</th><th>Detection</th><th>Fuzz</th><th>Active fuzz</th><th>Intruder</th><th>Your verdict</th></tr></thead><tbody>'
    renderer_mtime = max(
        Path(__file__).stat().st_mtime,
        Path(__file__).with_name("render_molt_pilot.py").stat().st_mtime,
        Path(__file__).with_name("molt_review_scorer_input.py").stat().st_mtime,
    )
    for stat in selected:
        feature = stat["id"]
        row = features[str(feature)]
        title = row.get("title") or "Explanation pending"
        fully_scored = all(
            row.get("methods", {}).get(m, {}).get("status") == "complete"
            for m in METHODS
        )
        limited = row.get("evidence_counts", {}).get("full_budget") is False
        search = f"{feature} {title} {row.get('explanation') or ''}".lower()
        body += f'<tr data-rank="{row["rank"]}" data-status="{esc(row["status"])}" data-explained="{int(bool(row.get("explanation")))}" data-scored="{int(fully_scored)}" data-limited="{int(limited)}" data-search="{esc(search)}"><td><a href="features/{feature}.html"><strong>{feature}</strong></a><span class="sub">rank {row["rank"]}</span></td><td>{100 * row["frequency"]:.4f}%</td><td class="label"><a href="features/{feature}.html">{esc(title)}</a><span class="sub">{esc(row["status"])}</span></td>'
        body += "".join(
            "<td>" + metric_cell(row.get("methods", {}).get(name)) + "</td>"
            for name in METHODS
        )
        body += "<td>" + verdict(feature) + "</td></tr>"
        result_path = output / "features" / f"{feature}.json"
        page = output / "features" / f"{feature}.html"
        if (
            not page.exists()
            or max(result_path.stat().st_mtime, renderer_mtime) > page.stat().st_mtime
        ):
            render_feature(
                root, output, json.loads(result_path.read_text()), selected, config
            )
    body += "</tbody></table></div></main>"
    extra = """
const FILTER='molt-layer-filters:'+CONFIG.fingerprint;
try{const saved=JSON.parse(sessionStorage.getItem(FILTER)||'{}');for(const id of ['search','rank','status'])if(saved[id]!==undefined)document.getElementById(id).value=saved[id]}catch{}
function filter(){const q=document.getElementById('search').value.toLowerCase(),rank=document.getElementById('rank').value,status=document.getElementById('status').value;let count=0;for(const row of document.querySelectorAll('tr[data-search]')){const d=row.dataset,match=status==='all'||(status==='observed'&&d.status!=='not_observed')||(status==='explained'&&d.explained==='1')||(status==='scored'&&d.scored==='1')||(status==='limited'&&d.limited==='1')||(status==='not_observed'&&d.status==='not_observed')||(status==='incomplete'&&['error','incomplete'].includes(d.status));row.hidden=!match||!d.search.includes(q)||(rank&&d.rank!==rank);if(!row.hidden)count++}document.getElementById('visible-count').textContent=count+' transforms';sessionStorage.setItem(FILTER,JSON.stringify({search:document.getElementById('search').value,rank,status}))}
for(const id of ['search','rank','status'])document.getElementById(id).addEventListener('input',filter);filter();
"""
    if run["state"] not in ("complete", "incomplete"):
        extra += 'setTimeout(()=>{if(!["INPUT","SELECT"].includes(document.activeElement?.tagName))location.reload()},30000);'
    write(
        output / "index.html",
        shell(f"MOLT · layer {layer} explanations", body, config, extra),
    )
    return run["state"], run["phase"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--publish-explorer", action="store_true")
    args = parser.parse_args()
    root = args.directory.resolve()
    published = set()
    while True:
        state, phase = render(root)
        print(f"Rendered full-layer review ({state}, {phase})", flush=True)
        publication_stage = (
            "finished"
            if state in ("complete", "incomplete")
            else "titles"
            if phase.startswith("scoring")
            else "initial"
        )
        if (
            args.publish_explorer
            and state != "preparing"
            and publication_stage not in published
        ):
            subprocess.run(
                [
                    str(Path.cwd() / ".venv-dashboard/bin/python"),
                    "tools/render_molt_dashboard.py",
                    str(root),
                ],
                check=True,
            )
            published.add(publication_stage)
        if not args.watch or state in ("complete", "incomplete"):
            break
        time.sleep(15)
