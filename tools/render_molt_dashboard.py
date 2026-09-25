#!/usr/bin/env python3
"""Render collected MOLT gates with SAE-Vis 0.3.7; no decoder-direction assumptions."""

import argparse
import fcntl
import hashlib
import html
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import sae_vis
from molt_dashboard_sampling import example_peak_indices
from sae_vis.data_config_classes import (
    ActsHistogramConfig,
    Column,
    LogitsTableConfig,
    SaeVisConfig,
    SaeVisLayoutConfig,
    SeqMultiGroupConfig,
)
from sae_vis.data_storing_fns import (
    ActsHistogramData,
    LogitsTableData,
    SaeVisData,
    SeqGroupData,
    SeqMultiGroupData,
    SequenceData,
)


class PreciseLogitsTableData(LogitsTableData):
    """Preserve small direct projections that SAE-Vis rounds to two decimals."""

    def data(self, *args, **kwargs):
        result = super().data(*args, **kwargs)
        for key, values in [
            ("posLogits", self.top_logits),
            ("negLogits", self.bottom_logits),
        ]:
            for row, value in zip(result[key], values):
                row["value"] = value
        result["maxLogits"] = max(abs(v) for v in self.top_logits + self.bottom_logits)
        return result


def safe_json(value):
    return (
        json.dumps(value, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def write_index(root, meta, stats):
    """Publish a self-contained score snapshot; the browser refreshes it while served."""
    review = {}
    review_path = root / "autointerp-layer" / "summary.json"
    if review_path.exists():
        summary = json.loads(review_path.read_text())
        settings = summary["settings"]
        provenance_keys = (
            "checkpoint",
            "global_step",
            "cache_sha256",
            "layer",
            "tokens",
        )
        if any(settings.get(key) != meta.get(key) for key in provenance_keys):
            raise ValueError("Scores belong to a different dashboard sample")
        review = dict(
            settings={key: settings[key] for key in provenance_keys},
            features={
                feature: {
                    key: row[key]
                    for key in (
                        "status",
                        "title",
                        "label",
                        "explanation",
                        "methods",
                        "evidence_counts",
                    )
                    if key in row
                }
                for feature, row in summary["features"].items()
            },
        )
        meta["layer_review_url"] = "autointerp-layer/"
    template = Path(__file__).with_name("molt_dashboard_index.html").read_text()
    template = (
        template.replace("__METADATA__", safe_json(meta))
        .replace("__FEATURES__", safe_json(stats))
        .replace("__INTERPRETATIONS__", safe_json(review))
    )
    (root / "index.html.tmp").write_text(template)
    (root / "index.html.tmp").replace(root / "index.html")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "directory", type=Path, nargs="?", default=Path("results/molt-qwen-dashboard")
    )
    p.add_argument(
        "--index-only",
        action="store_true",
        help="Refresh the explorer and scores from existing statistics, without rebuilding activation pages",
    )
    args = p.parse_args()
    root = args.directory
    render_lock = (root / ".render.lock").open("a")
    fcntl.flock(render_lock, fcntl.LOCK_EX)
    meta = json.loads((root / "metadata.json").read_text())
    with (root / "activations.npz").open("rb") as cache_file:
        meta["cache_sha256"] = hashlib.file_digest(cache_file, "sha256").hexdigest()
    explanations = {}
    explanation_path = root / "autointerp" / "summary.json"
    if explanation_path.exists():
        annotation = json.loads(explanation_path.read_text())
        provenance = annotation["settings"]
        if (
            provenance["checkpoint"] != meta["checkpoint"]
            or provenance["global_step"] != meta["global_step"]
            or provenance["cache_sha256"] != meta["cache_sha256"]
        ):
            raise ValueError("Explanations belong to a different dashboard sample")
        explanations = annotation["features"]
        meta["explanation_model"] = provenance["model"]
        meta["cache_sha256"] = provenance["cache_sha256"]
        if annotation.get("pilot_only"):
            meta["pilot_review_url"] = "pilot32/"
        if annotation.get("review_url"):
            meta["layer_review_url"] = annotation["review_url"]
    if args.index_only:
        stats = json.loads((root / "feature_stats.json").read_text())
        for stat in stats:
            stat.pop("explanation", None)
            explanation = explanations.get(str(stat["id"]), {})
            if explanation.get("status") == "explained":
                stat["explanation"] = explanation
        meta["has_logit_projections"] = (root / "logit_projections.json").exists()
        write_index(root, meta, stats)
        print(f"Updated layer {meta['layer']} explorer with {len(stats)} transforms")
        return
    cache = np.load(root / "activations.npz")
    tokens, offsets = cache["tokens"], cache["offsets"]
    positions, features, values = cache["positions"], cache["features"], cache["values"]
    vocab = {
        int(k): v for k, v in json.loads((root / "vocab.json").read_text()).items()
    }
    logit_path = root / "logit_projections.json"
    logits = json.loads(logit_path.read_text()) if logit_path.exists() else None
    meta["has_logit_projections"] = bool(logits)
    if logits:
        if (
            logits["checkpoint"] != meta["checkpoint"]
            or logits["tokens"] != meta["tokens"]
        ):
            raise ValueError("Logit projections belong to a different dashboard sample")
        vocab.update({int(k): v for k, v in logits["vocab"].items()})
    assets, pages = root / "assets", root / "features"
    assets.mkdir(exist_ok=True)
    pages.mkdir(exist_ok=True)
    shutil.copyfile(
        Path(__file__).with_name("molt_conversation.js"),
        assets / "molt-conversation.js",
    )
    lib = Path(sae_vis.__file__).parent
    shutil.copyfile(lib / "init.js", assets / "sae-vis.js")
    (assets / "sae-vis.css").write_text(
        (lib / "style.css").read_text()
        + Path(__file__).with_name("molt_dashboard_features.css").read_text()
    )
    for name, url in [
        ("d3.js", "https://cdn.jsdelivr.net/npm/d3@6.7.0/dist/d3.min.js"),
        ("plotly.js", "https://cdn.plot.ly/plotly-2.35.2.min.js"),
    ]:
        if not (assets / name).exists():
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                (assets / name).write_bytes(response.read())
    layout = SaeVisLayoutConfig(
        [
            Column(ActsHistogramConfig(n_bins=30), width=340),
            Column(
                SeqMultiGroupConfig(
                    buffer=None,
                    n_quantiles=2,
                    top_acts_group_size=12,
                    quantile_group_size=6,
                    top_logits_hoverdata=0,
                ),
                width=850,
            ),
        ],
        height=1000,
    )
    base_cfg = SaeVisConfig(feature_centric_layout=layout)
    logit_layout = SaeVisLayoutConfig(
        [
            Column(
                ActsHistogramConfig(n_bins=30),
                LogitsTableConfig(n_rows=logits["top_k"] if logits else 10),
                width=340,
            ),
            layout.columns[1],
        ],
        height=1000,
    )
    logit_cfg = SaeVisConfig(feature_centric_layout=logit_layout)
    stats = []
    order = np.argsort(features, kind="stable")
    cuts = np.searchsorted(features[order], np.arange(meta["n_features"] + 1))
    for f in range(meta["n_features"]):
        ix = order[cuts[f] : cuts[f + 1]]
        pos, val = positions[ix], values[ix]
        count = len(ix)
        stat = dict(
            id=f,
            rank=meta["ranks"][f],
            count=count,
            frequency=count / len(tokens),
            maximum=float(val.max()) if count else 0,
            mean_active=float(val.mean()) if count else 0,
        )
        explanation = explanations.get(str(f), {})
        if explanation.get("status") == "explained":
            stat["explanation"] = explanation
        stats.append(stat)
        conversations, selected_groups = example_peak_indices(pos, val, offsets)
        groups = []
        for title, selected in selected_groups:
            sequences = []
            for j in selected:
                conv = int(conversations[j])
                center = int(pos[j])
                lo, hi = (
                    max(int(offsets[conv]), center - 16),
                    min(int(offsets[conv + 1]), center + 17),
                )
                acts = np.zeros(hi - lo, dtype=np.float32)
                left, right = np.searchsorted(pos, [lo, hi])
                acts[pos[left:right] - lo] = val[left:right]
                length = hi - lo
                sequences.append(
                    SequenceData(
                        token_ids=tokens[lo:hi].tolist(),
                        feat_acts=acts.tolist(),
                        token_posns=[
                            f"conversation {conv}, token {n - int(offsets[conv])}"
                            for n in range(lo, hi)
                        ],
                        token_logits=[0.0] * length,
                        top_token_ids=[[] for _ in range(length)],
                        bottom_token_ids=[[] for _ in range(length)],
                        top_logits=[[] for _ in range(length)],
                        bottom_logits=[[] for _ in range(length)],
                    )
                )
            groups.append(
                SeqGroupData(
                    seq_data=sequences,
                    title=title if count else "No activation in this sample",
                )
            )
        # Histograms describe positive gate strengths; zeros are reported in firing frequency.
        upper = float(val.max()) if count else 1.0
        heights, edges = np.histogram(val, bins=30, range=(0, max(upper, 1e-6)))
        hist = ActsHistogramData(
            bar_heights=heights.tolist(),
            bar_values=((edges[:-1] + edges[1:]) / 2).tolist(),
            tick_vals=np.linspace(0, upper, 5).tolist(),
            title="Gate strength when active",
        )
        projection = logits["features"].get(str(f)) if logits else None
        cfg = logit_cfg if projection else base_cfg
        components = dict(actsHistogram=hist, seqMultiGroup=SeqMultiGroupData(groups))

        def logit_table(entry):
            return PreciseLogitsTableData(
                **{
                    k: entry[k]
                    for k in (
                        "top_token_ids",
                        "top_logits",
                        "bottom_token_ids",
                        "bottom_logits",
                    )
                }
            )

        if projection:
            components["logitsTable"] = logit_table(projection["mean"])
        vis = SaeVisData(
            feature_data_dict={f: components},
            cfg=cfg,
            vocab_dict={k: vocab for k in ("embed", "unembed", "probes")},
        )
        logit_control = ""
        logit_script = ""
        if projection:
            entries = [projection["mean"], *projection["examples"]]
            table_data = [
                logit_table(entry).data(logit_layout, vis.decode_fn)
                for entry in entries
            ]
            labels = [
                f"Mean over all {projection['active_tokens']:,} active sampled tokens"
            ]
            for entry in projection["examples"]:
                token = vocab[int(tokens[entry["position"]])]
                labels.append(
                    f"{entry['group']} · conversation {entry['conversation']}, token {entry['token_position']} · {token!r} · gate {entry['gate']:.3f}"
                )
            options = "".join(
                f'<option value="{i}">{html.escape(label)}</option>'
                for i, label in enumerate(labels)
            )
            logit_control = (
                '<section class="projection-controls"><label for="logit-context">Direct logit projection</label><select id="logit-context">'
                + options
                + "</select><p>Positive = upward direct contribution; negative = downward. Scores apply to the <b>next-token prediction</b> after the selected token. Final RMS is held fixed; effects through later layers are excluded. Binary mode changes highlights only.</p></section>"
            )
            logit_script = (
                "<script>const LOGIT_READOUTS="
                + safe_json(table_data)
                + """;
document.addEventListener('DOMContentLoaded',()=>{
    const selector=document.getElementById('logit-context');
    function updateReadout(){
        setupLogitTables(START_KEY,LOGIT_READOUTS[Number(selector.value)],`logitsTable-${START_KEY}`);
        d3.select(`#logitsTable-${START_KEY}`).selectAll('td.right-aligned').text(d=>d.value.toPrecision(4));
        const headings=document.querySelectorAll('.logitsTable h4');
        if(headings.length===2){headings[0].textContent='NEGATIVE · DIRECT';headings[1].textContent='POSITIVE · DIRECT';}
        document.querySelectorAll('.seq').forEach((seq,i)=>seq.classList.toggle('selected-projection',i===Number(selector.value)-1));
    }
    document.querySelectorAll('.seq').forEach((seq,i)=>{
        seq.title='Click to inspect the direct logit projection at this example’s peak token';
        seq.addEventListener('click',()=>{selector.value=String(i+1);updateReadout();});
    });
    selector.addEventListener('change',updateReadout);updateReadout();
});</script>"""
            )
        # Use SAE-Vis's public rendering method, then share its JS/CSS between pages.
        target = pages / f"{f}.html"
        temporary = pages / f".{f}.tmp.html"
        vis.save_feature_centric_vis(str(temporary))
        page = temporary.read_text()
        page = page.replace((lib / "init.js").read_text(), "")
        page = page.replace(
            "<style>\n" + (lib / "style.css").read_text() + "\n</style>",
            '<link rel="stylesheet" href="../assets/sae-vis.css">',
        )
        page = page.replace("https://d3js.org/d3.v6.min.js", "../assets/d3.js").replace(
            "https://cdn.plot.ly/plotly-latest.min.js", "../assets/plotly.js"
        )
        # Escape dataset-derived strings at the script boundary (SAE-Vis handles token HTML).
        start = page.index("    return ", page.index("function defineData()")) + len(
            "    return "
        )
        end = page.index(";\n}", start)
        page = (
            page[:start]
            + page[start:end].replace("<", "\\u003c").replace(">", "\\u003e")
            + page[end:]
        )
        heading = f"<h2>Layer {meta['layer']} · Transform {f} <small>rank {stat['rank']}</small></h2><p>Fires on {count:,} / {len(tokens):,} tokens ({100 * stat['frequency']:.3f}%). Maximum gate: {stat['maximum']:.4f}. Mean when active: {stat['mean_active']:.4f}.</p>"
        if explanation.get("status") == "explained":
            score = explanation.get("detection")
            score_text = (
                f"Held-out detection: {score['tp'] + score['tn']}/{score['total']} correct · {score['parse_failures']} unscored · "
                if score
                else "Held-out detection pending or unavailable · "
            )
            heading += (
                '<section style="background:#eef6f4;padding:12px;border-radius:8px">'
                "<strong>LLM gate explanation · provisional</strong><p>"
                + html.escape(explanation["label"])
                + "</p><small>"
                + score_text
                + html.escape(meta["explanation_model"])
                + ". Describes when the gate fires; does not establish the transform’s causal effect.</small></section>"
            )
            if explanation.get("explanation"):
                heading += "<p>" + html.escape(explanation["explanation"]) + "</p>"
            if explanation.get("pilot_url"):
                heading += (
                    '<p><a href="'
                    + html.escape(explanation["pilot_url"], quote=True)
                    + '">Review all five scoring methods and add your notes →</a></p>'
                )
            if explanation.get("review_url"):
                heading += (
                    '<p><a href="'
                    + html.escape(explanation["review_url"], quote=True)
                    + '">Review explanation, title, and all four scoring methods →</a></p>'
                )
        binary = """<script>
if (new URLSearchParams(location.search).get('binary') === '1') {
    for (const feature of Object.values(DATA)) {
        for (const group of feature.seqMultiGroup) {
            group.seqGroupMetadata.maxAct = 1;
            for (const sequence of group.seqGroupData) {
                for (const token of sequence.seqData) {
                    if (token.featAct > 0) token.featAct = 1;
                }
            }
        }
        feature.actsHistogram = {x:[0,1], y:[TOTAL-ACTIVE,ACTIVE],ticks:[0,1],title:'Inactive / active token counts'};
    }
}
</script>""".replace("TOTAL", str(len(tokens))).replace("ACTIVE", str(count))
        temporary.write_text(
            '<!doctype html><html><head><meta charset="utf-8"><title>MOLT transform '
            + str(f)
            + "</title><style>body{font-family:system-ui;margin:18px}small{color:#607080;font-size:16px}#dropdown-container{display:none}</style></head><body>"
            + '<div class="feature-intro">'
            + heading
            + "</div>"
            + logit_control
            + page
            + binary
            + '<script src="../assets/sae-vis.js"></script>'
            + logit_script
            + "</body></html>"
        )
        temporary.replace(target)
    (root / "feature_stats.json.tmp").write_text(json.dumps(stats, indent=2))
    (root / "feature_stats.json.tmp").replace(root / "feature_stats.json")
    write_index(root, meta, stats)
    print(
        f"Rendered {len(stats)} transform pages; {sum(s['count'] > 0 for s in stats)} active in sample"
    )


if __name__ == "__main__":
    main()
