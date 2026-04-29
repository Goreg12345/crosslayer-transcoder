// Per-feature view. Reads ?feature=<id>, fetches data/features/<padded>.json
// and data/metadata.json, renders header + token-highlighted examples.

const params = new URLSearchParams(location.search);
const featureId = parseInt(params.get('feature') ?? '0', 10);

function pad(id, width) {
  return String(id).padStart(width, '0');
}

function widthFor(nFeatures) {
  return Math.max(4, String(nFeatures - 1).length);
}

async function main() {
  const md = await fetch('data/metadata.json').then(r => r.json());
  const w = widthFor(md.n_features);
  const f = await fetch(`data/features/${pad(featureId, w)}.json`).then(r => r.json());

  document.title = `Feature #${f.feature_id} — MoLT dashboard`;
  document.getElementById('ckpt').textContent =
    (md.hf_filename || md.ckpt_path) + ` · layer ${md.layer}`;
  document.getElementById('title').textContent = `Feature #${f.feature_id}`;

  const meta = document.getElementById('meta');
  meta.innerHTML = '';
  addMeta(meta, `tier ${f.tier} · rank ${f.rank}`);
  addMeta(meta, `activation rate: ${(f.activation_rate * 100).toFixed(3)}%`);
  addMeta(meta, `max activation: ${f.max_activation.toFixed(3)}`);
  addMeta(meta, `examples: ${f.examples.length}`);

  // Prev/next nav.
  const nav = document.createElement('span');
  if (featureId > 0) {
    nav.appendChild(link(`?feature=${featureId - 1}`, '← prev'));
    nav.appendChild(document.createTextNode(' '));
  }
  if (featureId < md.n_features - 1) {
    nav.appendChild(link(`?feature=${featureId + 1}`, 'next →'));
  }
  meta.appendChild(nav);

  renderActHistogram(
    document.getElementById('act-histogram'),
    f.act_histogram,
    md.act_histogram_edges,
    f.max_activation,
  );
  renderLogitsPanel(document.getElementById('logits'), f.logits);

  const root = document.getElementById('examples');
  if (f.examples.length === 0) {
    root.innerHTML = '<div class="dead">This transform did not fire on any token in the sampled corpus.</div>';
    return;
  }
  for (const ex of f.examples) {
    root.appendChild(renderExample(ex, f.max_activation));
  }
}

function addMeta(parent, text) {
  const s = document.createElement('span');
  s.textContent = text;
  parent.appendChild(s);
}

function link(href, text) {
  const a = document.createElement('a');
  a.href = href;
  a.textContent = text;
  return a;
}

function renderExample(ex, globalMax) {
  const wrap = document.createElement('div');
  wrap.className = 'example';

  const header = document.createElement('div');
  header.className = 'example-header';
  header.textContent = `peak ${ex.peak_activation.toFixed(3)} · position ${ex.peak_token_pos} of ${ex.tokens.length}`;
  wrap.appendChild(header);

  const tokens = document.createElement('div');
  tokens.className = 'tokens';
  // Scale alpha by the per-feature global max so different sequences are
  // visually comparable within the same feature.
  const scale = globalMax > 0 ? globalMax : 1.0;
  for (let i = 0; i < ex.tokens.length; i++) {
    const span = document.createElement('span');
    span.className = 'tok';
    if (i === ex.peak_token_pos) span.classList.add('peak');
    const a = ex.activations[i];
    const alpha = Math.max(0, Math.min(1, a / scale));
    span.style.backgroundColor = `rgba(220, 50, 50, ${alpha.toFixed(3)})`;
    span.title = `act ${a.toFixed(4)}`;
    // Preserve raw whitespace inside the span (CSS white-space: pre handles this).
    span.textContent = ex.tokens[i] === '' ? ' ' : ex.tokens[i];
    tokens.appendChild(span);
  }
  wrap.appendChild(tokens);

  return wrap;
}

// Activation histogram. Counts come pre-binned into log-spaced bins (the
// shared edges in metadata). We render the truncated range [10^-3, max] so
// dead/sparse bins on the right don't squash the visible portion.
function renderActHistogram(parent, counts, edges, maxAct) {
  parent.innerHTML = '';
  if (!counts || !edges || counts.length === 0) {
    parent.appendChild(elx('div', { class: 'panel-empty' },
      'No activation histogram available.'));
    return;
  }
  // Trim trailing empty bins past `maxAct` so the x-axis ends near data.
  let lastNonzero = -1;
  for (let i = 0; i < counts.length; i++) if (counts[i] > 0) lastNonzero = i;
  const visible = lastNonzero >= 0 ? lastNonzero + 1 : counts.length;
  parent.appendChild(barChart({
    counts: counts.slice(0, visible),
    edges: edges.slice(0, visible + 1),
    logX: true,
    klass: 'histogram',
  }));
  const total = counts.reduce((a, b) => a + b, 0);
  const summary = elx('div', { class: 'panel-empty', style: { color: '#555' } },
    `${total.toLocaleString()} firing tokens · max activation ${maxAct.toFixed(3)}`);
  parent.appendChild(summary);
}

function renderLogitsPanel(parent, logits) {
  parent.innerHTML = '';
  if (!logits || !logits.top_pos || logits.top_pos.length === 0) {
    parent.appendChild(elx('div', { class: 'panel-empty' },
      'No logit attribution available — feature appears degenerate (zero singular value) or logits were not computed.'));
    return;
  }
  const wrap = elx('div', { class: 'logits-panel' });
  wrap.appendChild(logitsTable('boosts', logits.top_pos, 'pos'));
  wrap.appendChild(logitsTable('suppresses', logits.top_neg, 'neg'));
  parent.appendChild(wrap);

  const hist = logits.histogram;
  if (hist && hist.counts && hist.counts.length > 0) {
    parent.appendChild(elx('div', { class: 'panel-empty', style: { marginTop: '0.5rem' } },
      'Logit distribution over vocabulary:'));
    parent.appendChild(barChart({
      counts: hist.counts,
      edges: hist.bin_edges,
      logX: false,
      klass: 'histogram',
    }));
  }
}

function logitsTable(label, rows, kind) {
  const t = elx('table', { class: `logits-table ${kind}` });
  const thead = elx('thead');
  thead.appendChild(elx('tr', {},
    elx('th', { colspan: '2' }, label)));
  t.appendChild(thead);
  const tbody = elx('tbody');
  for (const r of rows) {
    const tr = elx('tr');
    tr.appendChild(elx('td', { class: 'tok' }, r.token === '' ? ' ' : r.token));
    tr.appendChild(elx('td', { class: 'val' }, r.value.toFixed(3)));
    tbody.appendChild(tr);
  }
  t.appendChild(tbody);
  return t;
}

// Minimal SVG bar chart. `counts` and `edges` define n bars; we draw to a
// 720x120 viewBox with axis labels at the endpoints (and the midpoint for
// log-x histograms, since 10^-3 -> 10^3 spans 6 decades).
function barChart({ counts, edges, logX, klass }) {
  const W = 720, H = 120, padL = 8, padR = 8, padT = 4, padB = 16;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const n = counts.length;
  const maxC = Math.max(1, ...counts);
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('class', klass);

  const barW = innerW / n;
  for (let i = 0; i < n; i++) {
    const h = (counts[i] / maxC) * innerH;
    const rect = document.createElementNS(ns, 'rect');
    rect.setAttribute('x', String(padL + i * barW));
    rect.setAttribute('y', String(padT + innerH - h));
    rect.setAttribute('width', String(Math.max(0.5, barW - 0.5)));
    rect.setAttribute('height', String(h));
    rect.setAttribute('class', 'bar');
    svg.appendChild(rect);
  }
  // Baseline.
  const axis = document.createElementNS(ns, 'line');
  axis.setAttribute('x1', String(padL));
  axis.setAttribute('y1', String(padT + innerH + 0.5));
  axis.setAttribute('x2', String(padL + innerW));
  axis.setAttribute('y2', String(padT + innerH + 0.5));
  axis.setAttribute('class', 'axis');
  svg.appendChild(axis);

  // Endpoint labels.
  const fmtLabel = (v) => {
    if (logX) {
      const exp = Math.round(Math.log10(v));
      return Math.abs(v - Math.pow(10, exp)) < 1e-9 * Math.abs(v) ? `1e${exp}` : v.toPrecision(2);
    }
    return v.toFixed(2);
  };
  const lo = edges[0], hi = edges[edges.length - 1];
  const tLo = document.createElementNS(ns, 'text');
  tLo.setAttribute('x', String(padL));
  tLo.setAttribute('y', String(H - 4));
  tLo.setAttribute('class', 'axis-label');
  tLo.textContent = fmtLabel(lo);
  svg.appendChild(tLo);
  const tHi = document.createElementNS(ns, 'text');
  tHi.setAttribute('x', String(padL + innerW));
  tHi.setAttribute('y', String(H - 4));
  tHi.setAttribute('class', 'axis-label');
  tHi.setAttribute('text-anchor', 'end');
  tHi.textContent = fmtLabel(hi);
  svg.appendChild(tHi);
  return svg;
}

// Tiny element helper used by the histogram/logits renderers. Named `elx`
// to avoid colliding with the bundle's `el` (same idea, kept distinct so
// dashboard.js stays self-contained for the multi-file path).
function elx(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === 'class') e.className = v;
    else if (k === 'style') Object.assign(e.style, v);
    else e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return e;
}

main().catch(err => {
  document.getElementById('examples').innerHTML =
    `<div class="dead">Failed to load feature data: ${err.message}. ` +
    `If you opened this with file://, serve the directory first ` +
    `(e.g. <code>python -m http.server</code>).</div>`;
});
