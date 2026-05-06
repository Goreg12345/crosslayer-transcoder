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

main().catch(err => {
  document.getElementById('examples').innerHTML =
    `<div class="dead">Failed to load feature data: ${err.message}. ` +
    `If you opened this with file://, serve the directory first ` +
    `(e.g. <code>python -m http.server</code>).</div>`;
});
