// Index page. One fetch of metadata.json populates the sortable table.

let allRows = [];
let sortKey = 'activation_rate';
let sortDir = 'desc';
let metadata = null;

function pad(id, width) {
  return String(id).padStart(width, '0');
}

async function main() {
  metadata = await fetch('data/metadata.json').then(r => r.json());

  document.getElementById('meta').textContent =
    `${metadata.n_features} transforms · ` +
    `${(metadata.hf_filename || metadata.ckpt_path)} · layer ${metadata.layer} · ` +
    `${metadata.n_tokens_collected.toLocaleString()} tokens from ${metadata.dashboard_dataset}`;

  // Build the row data.
  for (let i = 0; i < metadata.n_features; i++) {
    allRows.push({
      feature_id: i,
      tier: metadata.feature_tier[i],
      rank: metadata.feature_rank[i],
      activation_rate: metadata.feature_activation_rate[i],
      max_activation: metadata.feature_max_activation[i],
    });
  }

  // Populate tier filter.
  const uniqueTiers = [...new Set(metadata.feature_tier)].sort((a, b) => a - b);
  const tierSel = document.getElementById('tier-filter');
  for (const t of uniqueTiers) {
    const opt = document.createElement('option');
    opt.value = String(t);
    opt.textContent = `tier ${t} (rank ${metadata.ranks[t]})`;
    tierSel.appendChild(opt);
  }

  // Wire interactions.
  document.querySelectorAll('th[data-key]').forEach(th => {
    th.addEventListener('click', () => {
      const k = th.dataset.key;
      if (sortKey === k) {
        sortDir = sortDir === 'desc' ? 'asc' : 'desc';
      } else {
        sortKey = k;
        sortDir = (k === 'feature_id' || k === 'tier') ? 'asc' : 'desc';
      }
      render();
    });
  });
  ['change', 'input'].forEach(evt => {
    tierSel.addEventListener(evt, render);
    document.getElementById('min-rate').addEventListener(evt, render);
    document.getElementById('max-rate').addEventListener(evt, render);
  });

  render();
}

function render() {
  const tier = document.getElementById('tier-filter').value;
  const minRate = parseFloat(document.getElementById('min-rate').value) || 0;
  const maxRate = parseFloat(document.getElementById('max-rate').value);
  const maxRateOk = Number.isFinite(maxRate) ? maxRate : 1;

  const filtered = allRows.filter(r =>
    (tier === '' || String(r.tier) === tier) &&
    r.activation_rate >= minRate &&
    r.activation_rate <= maxRateOk
  );

  filtered.sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey];
    if (av < bv) return sortDir === 'asc' ? -1 : 1;
    if (av > bv) return sortDir === 'asc' ? 1 : -1;
    return a.feature_id - b.feature_id;
  });

  document.querySelectorAll('th[data-key]').forEach(th => {
    th.classList.remove('sorted-asc', 'sorted-desc');
    if (th.dataset.key === sortKey) {
      th.classList.add(sortDir === 'asc' ? 'sorted-asc' : 'sorted-desc');
    }
  });

  document.getElementById('count').textContent =
    `${filtered.length.toLocaleString()} of ${allRows.length.toLocaleString()} shown`;

  const tbody = document.querySelector('#features tbody');
  // Render up to 2000 rows; the user can filter further if they need more.
  const cap = 2000;
  const rows = filtered.slice(0, cap);
  const html = rows.map(r => {
    const isDead = r.activation_rate === 0;
    return `<tr${isDead ? ' class="dead"' : ''}>
      <td><a href="dashboard.html?feature=${r.feature_id}">#${r.feature_id}</a></td>
      <td class="num">${r.tier}</td>
      <td class="num">${r.rank}</td>
      <td class="num">${(r.activation_rate * 100).toFixed(3)}%</td>
      <td class="num">${r.max_activation.toFixed(3)}</td>
    </tr>`;
  }).join('');
  tbody.innerHTML = html;
  if (filtered.length > cap) {
    tbody.insertAdjacentHTML(
      'beforeend',
      `<tr><td colspan="5" style="color:#888;font-style:italic;text-align:center;">${filtered.length - cap} more rows hidden — narrow the filter</td></tr>`
    );
  }
}

main().catch(err => {
  document.getElementById('meta').innerHTML =
    `<span style="color:#a00">Failed to load metadata.json: ${err.message}. ` +
    `If you opened this with file://, serve the directory first ` +
    `(e.g. <code>python -m http.server</code>).</span>`;
});
