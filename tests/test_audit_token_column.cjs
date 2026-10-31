const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../tools/molt_steering.html'), 'utf8');
assert.ok(html.includes('<th>Input position</th><th>Token</th><th>Source</th>'));
const source = html.slice(html.indexOf('function audit('), html.indexOf("el('run').onclick="));
const elements = {};
const context = vm.createContext({
  el(id) {
    return elements[id] ??= {replaceChildren(...rows) {this.rows = rows}};
  },
  tableRow(values) {
    return {children: values.map(textContent => ({textContent}))};
  },
});
vm.runInContext(source, context);

context.audit({steered: {trace: [
  {position: 2, source: 'prompt', token_id: 3, token_text: '<user>', before: 1, after: 0, delta_l2: 2},
  {position: 3, source: 'generated', token_id: 7, token_text: ' next\n\t', before: 2, after: 0, delta_l2: 3},
  {position: 4, source: 'generated', token_id: 8, token_text: '', before: 0, after: 1, delta_l2: 1},
]}});
const rows = elements.trace.rows;
assert.equal(rows[0].children.length, 6);
assert.equal(rows[0].children[1].textContent, '<user>');
assert.equal(rows[0].children[1].title, 'Token ID 3');
assert.equal(rows[1].children[1].textContent, '·next↵⇥');
assert.equal(rows[1].children[3].textContent, '2.0000');
assert.equal(rows[1].children[4].textContent, '0.0000');
assert.equal(rows[2].children[1].textContent, '∅');
context.audit({steered: {trace: [
  {position: 3, source: 'generated', before: 0, after: 1, delta_l2: 2},
]}});
assert.equal(elements.trace.rows[0].children[1].textContent, '—');
assert.equal(elements.trace.rows[0].children[1].title, '');
console.log('PASS: six-column token audit, whitespace, token IDs, numeric columns and legacy results');
