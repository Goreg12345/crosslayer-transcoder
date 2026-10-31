const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../tools/molt_steering.html'), 'utf8');
assert.match(html, /id="allow-negative" type="checkbox"> Allow negative gates/);
const elements = {
  mode: {value: 'set'}, 'allow-negative': {checked: false},
  'strength-reference': {value: 'raw'}, strength: {value: '-2'},
  'generated-strength': {value: '-3'}, 'mode-help': {}, feature: {value: '0'},
};
const context = vm.createContext({
  el: id => elements[id], assignments: new Map([[1, -2]]),
  renderTokens() {}, renderAssignments() {}, controls() {},
  stats: [{id: 0, mean_active: 1.5}],
});
vm.runInContext(html.slice(html.indexOf('function gateTarget('), html.indexOf('function renderTokens(')), context);
vm.runInContext(html.slice(html.indexOf('function strengthMinimum('), html.indexOf("for(const id of ['prompt'")), context);
assert.equal(context.strengthMinimum(), 0);
elements.mode.value = 'add';
assert.equal(context.strengthMinimum(), -100);
assert.equal(context.gateTarget(2, -3), 0);
elements['allow-negative'].checked = true;
assert.equal(context.gateTarget(2, -3), -1);
for (const mode of ['set', 'multiply']) {
  elements.mode.value = mode;
  assert.equal(context.strengthMinimum(), -100);
  context.modeHelp();
  assert.equal(elements.strength.min, -100);
  assert.equal(elements['generated-strength'].min, -100);
}
assert.equal(context.gateTarget(2, -1), -2);
assert.ok(context.gateTarget(0, -1) === 0);
elements.mode.value = 'set';
elements['allow-negative'].checked = false;
elements['allow-negative'].onchange();
assert.equal(context.assignments.size, 0);
assert.equal(elements.strength.value, '0');
assert.equal(elements['generated-strength'].value, '0');
assert.equal(elements.strength.min, 0);

Object.assign(context, {composeInput: () => ({prompt: 'test'}), inspected: {tokenization_id: 'test'}, activeLayer: 22});
Object.assign(elements, {
  'generated-enabled': {checked: false}, 'max-new': {value: '64'},
  temperature: {value: '0'}, 'top-p': {value: '0.95'}, seed: {value: '42'},
});
vm.runInContext(html.slice(html.indexOf('function request('), html.indexOf('function tableRow(')), context);
assert.equal(context.request().allow_negative, false);
elements['allow-negative'].checked = true;
assert.equal(context.request().allow_negative, true);
console.log('PASS: signed controls, preview targets, opt-out reset and submitted opt-in flag');
