// NODE_PATH=/path/to/node_modules node tests/browser/steering_conversation.cjs
// All inference endpoints are mocked; no GPU work is submitted.
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const { serialize } = require('../../tools/molt_conversation.js');
const turn = (role,text,kind='text') => ({role,segments:[{kind,text}]});
const fixture = {turns:[turn('system','Be helpful.'),turn('user','First question'),
  {role:'assistant',segments:[{kind:'think',text:'Historical reasoning'},{kind:'text',text:'First answer'}]},
  turn('user','Follow-up'),turn('assistant','Let me consider','think')],ending:'continue_thinking'};
const expected = '<|im_start|>system\nBe helpful.<|im_end|>\n<|im_start|>user\nFirst question<|im_end|>\n<|im_start|>assistant\n<think>\nHistorical reasoning\n</think>\n\nFirst answer<|im_end|>\n<|im_start|>user\nFollow-up<|im_end|>\n<|im_start|>assistant\n<think>\nLet me consider';
assert.equal(serialize(fixture),expected);
assert.equal(serialize({turns:[turn('raw','  literal\n')],ending:'closed'}),'  literal\n');
assert.equal(serialize({turns:[turn('assistant','Partial')],ending:'continue_last'}),'<|im_start|>assistant\nPartial');
assert.throws(()=>serialize({turns:[turn('user','x')],ending:'continue_thinking'}),/final segment is thinking/);
assert.throws(()=>serialize({turns:[],ending:'closed'}),/at least one/);
assert.throws(()=>serialize({turns:[turn('unknown','x')],ending:'closed'}),/unknown role/);
for (const [ending,suffix] of Object.entries({assistant:'<|im_start|>assistant\n',assistant_think:'<|im_start|>assistant\n<think>\n',assistant_answer:'<|im_start|>assistant\n<think>\n\n</think>\n\n',closed:''})) {
  assert.equal(serialize({turns:[turn('user','Hello')],ending}),'<|im_start|>user\nHello<|im_end|>\n'+suffix);
}
(async()=>{
  const browser=await chromium.launch({headless:true});
  try {
    const page=await browser.newPage({viewport:{width:1440,height:1100}});
    const errors=[], inspected=[], generated=[];
    page.on('pageerror', e=>errors.push(e.message));
    await page.route('**/feature_stats.json',r=>r.fulfill({json:[{id:29,mean_active:1.5,maximum:3}]}));
    await page.route('**/api/**', async r=>{
      const path=new URL(r.request().url()).pathname;
      if(path==='/api/health')return r.fulfill({json:{ready:true,busy:false,model:'Qwen/Qwen3-4B',layer:22,global_step:100000}});
      if(path==='/api/inspect'){
        inspected.push(r.request().postDataJSON());
        return r.fulfill({json:{tokenization_id:'fixture',ids:[1,2,3],tokens:[0,1,2].map(i=>({position:i,id:i+1,text:'token'+i,gate:1,special:false}))}});
      }
      if(path==='/api/generate'){generated.push(r.request().postDataJSON());return r.fulfill({status:202,json:{id:'test'}})}
      if(path==='/api/jobs/test')return r.fulfill({json:{state:'done',baseline:'Baseline',steered:'Steered',result:{baseline:{text:'Baseline'},steered:{text:'Steered',trace:[]}}}});
      throw Error('Unexpected API '+path);
    });
    await page.goto(process.env.STEERING_URL||'http://100.76.58.1:8765/steer?feature=29');
    await page.locator('.segment-text').fill('First question');
    await page.locator('#add-assistant').click();
    await page.locator('.segment-text').last().fill('First answer');
    await page.locator('#add-think').click();
    await page.locator('.segment-text').last().fill('Historical reasoning');
    await page.locator('.conversation-turn').nth(1).locator('[title="Move segment up"]').last().click();
    await page.locator('#add-user').click();
    await page.locator('.segment-text').last().fill('Follow-up');
    await page.locator('#add-think').click();
    await page.locator('.segment-text').last().fill('Let me consider');
    await page.locator('#add-system').click();
    await page.locator('.segment-text').last().fill('Be helpful.');
    for(let i=5;i>1;i--)await page.locator(`[title="Move turn ${i} up"]`).click();
    await page.locator('#conversation-ending').selectOption('continue_thinking');
    assert.equal(await page.locator('#rendered-input').inputValue(),expected);
    assert.equal(await page.locator('[title="Move turn 1 up"]').isDisabled(),true);
    await page.locator('[title="Duplicate turn 2"]').click();
    assert.equal(await page.locator('.conversation-turn').count(),6);
    await page.locator('[title="Delete turn 3"]').click();
    assert.equal(await page.locator('#rendered-input').inputValue(),expected);
    await page.locator('#inspect').click();
    await page.locator('#inspect-status').filter({hasText:'last input token selected'}).waitFor();
    assert.equal(inspected[0].format,'raw');
    assert.equal(inspected[0].prompt,expected);
    assert.deepEqual(inspected[0].conversation,fixture);
    await page.locator('#assign').click();
    await page.locator('#run').click();
    await page.locator('#run-status').filter({hasText:'Comparison complete'}).waitFor();
    assert.equal(generated[0].prompt,expected);
    assert.equal(generated[0].tokenization_id,'fixture');
    assert.deepEqual(generated[0].conversation,fixture);
    await page.locator('.segment-text').first().fill('Changed system');
    assert.equal(await page.locator('#run').isDisabled(),true);
    assert.equal(await page.locator('.token').count(),0);
    await page.getByText('Import / export conversation JSON',{exact:true}).click();
    await page.locator('#export-conversation').click();
    const edited=JSON.parse(await page.locator('#conversation-json').inputValue());
    assert.equal(edited.turns[0].segments[0].text,'Changed system');
    await page.locator('#conversation-json').fill(JSON.stringify(fixture));
    await page.locator('#import-conversation').click();
    assert.equal(await page.locator('#rendered-input').inputValue(),expected);
    await page.locator('#conversation-json').fill('{bad');
    await page.locator('#import-conversation').click();
    assert.match(await page.locator('#error').textContent(),/Conversation import/);
    assert.equal(await page.locator('#rendered-input').inputValue(),expected);
    await page.locator('#conversation-json').fill(JSON.stringify(fixture));
    await page.locator('#import-conversation').click();
    await page.getByText('Import / export conversation JSON',{exact:true}).click();
    await page.locator('#input-preview summary').click();
    await page.screenshot({path:'/tmp/molt-conversation-builder.png',fullPage:true});
    await page.locator('#edit-raw').click();
    assert.equal(await page.locator('#format').inputValue(),'raw');
    assert.equal(await page.locator('#prompt').inputValue(),expected);
    await page.locator('#inspect').click();
    await page.locator('#inspect-status').filter({hasText:'last input token selected'}).waitFor();
    assert.equal(inspected[1].prompt,expected);
    assert.equal(inspected[1].format,'raw');
    assert.deepEqual(errors,[]);
    console.log('PASS: ChatML preservation, endings, editor actions, exact inspect/generate payload, invalidation, import/export and raw handoff; no GPU inference.');
  } finally {await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
