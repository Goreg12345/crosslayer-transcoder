// Run with NODE_PATH pointing to a Playwright install. All inference APIs are mocked.
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
(async () => {
  const browser = await chromium.launch({headless: true});
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    let scenario = 'retry', inspectCalls = 0, generateCalls = 0;
    const requests = [], cancellations = [];
    await page.route('**/feature_stats.json', r => r.fulfill({json: [{id:29,mean_active:1.5,maximum:3}]}));
    await page.route('**/api/**', async route => {
      const path = new URL(route.request().url()).pathname;
      if (path === '/api/health') return route.fulfill({json: {ready:true,busy:true,model:'Qwen/Qwen3-4B',layer:22,global_step:100000}});
      if (path === '/api/inspect') {
        inspectCalls++;
        if (inspectCalls === 1) return route.fulfill({status:409,json:{error:'GPU is busy'}});
        return route.fulfill({json:{tokenization_id:'same-prompt',ids:[1,2,3],tokens:[0,1,2].map(i=>({position:i,id:i+1,text:'word'+i,gate:1,special:false}))}});
      }
      if (path === '/api/generate') {
        generateCalls++;
        requests.push(route.request().postDataJSON());
        if (scenario === 'invalid') return route.fulfill({status:400,json:{error:'Invalid strength'}});
        if (scenario === 'cancel-wait' || (scenario === 'retry' && generateCalls===1)) {
          return route.fulfill({status:409,json:{error:'GPU is busy'}});
        }
        if (scenario === 'accept-race') await new Promise(resolve=>setTimeout(resolve,400));
        return route.fulfill({status:202,json:{id:'our-job'}});
      }
      if (path.endsWith('/cancel')) {
        cancellations.push(path);
        return route.fulfill({json:{cancel_requested:true}});
      }
      if (path === '/api/jobs/our-job') return route.fulfill({json:{state:scenario==='accept-race'?'cancelled':'done',phase:'steered',tokens:1,baseline:'original response',steered:'steered response',result:{baseline:{text:'original response'},steered:{text:'steered response',trace:[]}}}});
      throw Error('Unexpected API request: '+path);
    });
    await page.goto(process.env.STEERING_URL || 'http://100.76.58.1:8765/steer?feature=29');
    await page.locator('#availability').filter({hasText:'Another steering request'}).waitFor();
    await page.locator('#inspect').click();
    await page.locator('#inspect-status').filter({hasText:'Waiting for another'}).waitFor();
    await page.locator('#inspect-status').filter({hasText:'last input token selected'}).waitFor();
    await page.locator('#assign').click();
    await page.locator('#run').click();
    await page.locator('#run-status').filter({hasText:'Waiting for another'}).waitFor();
    assert.equal(await page.locator('#error').isVisible(),false);
    await page.locator('#run-status').filter({hasText:'Comparison complete'}).waitFor();
    assert.equal(generateCalls,2);
    assert.deepEqual(requests[0],requests[1]);
    assert.equal(await page.locator('.token.assigned').count(),1);

    scenario='cancel-wait';
    await page.locator('#run').click();
    await page.locator('#run-status').filter({hasText:'Waiting for another'}).waitFor();
    assert.equal(await page.locator('#baseline').innerText(),'original response');
    await page.locator('#cancel').click();
    await page.locator('#run-status').filter({hasText:'Cancelled before starting'}).waitFor();
    assert.equal(cancellations.length,0); // Never cancel someone else's active job.
    assert.equal(await page.locator('#download').isEnabled(),true);
    assert.equal(await page.locator('.token.assigned').count(),1);

    scenario='accept-race';
    const response=page.waitForResponse(r=>r.url().endsWith('/api/generate'));
    await page.locator('#run').click();
    await page.locator('#cancel').click();
    await response;
    await page.locator('#run-status').filter({hasText:'Stopped. Partial results'}).waitFor();
    assert.deepEqual(cancellations,['/api/jobs/our-job/cancel']);

    scenario='invalid';const before=generateCalls;
    await page.locator('#run').click();
    await page.locator('#error').filter({hasText:'Invalid strength'}).waitFor();
    await page.waitForTimeout(900);
    assert.equal(generateCalls,before+1); // Validation errors must not be retried.
    assert.deepEqual(errors,[]);
    console.log('PASS: busy retries, preserved requests/results, pending cancellation, acceptance race and non-retryable validation errors; no GPU inference.');
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exit(1)});
