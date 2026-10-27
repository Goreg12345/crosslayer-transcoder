// Uses the local sharing credentials without logging them. Set MOLT_STEERING_SMOKE=1
// to run one short, real GPU steering comparison; otherwise this is read-only.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { chromium, request } = require('playwright');

(async () => {
  const access = JSON.parse(fs.readFileSync(process.env.MOLT_SHARE_CREDENTIALS || path.join(os.homedir(), '.config/molt-dashboard-sharing/credentials.json'), 'utf8'));
  const url = process.env.MOLT_SHARE_URL || access.url;
  const anonymous = await request.newContext();
  for (const route of ['/', '/steer', '/api/health', '/sample-10m/autointerp-layer/summary.json']) {
    const response = await anonymous.get(new URL(route, url).href);
    assert.equal(response.status(), 401, `Anonymous access to ${route}`);
    assert.match(response.headers()['www-authenticate'], /Basic/);
  }
  assert.equal((await anonymous.post(new URL('/api/generate', url).href, {data:{}})).status(), 401);
  await anonymous.dispose();
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROMIUM_PATH});
  try {
    const context = await browser.newContext({httpCredentials:{username:access.username,password:access.password},viewport:{width:1500,height:1000}});
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(new URL('/#3', url).href);
    await page.waitForURL('**/sample-10m/#3');
    await page.locator('[data-method="detection"] .score-value').filter({hasText:'68.0%'}).waitFor();
    await page.frameLocator('#view').locator('.seq').first().waitFor();
    assert.equal(await page.locator('.score').count(), 4);
    assert.match(await page.locator('#feature-description').innerText(), /The gate activates/);
    await page.locator('#steer-link').click();
    await page.locator('#model-status').filter({hasText:'checkpoint step 100,000'}).waitFor();
    assert.equal(await page.locator('#feature').inputValue(), '3');
    const origin = new URL(url).origin;
    const tokenized = await page.evaluate(async () => {
      const r = await fetch('/api/tokenize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:'Hello','format':'chat'})});
      return {status:r.status,data:await r.json()};
    });
    assert.equal(tokenized.status, 200);
    assert.ok(tokenized.data.ids.length > 0);
    const crossOrigin = await context.request.post(new URL('/api/tokenize', url).href, {headers:{Origin:'https://different.example'},data:{prompt:'Hello',format:'chat'}});
    assert.equal(crossOrigin.status(), 403);
    for (const route of ['/sample-10m/activations.npz','/sample-10m/.render.lock']) {
      assert.equal((await context.request.get(new URL(route,url).href)).status(), 404);
    }
    if (process.env.MOLT_STEERING_SMOKE === '1') {
      await page.locator('.segment-text').fill('Reply with the word hello.');
      await page.locator('#max-new').fill('4');
      await page.locator('#inspect').click();
      await page.locator('#inspect-status').filter({hasText:'last input token selected'}).waitFor({timeout:120000});
      await page.locator('#mode').selectOption('set');
      await page.locator('#strength-reference').selectOption('raw');
      await page.locator('#strength').fill('2');
      await page.locator('#assign').click();
      await page.locator('#run').click();
      await page.locator('#run-status').filter({hasText:'Comparison complete'}).waitFor({timeout:120000});
      const result = await page.evaluate(() => lastResult);
      assert.ok(result.steered.trace.some(t => t.delta_l2 > 0));
      assert.ok(result.first_prediction.max_abs_logit_change > 0);
      assert.equal(await page.locator('#download').isEnabled(), true);
      console.log('Real GPU steering passed:', JSON.stringify({origin,interventions:result.steered.trace.length,maximumLogitChange:result.first_prediction.max_abs_logit_change}));
    }
    await page.screenshot({path:'/tmp/molt-shared-steering.png',fullPage:true});
    assert.deepEqual(errors, []);
    console.log('PASS: password protection, 10M explorer, four scores, iframe assets, steering UI, authenticated same-origin API, cross-origin rejection, and extraction-file isolation.');
  } finally {await browser.close();}
})().catch(error => {console.error(error);process.exit(1);});
