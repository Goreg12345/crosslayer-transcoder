const assert = require('node:assert/strict');
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH});
  try {
    const page = await browser.newPage({viewport:{width:1560,height:1080}});
    const errors = [], inference = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('request', r => {if (r.method() === 'POST') inference.push(r.url());});
    const base = process.env.MOLT_DASHBOARD_URL || 'http://100.76.58.1:8765/';
    await page.goto(base + '#3');
    await page.waitForURL('**/sample-10m/#3');
    await page.waitForFunction(() => document.getElementById('sync-status').textContent.includes('finished'));
    const summaryURL = new URL('autointerp-layer/summary.json', page.url()).href;
    const summary = await (await page.request.get(summaryURL)).json();
    const row = summary.features['3'];
    const methods = ['detection','fuzz','active_fuzz','intruder'];
    assert.match(await page.locator('#summary').innerText(), /10,000,000 held-out tokens/);
    await page.locator('#search').fill(row.title);
    await page.locator('[data-feature="3"]').click();
    assert.equal(await page.locator('.feature.selected strong').innerText(), row.title);
    assert.equal(await page.locator('#feature-title').innerText(), row.title);
    assert.equal(await page.locator('#feature-description').innerText(), row.explanation);
    assert.equal(await page.locator('.score').count(), 4);
    for (const method of methods) {
      const metrics = row.methods[method].metrics;
      assert.equal(await page.locator(`[data-method="${method}"] .score-value`).innerText(), `${(100*metrics.accuracy).toFixed(1)}%`);
      assert.match(await page.locator(`[data-method="${method}"] .score-detail`).innerText(), new RegExp(`${metrics.valid}/${metrics.total} cases scored`));
    }
    const frame = page.frameLocator('#view');
    await frame.locator('.seq').first().waitFor();
    assert.equal(await frame.locator('body > h2').isVisible(), false);
    assert.match(await page.locator('#review-link').getAttribute('href'), /features\/3.html$/);
    await page.locator('#binary').check();
    await page.waitForFunction(() => document.getElementById('view').contentWindow.location.search === '?binary=1');
    await frame.locator('.seq').first().waitFor();
    const binaryActs = await page.locator('#view').evaluate(iframe => iframe.contentWindow.eval('Object.values(DATA)[0].seqMultiGroup.flatMap(g=>g.seqGroupData.flatMap(s=>s.seqData.filter(t=>t.featAct>0).map(t=>t.featAct)))'));
    assert.ok(binaryActs.length > 0 && binaryActs.every(a => a === 1));
    await page.locator('#binary').uncheck();
    await page.locator('#search').fill('');
    await page.locator('#rank').selectOption('512');
    await page.locator('#sort').selectOption('id');
    await page.locator('#next').click();
    assert.equal(new URL(page.url()).hash, '#4');
    await page.locator('#previous').click();
    assert.equal(new URL(page.url()).hash, '#3');
    await page.locator('#rank').selectOption('');

    // An update changes the score panel without reloading the activation iframe.
    let response = structuredClone(summary);
    response.features['3'] = {...row,
      title:'Literal <img src=x onerror="window.injected=true"> title',
      explanation:'Full description with <b>literal tags</b> & punctuation.',
      status:'incomplete',
      methods:{
        detection:{status:'complete',metrics:{accuracy:0,total:10,valid:10,parse_failures:0,accuracy_ci95:[0,0.278]}},
        fuzz:{status:'unavailable',reason:'Insufficient positive evaluation windows'},
        active_fuzz:{status:'incomplete',metrics:{accuracy:2/9,total:10,valid:9,parse_failures:1,accuracy_ci95:[0.06,0.55]}}
      }};
    await page.route(summaryURL, route => route.fulfill({json:response}));
    const src = await page.locator('#view').getAttribute('src');
    const activeFrame = page.frames().find(f => f.url().includes('/features/3.html'));
    await page.evaluate(() => refreshAnnotations());
    assert.equal(await page.locator('#feature-title').innerText(), response.features['3'].title);
    assert.equal(await page.locator('#feature-description').innerText(), response.features['3'].explanation);
    assert.equal(await page.locator('#interpretation img, #interpretation b').count(), 0);
    assert.equal(await page.locator('[data-method="detection"] .score-value').innerText(), '0.0%');
    assert.equal(await page.locator('[data-method="fuzz"] .score-value').innerText(), 'Unavailable');
    assert.match(await page.locator('[data-method="active_fuzz"] .score-detail').innerText(), /9\/10 cases scored · incomplete · 1 unscored/);
    assert.equal(await page.locator('[data-method="intruder"] .score-value').innerText(), 'Pending');
    assert.equal(await page.locator('#view').getAttribute('src'), src);
    assert.equal(page.frames().find(f => f.url().includes('/features/3.html')), activeFrame);
    assert.equal(await page.evaluate(() => window.injected), undefined);

    response = structuredClone(summary);
    await page.evaluate(() => refreshAnnotations());
    response.settings.cache_sha256 = 'different cache';
    response.features['3'].title = 'WRONG SAMPLE TITLE';
    await page.evaluate(() => refreshAnnotations());
    assert.equal(await page.locator('#feature-title').innerText(), row.title);
    assert.match(await page.locator('#sync-status').innerText(), /different sample/);
    await page.unroute(summaryURL);
    await page.evaluate(() => refreshAnnotations());

    const limited = Object.values(summary.features).find(r => r.evidence_counts?.full_budget === false);
    await page.evaluate(id => show(id), limited.id);
    assert.match(await page.locator('#feature-status').innerText(), /Limited evidence/);
    assert.match(await page.locator('#feature-meta').innerText(), /evaluation windows/);
    await page.evaluate(() => show(0));
    assert.equal(await page.locator('.score-value.missing').count(), 4);
    assert.deepEqual(await page.locator('.score-value').allInnerTexts(), Array(4).fill('Unavailable'));
    assert.match(await page.locator('#feature-description').innerText(), /did not activate/);
    await page.evaluate(() => show(3));
    await page.screenshot({path:'/tmp/molt-explorer-scores.png'});
    await page.setViewportSize({width:600,height:900});
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.setViewportSize({width:1560,height:1080});

    await page.goto(new URL('/layers/layer-0/#117', base).href);
    await page.waitForFunction(() => /30 seconds|Scoring finished/.test(document.getElementById('sync-status').textContent));
    assert.match(await page.locator('#feature-id').innerText(), /Layer 0/);
    assert.equal(await page.locator('#steer-link').isVisible(), false);
    assert.equal(await page.locator('.score').count(), 4);
    assert.ok((await page.locator('#feature-description').innerText()).length > 50);
    await page.goto(new URL('/layers/layer-1/', base).href);
    assert.equal(await page.locator('.score').count(), 4);
    assert.equal(await page.locator('#steer-link').isVisible(), false);
    // Keep pending-state coverage after the live layer has gained explanations.
    const layer1SummaryURL = new URL('autointerp-layer/summary.json', page.url()).href;
    const pendingSummary = await (await page.request.get(layer1SummaryURL)).json();
    const pendingFeature = new URL(page.url()).hash.slice(1);
    pendingSummary.features[pendingFeature] = {status:'explaining'};
    await page.route(layer1SummaryURL, route => route.fulfill({json:pendingSummary}));
    await page.evaluate(() => refreshAnnotations());
    assert.match(await page.locator('#feature-title').innerText(), /Explanation pending/);
    assert.deepEqual(await page.locator('.score-value').allInnerTexts(), Array(4).fill('Pending'));
    assert.deepEqual(errors, []);
    assert.deepEqual(inference, []);
    console.log('PASS: main redirect, title selection, full description, four accurate scores, binary examples, navigation, live refresh, unavailable/incomplete/zero scores, sample validation, limited evidence, responsive layout, and other layers. No inference requests or browser errors.');
  } finally {await browser.close();}
})().catch(error => {console.error(error);process.exit(1);});
