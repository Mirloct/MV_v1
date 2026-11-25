/* Browser acceptance: first use, daily review, weekly reload, monthly replacement.
 * Inputs: generated artifacts/reports/analyst_dashboard.html. Outputs: JSON evidence.
 * Created/Last modified: 2026-11-25. Changelog: added state/tab/download checks. */
const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || 'C:/Users/Marco/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

async function checkTabs(page) {
  for (const tab of ['if_only','vae_only','intersection','reviewed']) {
    await page.locator(`[data-tab-target="${tab}"]`).click();
    await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
    const state = await page.evaluate(() => ({
      visible: [...document.querySelectorAll('#priorityTableBody tr[data-profile]')].filter(r=>getComputedStyle(r).display!=='none').length,
      count: Number(document.getElementById('queueKpiValue').textContent.replace(/[^0-9]/g,''))
    }));
    if(state.visible !== state.count) throw Error(`incoherent ${tab}: ${JSON.stringify(state)}`);
  }
}

async function main() {
  const browser = await chromium.launch({headless:true, executablePath:process.env.CHROME_PATH || (process.platform==='win32' ? 'C:/Program Files/Google/Chrome/Application/chrome.exe' : undefined)});
  const context = await browser.newContext({acceptDownloads:true});
  const page = await context.newPage();
  const errors=[];
  page.on('pageerror',e=>errors.push(String(e)));
  const url=pathToFileURL(path.resolve('artifacts/reports/analyst_dashboard.html')).href;
  try {
    await page.goto(url); await page.evaluate(()=>localStorage.clear()); await page.reload();
    await checkTabs(page);
    const key=await page.evaluate(()=>Object.keys(PROFILES)[0]);
    if(!key) throw Error('fixture must contain review cases');
    await page.evaluate(k=>changeCaseStatus(k,'en_revision'),key);
    await checkTabs(page);
    await page.reload(); await checkTabs(page);
    const status=await page.evaluate(k=>caseFor(k).status,key);
    if(status!=='en_revision') throw Error('weekly reload lost case state');
    await page.evaluate(k=>changeCaseStatus(k,'cerrado'),key); await checkTabs(page);
    await page.evaluate(k=>openProfile(k),key);
    const download=page.waitForEvent('download'); await page.locator('#downloadBtn').click();
    if(!(await download).suggestedFilename().endsWith('_historial_completo.csv')) throw Error('history download missing');
    await page.keyboard.press('Escape');
    for(const width of [390,768,1440]) {
      await page.setViewportSize({width,height:900}); await checkTabs(page);
      const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);
      if(overflow) throw Error(`page overflow ${width}`);
    }
    // A monthly replacement may reorder profiles; entity ID owns case state.
    const retained=await page.evaluate(k=>{const original=PROFILES[k];PROFILES.monthly=original;return caseFor('monthly').status;},key);
    if(retained!=='cerrado') throw Error('monthly profile remap lost case state');
    if(errors.length) throw Error(errors.join('\n'));
    console.log(JSON.stringify({firstUse:'PASS',daily:'PASS',weekly:'PASS',monthlyRemap:'PASS',widths:[390,768,1440],errors},null,2));
  } finally {await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
