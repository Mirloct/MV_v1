/* Inputs: generated synthetic long-tail report. Outputs: validation screenshots.
 * Created/Last modified: 2026-11-26. Changelog: cards and chart browser checks. */
const fs = require('fs');
const path = require('path');
const {pathToFileURL} = require('url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'C:/Users/Marco/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

async function main() {
  const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH || (process.platform==='win32'?'C:/Program Files/Google/Chrome/Application/chrome.exe':undefined)});
  const page=await browser.newPage(); const errors=[];
  page.on('pageerror',e=>errors.push(String(e)));
  const output=path.resolve('artifacts/validation/report-refresh'); fs.mkdirSync(output,{recursive:true});
  try {
    await page.goto(pathToFileURL(path.resolve('artifacts/reports/report_refresh_example/anomaly_report.html')).href);
    await page.waitForFunction(()=>document.getElementById('fig-scores-vae')?.data?.length===2);
    const counts=await page.locator('#oot-percentile-counts .value').allTextContents();
    if(JSON.stringify(counts)!==JSON.stringify(['100','60','20']))throw Error(`card counts ${counts}`);
    const chart=await page.evaluate(()=>{
      const d=document.getElementById('fig-scores-vae').data;
      return {center:d[0].y.reduce((a,b)=>a+b,0),full:d[1].y.reduce((a,b)=>a+b,0),extreme:d[1].customdata.at(-1)[1]};
    });
    if(chart.full!==1000 || chart.center!==950 || Math.abs(chart.extreme-150000)>0.001)throw Error(JSON.stringify(chart));
    for(const width of [1440,390]) {
      await page.setViewportSize({width,height:1000});
      await page.evaluate(()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))));
      await page.locator('#fig-scores-vae').screenshot({path:path.join(output,`vae-${width}.png`)});
      if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error(`overflow ${width}: `+JSON.stringify(await page.evaluate(()=>[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+1).slice(0,10).map(e=>({tag:e.tagName,id:e.id,cls:e.getAttribute('class'),right:e.getBoundingClientRect().right})))));
    }
    if(errors.length)throw Error(errors.join('\n'));
    console.log(JSON.stringify({cards:counts,chart,widths:[1440,390],errors},null,2));
  } finally {await browser.close();}
}
main().catch(e=>{console.error(e);process.exitCode=1;});
