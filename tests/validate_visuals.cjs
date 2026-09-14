/* Real-browser acceptance pass for the three user-facing HTML artifacts.
 * Run from the repository root with the bundled Node runtime. */

const fs = require("fs");
const path = require("path");
const { pathToFileURL } = require("url");
const { chromium } = require("C:/Users/Marco/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright");

const root = process.cwd();
const outputDir = path.join(root, "artifacts", "validation", "2026-09-14");
fs.mkdirSync(outputDir, { recursive: true });

function artifact(rel) {
  return pathToFileURL(path.join(root, rel)).href;
}

async function main() {
  const browser = await chromium.launch({
    headless: true,
    executablePath: "C:/Program Files/Google/Chrome/Application/chrome.exe",
  });
  const context = await browser.newContext({ acceptDownloads: true, viewport: { width: 1440, height: 1000 } });
  const results = [];

  // Iteration 1 — dashboard tabs, profile, and real per-observation download.
  {
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", error => errors.push(String(error)));
    await page.goto(artifact("artifacts/reports/analyst_dashboard.html"), { waitUntil: "load" });
    const expected = { if_only: 16, vae_only: 16, intersection: 10 };
    for (const [tab, count] of Object.entries(expected)) {
      await page.locator(`[data-tab-target="${tab}"]`).click();
      await page.waitForTimeout(100);
      const visible = await page.locator(`#priorityTableBody tr[data-tab="${tab}"]:visible`).count();
      if (visible !== count) throw new Error(`dashboard tab ${tab}: expected ${count}, saw ${visible}`);
    }
    await page.locator('[data-tab-target="intersection"]').click();
    const firstIntersection = page.locator('#priorityTableBody tr[data-tab="intersection"]:visible').first();
    await firstIntersection.click();
    await page.waitForTimeout(150);
    const modalState = await page.evaluate(() => ({
      overlayClass: document.querySelector("#overlay")?.className,
      activeProfile: window.ACTIVE_PROFILE?.id || null,
      openProfileType: typeof window.openProfile,
    }));
    if (!String(modalState.overlayClass).includes("open")) {
      throw new Error(`dashboard modal did not open: ${JSON.stringify({ modalState, errors })}`);
    }
    await page.locator('#downloadBtn:not([disabled])').waitFor();
    const downloadPromise = page.waitForEvent("download");
    await page.locator("#downloadBtn").click();
    const download = await downloadPromise;
    const csvPath = path.join(outputDir, "01-dashboard-observation.csv");
    await download.saveAs(csvPath);
    const csv = fs.readFileSync(csvPath, "utf8");
    const headerColumns = (csv.split(/\r?\n/, 1)[0].match(/","/g) || []).length + 1;
    if (headerColumns !== 22) throw new Error(`dashboard CSV: expected 22 columns, saw ${headerColumns}`);
    await page.screenshot({ path: path.join(outputDir, "01-dashboard-tabs-download.png"), fullPage: false });
    results.push({ iteration: 1, artifact: "dashboard", tabs: expected, downloadedColumns: headerColumns,
      screenshot: "01-dashboard-tabs-download.png", pageErrors: errors });
    await page.close();
  }

  // Iteration 2 — report experiments and absence of warning incidents.
  {
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", error => errors.push(String(error)));
    await page.goto(artifact("artifacts/reports/anomaly_report.html"), { waitUntil: "load" });
    if (await page.locator("#incidents").count()) throw new Error("report still renders warning incidents");
    const experimentsHeading = page.locator("#diagnostic-suite-experiments");
    await experimentsHeading.waitFor();
    const experiments = experimentsHeading.locator("xpath=..");
    const executed = (await experiments.textContent()).match(/EXECUTED/g)?.length || 0;
    if (executed < 6) throw new Error(`report experiments: expected >=6 EXECUTED rows, saw ${executed}`);
    await page.evaluate(() => {
      document.querySelector("#diagnostic-suite-experiments").scrollIntoView({ block: "start" });
      window.scrollBy(0, -145);
    });
    await page.screenshot({ path: path.join(outputDir, "02-report-experiments-no-warnings.png"), fullPage: false });
    results.push({ iteration: 2, artifact: "report", executedRows: executed, incidentSection: false,
      screenshot: "02-report-experiments-no-warnings.png", pageErrors: errors });
    await page.close();
  }

  // Iteration 3 — consolidated docs with Mermaid upgraded to SVG.
  {
    const page = await context.newPage();
    const errors = [];
    page.on("pageerror", error => errors.push(String(error)));
    await page.goto(artifact("docs/documentation.html"), { waitUntil: "load" });
    await page.waitForFunction(() => document.querySelectorAll("#guia-practica .mermaid svg").length >= 3,
      null, { timeout: 30000 });
    const diagrams = await page.locator("#guia-practica .mermaid svg").count();
    await page.evaluate(() => window.scrollTo(0, document.querySelector("#guia-practica").offsetTop));
    await page.screenshot({ path: path.join(outputDir, "03-docs-mermaid-guide.png"), fullPage: false });
    results.push({ iteration: 3, artifact: "documentation", mermaidSvgs: diagrams,
      screenshot: "03-docs-mermaid-guide.png", pageErrors: errors });
    await page.close();
  }

  await browser.close();
  const resultPath = path.join(outputDir, "visual-validation.json");
  fs.writeFileSync(resultPath, JSON.stringify({ generatedAt: new Date().toISOString(), results }, null, 2));
  process.stdout.write(JSON.stringify({ outputDir, results }, null, 2));
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
