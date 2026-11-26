# Report refresh — 2026-11-26

Decision: Excel control exports now start at P95; the operational analyst
dashboard remains the numeric P99 detector union. Report cards are cumulative
unique-person counts at P95/P97/P99, including ties, using full OOT detector
populations before Excel filtering. Each card shows the union and its detector
breakdown so overlap cannot be double-counted.

| Task | Status | Acceptance |
|---|---|---|
| Resume GitHub validation | Done | Prior code 7a7987f CI run 45710038004 succeeded |
| Readable VAE score distribution | Done | Stacked central detail and complete log1p tail; raw hover intervals |
| P95 control workbook | Done | API/config/CLI and names P95, full-population cutoff |
| P95/P97/P99 cards | Done | Unique union count, cumulative semantics, HTML/Markdown parity |
| Four adversarial rounds | Done | Boundaries/ties; integration; failures; browser/release all verified |
| Update context/docs and regenerate preview | Done | Context/config/docs regenerated; actual pipeline and example both completed |
| Publish and verify CI | Done | PR1 merged as63a4a76; PR CI45841678534 succeeded |

Requested scope completed. No model or scientific calibration changes
were made. A detail viewport improves readability; the full distribution
and out-of-detail counts remain available for extreme-case interpretation.

## Executed checks

- Round 1 science/data: card population fixtures yield cumulative unions
  100/60/20 out of 1,000, with disjoint detector tails; tie fixtures retain all
  equivalent scores. Export specialist: five science tests approved.
- Round 2 integration: HTML and Markdown agree; config/API/CLI default P95
  while dashboard default is P99. Combined relevant suites: 78 passed,
  29 subtests. Export specialist's four rounds: 5/13/5/38 tests approved.
- Round 3 failures: missing scores omit cards, invalid finite-score contract
  raises clearly; plot fixtures cover empty/nonfinite/zero/constant/negative
  input and preserve full-tail counts. Plot specialist's rounds: 3/3/4/2 tests
  approved, with five failure subcases.
- Round 4 browser/release: synthetic scores below 3,000 plus an extreme150,000
  rendered; browser observed center950/full1000 and card counts100/60/20, no
  page errors or horizontal overflow at1440/390. Manual screenshots nevertheless
  found overlapping mobile labels in the initial two-column layout. Vertically
  stacked panels were regenerated and manually inspected at390px: labels are
  readable and browser checks pass at both widths after paint.

Failed initial approach: uniform full-range histogram bins hid the body; simply
reducing the axis range would leave those coarse bins. Detail uses separate
bins, and the full view uses an explicitly labelled log1p axis with raw-score
intervals in hover. No finite extremes are dropped from the full view.

Research references (retrieved2026-11-26):
- https://plotly.com/python/histograms/
- https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html

Trade-off: two views take more vertical space, but preserve both central
resolution and extreme-case visibility. The histogram's P95 is descriptive;
its row-level denominator is distinct from unique-person OOT card counts.

## Final release evidence

- Full local suite: **487 passed, 85 subtests**, 11 non-failing numerical
  warnings; command `py -m pytest -q`, 399.45s.
- Final chart/card regressions: **18 passed, five subtests** after mobile fixes.
- Independent review: **PASS**, 121 focused tests,49 subtests and200 seeded
  population fixtures; no actionable defects found.
- Actual development-data pipeline: `py best_params_no_suite.py --quick
  --skip-dependency-check`, run068a586d93d2, exit0. Each P95 workbook has25
  unique people (20 p95 +5 p99); report unions P95=43, P97=25, P99=10.
  Analyst dashboard remains10 P99 cases. The source is the existing synthetic
  panel, so these counts demonstrate implementation behavior only.
- Browser workflows: `node tests/review_flows.cjs` passed first-use/daily/weekly/
  monthly-remap and390/768/1440 widths; no page errors.
- Example: `py tools/render_report_refresh_example.py` then
  `node tests/report_refresh_visuals.cjs` passed count conservation, extreme
  retention, cards and1440/390 widths. Screenshots:
  `artifacts/validation/report-refresh/vae-1440.png`, `vae-390.png`.
- Complexity gate: passed, no baseline increase. Snapshot commit9da3d03.
- CI: https://github.com/Mirloct/MV_v1/actions/runs/45841678534 succeeded.
  A duplicate push workflow for the same tree was cancelled; PR CI was retained.
- Release: https://github.com/Mirloct/MV_v1/pull/1 merged to main as
  `63a4a76d5c1625e749a788327ccb77db6eea5504`.

Rollback: revert PR1 as a whole to restore the prior control-export defaults
and score view; retain ignored generated artifacts separately if needed.
