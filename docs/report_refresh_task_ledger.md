# Report refresh — 2026-11-26

Decision: Excel control exports now start at P95; the operational analyst
dashboard remains the numeric P99 detector union. Report cards are cumulative
unique-person counts at P95/P97/P99, including ties, using full OOT detector
populations before Excel filtering. Each card shows the union and its detector
breakdown so overlap cannot be double-counted.

| Task | Status | Acceptance |
|---|---|---|
| Resume GitHub validation | Done | Prior code 7a7987f CI run 45710038004 succeeded |
| Readable VAE score distribution | In progress | Central detail and accessible full tail; no silent data loss |
| P95 control workbook | Done | API/config/CLI and names P95, full-population cutoff |
| P95/P97/P99 cards | Done | Unique union count, cumulative semantics, HTML/Markdown parity |
| Four adversarial rounds | Pending | Boundaries/ties; integration; failures; browser/release |
| Update context/docs and regenerate preview | Pending | Current contracts and report reflect requested behavior |
| Publish and verify CI | Pending | Clean worktree and successful current-code workflow |

Resume point: first incomplete task. No model or scientific calibration changes
are requested. A detail viewport improves readability; the full distribution
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
  found overlapping mobile labels in the initial two-column layout. Panels are
  being stacked vertically and must be rechecked before closing this task.

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
