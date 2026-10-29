# Experiment log

## E1: False alerts on stable batches (date: 29-10-2026)

- Setup: 100 batches of 1,000 rows sampled from the monitor pool (seeds 0-99);
  reference = training split; KS for 5 numeric, chi-square for 7 categorical features
- Result: no correction 37 / 100 batches alerted; Bonferroni 3 / 100
- Interpretation: Without correction, more than a third of batches where nothing changed raised a drift alert. That's because each batch runs 12 tests at p < 0.05, so one test is likely to cross the threshold by chance. If the features were independent, about 46% of batches would alert (1 - 0.95^12). The observed 37% is lower, probably because correlated features (e.g. relationship and marital-status) tend to raise false alarms together rather
than separately. Bonferroni (p < 0.05/12 ≈ 0.0042 per feature) cut false alerts to 3%, within the 5% per-batch target it is designed to guarantee. With only 100 batches this estimate is rough (±2-3 points), but the gap between 37 and 3 is far larger than that noise. 
- Conclusion: per-feature testing without correction is unusable for monitoring, so Bonferroni is adopted as the default before Phase 4. The expected cost is lower sensitivity to small real shifts, which Phase 4 will measure.
- Limitation: batches are drawn from the same ~9,800-row pool, so they overlap and are not fully independent samples