# Drift detection methods

Threshold and methods fixed on 29-10-2026, before any Phase 4 scenario was run.

## Reference data
- What: the training split, 29,305 rows (60% of the 48,842-row cleaned Adult
  dataset, stratified by income, seed 42)
- Why: drift means "different from what the model learned", and the model
  learned only from the training split. The test split is kept for measuring
  model performance, and the monitor pool (20%) is used only to build incoming
  batches, so the reference and the batches never share rows

## Numeric features: Kolmogorov-Smirnov test
- Features: age, education-num, capital-gain, capital-loss, hours-per-week
- What it measures: the largest vertical gap between the reference's and the
  batch's cumulative distributions (statistic 0 = identical, 1 = completely
  different), and a p-value for how likely a gap that size is if nothing changed
- Assumption / weakness: KS assumes continuous data with few tied values.
  capital-gain and capital-loss are mostly zeros, and education-num has only
  16 possible values, so ties are heavy. With many ties the test becomes
  conservative (less likely to flag), and a change among the few non-zero
  values barely moves the cumulative curve. KS is also most sensitive near
  the middle of a distribution and weak at detecting changes in the tails

## Categorical features: chi-square test
- Features: workclass, marital-status, occupation, relationship, race, sex,
  native-country
- Missing values handled as: their own category ("__missing__"), so a sudden
  rise in missing values shows up as a change in proportions
- Rare categories (<1% of reference) grouped as __other__ because: chi-square
  becomes unreliable when a category is expected to appear only a few times
  in a batch. native-country has many countries far below 1%, which in a
  1,000-row batch would be expected fewer than 10 times
- Cost of that choice: a shift between two rare categories (e.g. from one rare
  country to another) is invisible, as is a new rare category appearing
  inside __other__. Unseen categories are caught by the quality checks instead

## Threshold
- alpha = 0.05 per batch, Bonferroni-corrected to 0.05 / 12 = 0.0042 per feature
- Evidence for this choice: experiment E1 (37 vs 3 false alerts per 100 stable
  batches, without vs with correction)
- Chosen BEFORE running Phase 4 scenarios, so it isn't tuned to the results
- Bonferroni assumes the worst case (independent tests). The 12 features are
  correlated, so the correction is stricter than needed and may miss small
  real shifts; Phase 4 will measure that cost

## Alerts
- Quality alert when: the batch is unusable (empty or missing required
  columns: critical), or more than 1% of its rows are rejected (warning)
- Drift alert when: at least one feature's p-value is below 0.0042 on the
  batch's valid rows. One alert per batch, naming every drifted feature

## Known limitations
- Significant is not the same as harmful: with a 29,305-row reference, small
  changes become statistically clear. In E2, DriftGuard flagged occupation
  (JS 0.069) and sex (JS 0.081), both below Evidently's 0.1 size threshold.
  A drift alert says a change is real, not that it hurts the model (tested in
  Phase 4, H4)
- Correlated features shift together (seen in the marital batch): over-sampling
  married people also moved age, relationship, occupation and sex. One
  underlying change can produce an alert listing many features, so the alert
  doesn't show which feature actually caused the change
- Evidently uses different methods (distances, threshold 0.1): Wasserstein for
  numeric, Jensen-Shannon for categorical. In E2 the tools agreed on 33 of 36
  feature decisions. All 3 disagreements were small or borderline changes
  where the significance and size criteria differ by design