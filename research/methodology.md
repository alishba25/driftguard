# Methodology

Thresholds and drift methods fixed on 29-10-2026, and the experiment
configuration (experiments/configs/main.yaml) committed on 29-10-2026, both
before any Phase 4 scenario was run.

## 1. Data

- Dataset: UCI Adult (Census Income), id 2, CC BY 4.0, downloaded by
  src/download_data.py (see data_source.md). Task: predict whether income
  is above $50K
- Cleaning, without changing meaning: stripped spaces from text values,
  turned "?" into missing values, removed the trailing "." from some income
  labels. Result: 48,842 rows, 23.9% earning >50K
- Splits (stratified by income, seed 42):

| Split | Rows | Share | Used for |
| --- | --- | --- | --- |
| Train | 29,305 | 60% | Training the model; the drift reference; the allowed categories |
| Test | 9,768 | 20% | Measuring the baseline model |
| Monitor pool | 9,769 | 20% | Building all incoming batches |

The three splits share no rows, so the batches used for monitoring were never
seen by the model or by the reference.

## 2. Model

- Logistic regression (scikit-learn, max_iter = 1000, seed 42) in one
  pipeline with its preprocessing:
  - 5 numeric features (age, education-num, capital-gain, capital-loss,
    hours-per-week): median imputation, then standardisation
  - 7 categorical features (workclass, marital-status, occupation,
    relationship, race, sex, native-country): most-frequent imputation, then
    one-hot encoding; categories not seen in training are ignored
- Excluded: fnlwgt (a census sampling weight, not a property of the person)
  and education (the same information as education-num)
- All preprocessing is learned from the training split only
- Baseline on the test split: ROC-AUC 0.906, PR-AUC 0.761, accuracy 0.852,
  precision 0.737, recall 0.596, F1 0.659

## 3. Data-quality checks

Each batch is read as text first, so that converting values to numbers is
itself a check rather than a silent step. Seven rules:

| Rule | Fails when | Severity |
| --- | --- | --- |
| Not empty | The batch has no rows | Critical (whole batch unusable) |
| Required columns | Any of the 12 feature columns is absent | Critical |
| Numeric type | A numeric column holds a non-number (e.g. "forty") | Error (row rejected) |
| Not missing | A value is missing outside workclass, occupation, native-country | Error |
| Allowed category | A category never seen in the training split | Error |
| Range | age outside [17, 100], education-num outside [1, 16], capital-gain or capital-loss outside [0, 99999], hours-per-week outside [1, 99] | Error |
| Duplicate row | An exact copy of an earlier row in the batch (first copy kept) | Error |

- Ranges were set by hand from what is plausible, not from the training
  minimum and maximum, so normal but rare values are not rejected
- Allowed categories come from the training split only. A rare but real
  category may be absent from training by chance and then be rejected
- Every rule is recorded for every batch, including rules with zero failures

## 4. Drift detection

### Reference data
- What: the training split, 29,305 rows
- Why: drift means "different from what the model learned", and the model
  learned only from the training split. The test split is kept for measuring
  model performance, and the monitor pool is used only to build incoming
  batches, so the reference and the batches never share rows

### Numeric features: Kolmogorov-Smirnov test
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

### Categorical features: chi-square test
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

### Threshold
- alpha = 0.05 per batch, Bonferroni-corrected to 0.05 / 12 = 0.0042 per feature
- Evidence for this choice: experiment E1 (37 vs 3 false alerts per 100 stable
  batches, without vs with correction)
- Chosen BEFORE running Phase 4 scenarios, so it isn't tuned to the results
- Bonferroni assumes the worst case (independent tests). The 12 features are
  correlated, so the correction is stricter than needed and may miss small
  real shifts

### Alerts
- Quality alert when: the batch is unusable (empty or missing required
  columns: critical), or more than 1% of its rows are rejected (warning)
- Drift alert when: at least one feature's p-value is below 0.0042. One alert
  per batch, naming every drifted feature

## 5. Experiment design

### Runs and batches
- Settings (experiments/configs/main.yaml): 5 seeds (0-4); each run is 16
  batches of 600 rows, processed in order; the change is applied from batch 8
  (counting from 0), so batches 0-7 are always unchanged
- Each run shuffles the monitor pool with its seed and cuts consecutive,
  non-overlapping batches (16 x 600 = 9,600 of 9,769 rows)
- Paired design: for a given seed, every scenario starts from the same base
  batches; only the applied change differs. Differences between scenarios are
  therefore caused by the change, not by which rows were drawn
- Total: 8 scenarios x 5 seeds x 16 batches = 640 batches

### Scenarios

| Scenario | Change applied to batches 8-15 | Inputs changed? |
| --- | --- | --- |
| A_stable | None | No |
| B_corruption | 5% of rows broken, split evenly over five problems: missing age, sex = "Unknown", age = 250, hours-per-week = "forty", duplicated row | Yes |
| C_shift_low / mid / high | hours-per-week + 2 / + 5 / + 10, then rounded and clipped to [1, 99] | Yes |
| D_sudden | Batch rebuilt as 70% Married-civ-spouse (normally about 46%), sampled from the whole monitor pool | Yes |
| E_gradual | age + 1.25, + 2.5, ... + 10 years (growing linearly over the 8 batches), clipped to [17, 100] | Yes |
| F_concept | Income label flipped for 35% of Married-civ-spouse rows; inputs untouched | No |

- C, D and E keep every row valid, so they test drift detection, not the
  quality checks
- Every applied change and its severity is recorded per batch as ground
  truth, separately from anything the monitors receive

### What the monitors see
- Each batch is written to CSV text and read back through the pipeline's own
  reader, so monitors receive exactly what a real file would give them
- The income column is removed before monitoring. Labels are used only to
  compute the ground truth

### Strategies

| Strategy | Alerts when |
| --- | --- |
| None | Never |
| Quality only | The batch is unusable, or more than 1% of rows are rejected |
| Drift only | Any feature drifts, tested on the raw batch (non-numbers in numeric columns become missing; invalid categories are kept) |
| Combined | A quality alert, or any feature drifts on the batch's valid rows |

All four strategies are evaluated on the identical batches in one pass.

### Ground truth
Two separate truths are recorded for every batch and never combined:

1. input_changed: whether the scenario altered the input data (known by
   construction; False for A, for F, and for every batch before batch 8)
2. harmful: whether the model's ROC-AUC on the batch, computed with its true
   labels, was more than 0.03 below the baseline (0.906), i.e. below 0.876.
   The batch is scored as an unmonitored pipeline would score it, with no
   rows removed

ROC-AUC rather than PR-AUC defines harm because PR-AUC also moves when the
share of >50K earners changes (as it does in D), even if the model ranks
people equally well. On unchanged batches ROC-AUC averaged 0.904 with a
standard deviation of 0.011, so the threshold sits about 2.6 standard
deviations below normal.

### Metrics
- Alert precision, recall and F1, computed separately against each ground
  truth. Precision is reported as undefined when a strategy raises no alerts
- False alerts per 100 quiet batches: alerts on batches where
  input_changed is False (400 batches), scaled to 100
- Detection delay: per run, the number of batches from batch 8 to the first
  alert; averaged over the runs that alerted, with missed runs counted
  separately. Not computed for A and F, whose inputs never change
- Model impact: mean ROC-AUC and share of harmful batches after the change
- Alert attribution: on corrupted batches, which cause each strategy reported
- Runtime: milliseconds spent on monitoring per batch

### Comparison with Evidently (E2)
- Evidently 0.7 DataDriftPreset with default settings, run in a separate
  environment (it does not support pandas 3) on one clean and two shifted
  1,000-row demo batches
- With a reference of over 1,000 rows, Evidently uses Wasserstein distance
  for numeric and Jensen-Shannon distance for categorical columns, flagging
  drift at a distance of 0.1 or more. These measure the size of a change,
  while KS and chi-square p-values measure how certain it is that a change
  exists

## 6. Reproducibility
- All randomness is seeded; the full configuration is in
  experiments/configs/main.yaml
- One command reruns the experiment (experiments/run_experiments.py) and one
  rebuilds the summary tables (experiments/summarize.py)
- Verified 03-11-2026: a local rerun and a clean Docker rerun (fresh dataset
  download, freshly trained model) both reproduced all 640 batch results
  exactly, timings excluded (experiments/compare_runs.py)
- Exact library versions: requirements-lock.txt (local) and
  requirements-lock-docker.txt (Docker)

## 7. Known limitations of the method
- Significant is not the same as harmful: with a 29,305-row reference, small
  changes become statistically clear. In E2, DriftGuard flagged occupation
  (JS 0.069) and sex (JS 0.081), both below Evidently's 0.1 size threshold.
  A drift alert says a change is real, not that it hurts the model
- Correlated features shift together (seen in the marital batch): over-sampling
  married people also moved age, relationship, occupation and sex. One
  underlying change can produce an alert listing many features, so the alert
  doesn't show which feature actually caused the change
- Evidently uses different methods (distances, threshold 0.1): in E2 the tools
  agreed on 33 of 36 feature decisions. All 3 disagreements were small or
  borderline changes where the significance and size criteria differ by design
- One dataset (1994 US census), one model and one family of drift tests;
  all changes are synthetic
- Five seeds; D samples from the whole pool, so its batches can repeat rows
  from other batches in the same run
- Harm is defined by one ROC-AUC rule; other rules could classify batches
  differently, and ROC-AUC itself depends on the mix of people in a batch
- The paired design reuses base batches across scenarios, so one unusual base
  batch appears in every scenario. 13 of the 81 harmful batches fall outside
  D and F, mostly repeats of the same few unusual base batches