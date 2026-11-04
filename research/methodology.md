# Methodology

Thresholds and drift methods fixed on 29-10-2026, and the original experiment
configuration (experiments/configs/main.yaml) committed on 29-10-2026, both
before any Phase 4 scenario was run. The Phase 7 configurations
(experiments/configs/adult_20seeds.yaml and acs_states.yaml) were each
committed before their first run, with no change to thresholds or methods.

## 1. Data

### UCI Adult
- Dataset: UCI Adult (Census Income), id 2, CC BY 4.0, downloaded by
  src/download_data.py (see data_source.md). Task: predict whether income
  is above $50K
- Cleaning, without changing meaning: stripped spaces from text values,
  turned "?" into missing values, removed the trailing "." from some income
  labels. Result: 48,842 rows, 23.9% earning >50K
- Splits (stratified by income, seed 42):

| Split | Rows | Share | Used for |
| --- | --- | --- | --- |
| Train | 29,305 | 60% | Training the models; the drift reference; the allowed categories |
| Test | 9,768 | 20% | Measuring each model's baseline |
| Monitor pool | 9,769 | 20% | Building all incoming batches |

The three splits share no rows, so the batches used for monitoring were never
seen by the models or by the reference.

### Folktables ACSIncome (real shift)
- Dataset: the ACSIncome task from Folktables (Ding et al. 2021): 2018
  American Community Survey, 1-year person records, prepared by
  src/prepare_acs.py. Same task: income above $50,000
- Features: AGEP (age), SCHL (education level, 1-24), WKHP (hours worked per
  week) as numbers; COW (class of worker), MAR (marital status), OCCP
  (occupation), POBP (place of birth), RELP (relationship), SEX, RAC1P (race)
  as categories. Census codes are stored as text ("c4720") so they are always
  read as categories, never numbers. Folktables' missing code (-1) becomes a
  missing value
- Source population: 50,000 California residents (41.4% earning >50K), split
  60/20/20 into train (30,000), test (10,000) and monitor pool (10,000),
  stratified by income, seed 42
- Target populations: 10,000 residents each from New York, Texas, Florida and
  Puerto Rico, used only as real shifted batches

## 2. Models

Two model families, each in one pipeline with its preprocessing, learned from
the training split only (scikit-learn, seed 42). Trained by
src/train_model.py (logistic regression on Adult also by src/train_baseline.py).

- Logistic regression (max_iter = 1000):
  - Numeric features: median imputation, then standardisation
  - Categorical features: most-frequent imputation, then one-hot encoding;
    categories not seen in training are ignored
- Gradient boosting (HistGradientBoostingClassifier, default settings):
  - Numeric features: median imputation
  - Categorical features: most-frequent imputation, then ordinal encoding;
    unseen categories become -1
- Adult features: 5 numeric (age, education-num, capital-gain, capital-loss,
  hours-per-week) and 7 categorical (workclass, marital-status, occupation,
  relationship, race, sex, native-country). Excluded: fnlwgt (a census
  sampling weight, not a property of the person) and education (the same
  information as education-num)

Baselines on the test split:

| Dataset | Model | ROC-AUC | PR-AUC | Accuracy | F1 | Harmful below |
| --- | --- | --- | --- | --- | --- | --- |
| Adult | Logistic regression | 0.9056 | 0.7612 | 0.8524 | 0.6589 | 0.8756 |
| Adult | Gradient boosting | 0.9288 | 0.8293 | 0.8729 | 0.7122 | 0.8988 |
| ACSIncome (CA) | Logistic regression | 0.8922 | 0.8345 | 0.8156 | 0.7762 | 0.8622 |
| ACSIncome (CA) | Gradient boosting | 0.8987 | 0.8500 | 0.8188 | 0.7816 | 0.8687 |

## 3. Data-quality checks

Each batch is read as text first, so that converting values to numbers is
itself a check rather than a silent step. Seven rules (shown for Adult):

| Rule | Fails when | Severity |
| --- | --- | --- |
| Not empty | The batch has no rows | Critical (whole batch unusable) |
| Required columns | Any of the 12 feature columns is absent | Critical |
| Numeric type | A numeric column holds a non-number (e.g. "forty") | Error (row rejected) |
| Not missing | A value is missing outside workclass, occupation, native-country | Error |
| Allowed category | A category never seen in the training split | Error |
| Range | age outside [17, 100], education-num outside [1, 16], capital-gain or capital-loss outside [0, 99999], hours-per-week outside [1, 99] | Error |
| Duplicate row | An exact copy of an earlier row in the batch (first copy kept) | Error |

- ACSIncome uses the same rules on its 10 features: ranges AGEP [17, 99],
  SCHL [1, 24], WKHP [1, 99]; missing values allowed only in COW
- Ranges were set by hand from what is plausible, not from the training
  minimum and maximum, so normal but rare values are not rejected
- Allowed categories come from the training split only. A rare but real
  category may be absent from training by chance and then be rejected. On
  ACSIncome this happened in practice: 71 of 10,000 New York residents had a
  place of birth that never occurs among the 30,000 California training rows
- Every rule is recorded for every batch, including rules with zero failures

## 4. Drift detection

### Reference data
- What: the training split (29,305 rows for Adult, 30,000 for ACSIncome)
- Why: drift means "different from what the model learned", and the models
  learned only from the training split. The test split is kept for measuring
  model performance, and the monitor pool is used only to build incoming
  batches, so the reference and the batches never share rows

### Numeric features: Kolmogorov-Smirnov test
- Features: age, education-num, capital-gain, capital-loss, hours-per-week
  (Adult); AGEP, SCHL, WKHP (ACSIncome)
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
  native-country (Adult); COW, MAR, OCCP, POBP, RELP, SEX, RAC1P (ACSIncome)
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
- alpha = 0.05 per batch, Bonferroni-corrected to 0.05 / k per feature:
  0.05 / 12 = 0.0042 for Adult, 0.05 / 10 = 0.005 for ACSIncome
- Evidence for this choice: experiment E1 (37 vs 3 false alerts per 100 stable
  batches, without vs with correction). Measured in the main experiments:
  5.1 per 100 unchanged batches on Adult and 6.2 on ACSIncome, consistent
  with the 5% per-batch target
- Chosen BEFORE running Phase 4 scenarios, so it isn't tuned to the results
- Bonferroni assumes the worst case (independent tests). The features are
  correlated, so the correction is stricter than needed and may miss small
  real shifts

### Alerts
- Quality alert when: the batch is unusable (empty or missing required
  columns: critical), or more than 1% of its rows are rejected (warning)
- Drift alert when: at least one feature's p-value is below the corrected
  threshold. One alert per batch, naming every drifted feature

## 5. Experiment design

### Runs and batches
- Each run is 16 batches of 600 rows, processed in order; the change is
  applied from batch 8 (counting from 0), so batches 0-7 are always unchanged
- Each run shuffles the monitor pool with its seed and cuts consecutive,
  non-overlapping batches (16 x 600 = 9,600 rows)
- Paired design: for a given seed, every scenario starts from the same base
  batches; only the applied change differs. Differences between scenarios are
  therefore caused by the change, not by which rows were drawn
- Experiments:

| Experiment | Config | Data | Seeds | Models | Batches |
| --- | --- | --- | --- | --- | --- |
| E3 (original) | main.yaml | Adult | 0-4 | Logistic regression | 8 x 5 x 16 = 640 |
| E4 | adult_20seeds.yaml | Adult | 0-19 | Logistic regression, gradient boosting | 8 x 20 x 16 = 2,560 |
| E5 | acs_states.yaml | ACSIncome | 0-19 | Logistic regression, gradient boosting | 6 x 20 x 16 = 1,920 |

E3's 640 batches are identical to seeds 0-4 of E4 (checked).

### Scenarios: Adult (controlled)

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

### Scenarios: ACSIncome (real shift)

| Scenario | Batches 8-15 | Inputs changed? |
| --- | --- | --- |
| A_stable | More California residents | No |
| B_corruption | California residents, same 5% corruption as Adult (on AGEP, SEX, WKHP) | Yes |
| S_NY / S_TX / S_FL / S_PR | Real, unmodified residents of New York, Texas, Florida or Puerto Rico | Yes |

- Each run takes its own non-overlapping 4,800 rows (8 x 600) of the target
  state's 10,000, in the order given by the run's seed

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

All four strategies are evaluated on the identical batches in one pass. The
monitors never see the model, so with two models every batch is monitored
once and scored by both models.

### Ground truth
Separate truths are recorded for every batch and never combined:

1. input_changed: whether the scenario altered the input data (known by
   construction; False for A, for F, and for every batch before batch 8)
2. corrupted: true for batches 8-15 of B
3. harmful, per model: whether that model's ROC-AUC on the batch, computed
   with its true labels, was more than 0.03 below its own baseline (table in
   section 2). For Adult logistic regression: baseline 0.9056 (reported as
   0.906), so harmful means below 0.8756. The batch is scored as an
   unmonitored pipeline would score it, with no rows removed

ROC-AUC rather than PR-AUC defines harm because PR-AUC also moves when the
share of >50K earners changes (as it does in D), even if the model ranks
people equally well. Over the 320 stable Adult batches (A_stable, 20 seeds),
logistic regression's ROC-AUC averaged 0.906 with a standard deviation of
0.013, so the threshold sits about 2.4 standard deviations below normal
(2.4 to 2.8 for the other dataset-model pairs).

### Metrics
- Alert precision, recall and F1, computed separately against each ground
  truth. Precision is reported as undefined when a strategy raises no alerts
- False alerts per 100 quiet batches: alerts on batches where input_changed
  is False (1,600 in E4, 1,120 in E5), scaled to 100
- Detection delay: per run, the number of batches from batch 8 to the first
  alert; averaged over the runs that alerted, with missed runs counted
  separately. Not computed for A and F, whose inputs never change
- Model impact: mean ROC-AUC and share of harmful batches after the change,
  per model
- Alert attribution: on corrupted batches, which cause each strategy reported
- Runtime: milliseconds spent on monitoring per batch

### Uncertainty and sensitivity (experiments/analyze.py)
- 95% confidence intervals from a bootstrap over seeds: 2,000 resamples,
  each drawing 20 whole seeds with replacement (random seed 0). Seeds, not
  batches, are resampled because batches within a seed are dependent under
  the paired design
- Harm-threshold sensitivity: every harm result recomputed with thresholds of
  0.02 and 0.05 below baseline. 0.03 remains the pre-registered rule; the
  other two are robustness checks only

### Comparison with Evidently (E2)
- Evidently 0.7 DataDriftPreset with default settings, run in a separate
  environment (it does not support pandas 3) on one clean and two shifted
  1,000-row Adult demo batches
- With a reference of over 1,000 rows, Evidently uses Wasserstein distance
  for numeric and Jensen-Shannon distance for categorical columns, flagging
  drift at a distance of 0.1 or more. These measure the size of a change,
  while KS and chi-square p-values measure how certain it is that a change
  exists

## 6. Reproducibility
- All randomness is seeded; the full configuration of each experiment is in
  experiments/configs/
- One command reruns each experiment (experiments/run_experiments.py
  <config>); experiments/summarize.py, analyze.py and make_figures.py rebuild
  the tables, intervals and figures
- Verified 03-11-2026: a local rerun and a clean Docker rerun (fresh dataset
  download, freshly trained model) both reproduced all 640 E3 batch results
  exactly, timings excluded (experiments/compare_runs.py)
- Verified 04-11-2026: the code changes for two datasets and two models left
  E3 unchanged (IDENTICAL), and E3 matches seeds 0-4 of E4 exactly
- Exact library versions: requirements-lock.txt (local) and
  requirements-lock-docker.txt (Docker)

## 7. Known limitations of the method
- Significant is not the same as harmful: with a reference of about 30,000
  rows, small changes become statistically clear. In E2, DriftGuard flagged
  occupation (JS 0.069) and sex (JS 0.081), both below Evidently's 0.1 size
  threshold. A drift alert says a change is real, not that it hurts the model
- Correlated features shift together (seen in the marital batch): over-sampling
  married people also moved age, relationship, occupation and sex. One
  underlying change can produce an alert listing many features, so the alert
  doesn't show which feature actually caused the change
- Evidently uses different methods (distances, threshold 0.1): in E2 the tools
  agreed on 33 of 36 feature decisions. All 3 disagreements were small or
  borderline changes where the significance and size criteria differ by design
- Two datasets, both US census income data (1994 and 2018), two model
  families and one family of drift tests. Adult changes are synthetic; the
  ACSIncome shift is real but geographic only and from one year
- D samples from the whole pool, so its batches can repeat rows from other
  batches in the same run
- Harm is defined by one ROC-AUC rule. The sensitivity analysis shows the main
  conclusions hold at 0.02 and 0.05, but whether D counts as harmful does not.
  ROC-AUC itself depends on the mix of people in a batch, which matters for D
  and for the ACSIncome states
- The paired design reuses base batches across scenarios, so one unusual base
  batch appears in several scenarios. In E4, 23 of the 289 batches harmful for
  logistic regression fall outside D and F; 18 of them are three base batches
  repeated across the six scenarios that share them
- The quality schema is learned from one population, so valid but unseen
  categories in a new population are rejected as invalid