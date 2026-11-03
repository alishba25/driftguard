# Do Monitoring Alerts Mean Model Harm? A Controlled Comparison of Data-Quality Checks and Drift Detection on Tabular Data

Alishba Ansari
Independent researcher, Mumbai, India
alishba25ansari@gmail.com

---

## Abstract

Machine-learning systems can receive input data that is malformed or that
differs from their training data, but a monitoring alert does not necessarily
mean the model is performing worse. We compare four monitoring strategies (no
monitoring, data-quality checks only, statistical drift detection only, and
both combined) on a logistic regression model for the UCI Adult income task.
Across 640 batches covering eight controlled scenarios and five random seeds,
we score every alert against two separate ground truths: whether the input
data changed, and whether the model's ROC-AUC fell by more than 0.03. Drift
detection found 97.5% of input changes with 1 false alert per 100 unchanged
batches, but only 35 of its 238 alerts (15%) fell on batches where the model
was actually harmed. A change in the label relationship harmed the model in
39 of 40 batches, while drift alerts stayed at the rate seen on stable data.
Without a quality gate, drift detection attributed all 40 corrupted batches to
a distribution change in the sex feature; combining both methods left
detection unchanged but named the correct cause. All results reproduced
exactly in an independent containerised rerun. The results illustrate that
feature-distribution monitoring cannot, on its own, establish whether a model
has degraded.

---

## 1. Introduction

A deployed machine-learning model keeps returning predictions whatever data
it receives. Incoming data can be malformed (missing values, impossible
numbers, unknown categories), or valid but different from the training data,
for example because the population has changed. Some of these changes harm
the model's predictions; many do not. Monitoring systems therefore face two
separate questions: has the data changed, and does the change matter?

Data-quality validation and statistical drift detection are both standard
practice in production ML systems [Breck et al. 2019; Schelter et al. 2018].
Prior work has shown that simple per-feature drift tests are effective
detectors of dataset shift [Rabanser et al. 2019], and that many shifts are
benign, so that detecting harmful shifts requires information about model
performance [Podkopaev & Ramdas 2022]. Less attention has been paid to how
these tools behave together inside a working batch pipeline, and to how their
alerts relate to actual changes in model performance when both are measured
on the same batches.

This paper asks: **how effectively can data-quality checks combined with
drift detection distinguish harmful distribution changes from harmless
shifts in machine-learning input data?** We build a small, end-to-end batch
pipeline and run a controlled experiment with known changes, so that every
alert can be checked against what really happened.

Our contribution is empirical and deliberately modest. The individual
components we use (per-feature drift tests, rule-based data validation, and
the observation that drift and harm differ) are all established. We
contribute:

1. A reproducible, controlled comparison of four monitoring strategies (none,
   quality-only, drift-only, combined) on a tabular classification pipeline,
   across eight scenarios covering data corruption and sudden, gradual and
   label-relationship changes.
2. An evaluation that scores every alert against two separate ground truths,
   input change and performance harm, rather than treating them as one.
3. Open code, configuration and results that rebuild every number and figure
   with a few commands, verified by an independent containerised rerun.

We test four hypotheses:

- **H1:** A data-quality gate catches malformed, missing, duplicated or
  out-of-range records that drift detection alone may not classify correctly.
- **H2:** Combining quality and drift monitoring is more informative than
  either alone, though it may not win on every metric.
- **H3:** Drift detection depends on the type and severity of the shift.
- **H4:** Feature drift does not always correspond to reduced predictive
  performance.

## 2. Related work

**Drift and concept drift.** Gama et al. [2014] and Lu et al. [2019] survey
learning under concept drift and establish the vocabulary used here: sudden
and gradual drift, and the distinction between a change in the input
distribution and a change in the relationship between inputs and labels.

**Detecting dataset shift.** Rabanser et al. [2019] compared a range of
shift detectors and found that per-feature Kolmogorov-Smirnov tests combined
with a Bonferroni correction performed comparably to multivariate tests,
despite the conservative correction; small shifts remained hard to detect.
Our drift detector follows this established design, applied to tabular data.

**Harmful versus benign shift.** Podkopaev and Ramdas [2022] argue that many
distribution shifts do not degrade a model and that a warning system should
flag only harmful ones; their method requires at least some true labels. Our
results provide a concrete pipeline-level illustration of the same gap.

**Data validation in production.** Breck et al. [2019] describe the data
validation system used in Google's TFX platform, and Schelter et al. [2018]
describe declarative data-quality verification at Amazon. Sculley et al.
[2015] identify data dependencies as a major source of hidden technical debt
in ML systems. Together these motivate treating validation as a first-class
part of monitoring.

**Tools and benchmarks.** Evidently is a widely used open-source monitoring
library; with reference data of more than 1,000 rows it uses distance
measures rather than hypothesis tests by default [Evidently AI]. TableShift
[Gardner et al. 2023] and Folktables [Ding et al. 2021] provide tabular
datasets with real, naturally occurring shifts; Ding et al. also document
limitations of the UCI Adult dataset used here.

## 3. Methods

The thresholds and drift-detection methods were fixed, and the experiment
configuration was committed to version control, before any experimental
scenario was run.

### 3.1 Data and splits

We use the UCI Adult (Census Income) dataset [Becker & Kohavi 1996], where
the task is to predict whether a person's income exceeds $50K. Cleaning was
limited to stripping spaces from text values, converting "?" to missing
values, and removing a trailing full stop from some income labels, leaving
48,842 rows of which 23.9% earn more than $50K.

The data was split, stratified by income with a fixed seed, into a training
split (29,305 rows, 60%), a test split (9,768 rows, 20%) and a monitor pool
(9,769 rows, 20%). The training split is used to fit the model, as the
reference for drift detection, and to define allowed categories; the test
split measures the baseline model; and every incoming batch is built from
the monitor pool. The three splits share no rows.

### 3.2 Model

We use logistic regression [Pedregosa et al. 2011] inside a single pipeline
with its preprocessing. The five numeric features (age, education-num,
capital-gain, capital-loss, hours-per-week) are median-imputed and
standardised; the seven categorical features (workclass, marital-status,
occupation, relationship, race, sex, native-country) are imputed with the most
frequent value and one-hot encoded, ignoring categories not seen in training.
We exclude fnlwgt, a census sampling weight, and education, which duplicates
education-num. All preprocessing is learned from the training split only. On
the test split the baseline model achieved ROC-AUC 0.906, PR-AUC 0.761,
accuracy 0.852, precision 0.737, recall 0.596 and F1 0.659.

### 3.3 Data-quality checks

Each batch is first read as text, so that converting values to numbers is
itself a check rather than a silent step. Seven rules are applied (Table 1).
Critical failures make the whole batch unusable; errors cause only the
affected rows to be rejected.

**Table 1.** Data-quality rules.

| Rule | Fails when | Severity |
| --- | --- | --- |
| Not empty | The batch has no rows | Critical |
| Required columns | Any of the 12 feature columns is absent | Critical |
| Numeric type | A numeric column holds a non-number | Error |
| Not missing | A value is missing outside workclass, occupation and native-country | Error |
| Allowed category | A category never seen in the training split | Error |
| Range | age outside [17, 100], education-num outside [1, 16], capital-gain or capital-loss outside [0, 99999], hours-per-week outside [1, 99] | Error |
| Duplicate row | An exact copy of an earlier row in the batch (first copy kept) | Error |

Ranges were set from what is plausible rather than from the training minimum
and maximum, so that normal but rare values are not rejected. Every rule is
recorded for every batch, including rules with no failures.

### 3.4 Drift detection

Each batch is compared with the training split, since drift means "different
from what the model learned". Numeric features are tested with the two-sample
Kolmogorov-Smirnov (KS) test, which measures the largest gap between the two
cumulative distributions; categorical features are tested with a chi-square
test of their category proportions [Virtanen et al. 2020]. For categorical
features, missing values are treated as their own category, so a sudden rise
in missing values is visible, and categories with less than 1% of the
reference are grouped together, because chi-square is unreliable for
categories expected only a few times per batch.

With 12 features tested per batch, a per-test threshold of 0.05 would raise
frequent false alarms. We therefore apply a Bonferroni correction, flagging a
feature only when its p-value is below 0.05 / 12 = 0.0042. In a preliminary
test on 100 stable 1,000-row batches, this reduced batches with false alerts
from 37 to 3. Because the features are correlated, the correction is stricter
than strictly necessary.

A quality alert is raised when a batch is unusable or more than 1% of its
rows are rejected; a drift alert is raised when at least one feature drifts,
naming every drifted feature.

### 3.5 Experiment design

**Runs.** Each run is a sequence of 16 batches of 600 rows, processed in
order, built by shuffling the monitor pool with the run's seed and cutting
consecutive, non-overlapping batches. A scenario's change is applied from
batch 8 onwards, so batches 0-7 are always unchanged. Each scenario is run
with five seeds, giving 8 scenarios x 5 seeds x 16 batches = 640 batches.
For a given seed, every scenario starts from the same base batches, so
differences between scenarios are caused by the applied change rather than by
which rows were drawn.

**Scenarios.** Table 2 lists the eight scenarios. Scenarios C, D and E keep
every row valid, so they test drift detection rather than the quality checks.
Each applied change and its severity is recorded per batch as ground truth,
separately from anything the monitors receive.

**Table 2.** Scenarios, applied to batches 8-15.

| Scenario | Change applied | Inputs changed? |
| --- | --- | --- |
| A_stable | None | No |
| B_corruption | 5% of rows broken, split evenly over missing age, sex = "Unknown", age = 250, hours-per-week = "forty", and duplicated rows | Yes |
| C_shift_low / mid / high | hours-per-week increased by 2, 5 or 10, rounded and clipped to [1, 99] | Yes |
| D_sudden | Batch rebuilt to be 70% Married-civ-spouse (about 46% in the full dataset) | Yes |
| E_gradual | age increased by 1.25 years at batch 8, growing linearly to 10 years at batch 15, clipped to [17, 100] | Yes |
| F_concept | Income label flipped for 35% of Married-civ-spouse rows; inputs unchanged | No |

**What the monitors see.** Each batch is written to CSV and read back
through the pipeline's own reader, so monitors receive exactly what a real
file would provide. The income column is removed before monitoring; labels
are used only to compute the ground truth.

**Strategies.** Four strategies are evaluated on the identical batches.
*None* never alerts. *Quality only* raises the quality alert described above.
*Drift only* tests every feature on the raw batch, where non-numbers in
numeric columns become missing and invalid categories are kept. *Combined*
raises a quality alert, or a drift alert computed on the batch's valid rows.

**Ground truth.** Two truths are recorded for every batch and never combined.
The first, *input changed*, is known by construction: it is false for A, for
F, and for every batch before batch 8. The second, *harmful*, is true when the
model's ROC-AUC on the batch, computed with its true labels and with no rows
removed, is more than 0.03 below the baseline of 0.906 (that is, below 0.876).
We use ROC-AUC rather than PR-AUC because PR-AUC also moves when the share of
high earners changes, as it does in D, even when the model ranks people
equally well. On unchanged batches ROC-AUC averaged 0.904 with a standard
deviation of 0.011, so the threshold lies about 2.6 standard deviations below
normal.

**Metrics.** We report alert precision, recall and F1 against each ground
truth separately, treating precision as undefined for a strategy that raises
no alerts; false alerts per 100 batches whose inputs did not change (400
batches); detection delay, the number of batches from batch 8 to a run's first
alert, averaged over runs that alerted, with missed runs counted separately;
the mean ROC-AUC and share of harmful batches after the change; which cause
each strategy reported for corrupted batches; and monitoring runtime.

**Comparison with Evidently.** Separately, we compared our detector with
Evidently's default data drift preset (version 0.7) on one clean and two
shifted 1,000-row demonstration batches.

## 4. Results

### 4.1 Overall comparison

Table 3 summarises each strategy against both ground truths.

**Table 3.** Strategy performance over all 640 batches.

| Strategy | Recall (input change) | Precision (input change) | False alerts / 100 quiet batches | Share of alerts on harmful batches | Harmful batches alerted |
| --- | --- | --- | --- | --- | --- |
| None | 0.00 | undefined | 0 | undefined | 0 of 81 |
| Quality only | 0.17 | 1.00 | 0 | 5% (2 of 40) | 2 of 81 |
| Drift only | 0.975 | 0.983 | 1.0 | 15% (35 of 238) | 35 of 81 |
| Combined | 0.975 | 0.983 | 1.0 | 15% (35 of 238) | 35 of 81 |

"Quiet" batches are the 400 whose inputs were not changed. "Harmful" means a
ROC-AUC more than 0.03 below the 0.906 baseline.

Drift-only and combined monitoring made identical alert decisions on every
batch. Both detected input changes with high recall and precision, but most
of their alerts fell on batches where the model was not harmed.

### 4.2 Detection by scenario

![Figure 1](figures/fig1_alert_rate_by_strategy.png)

**Figure 1.** Share of changed batches (batches 8-15, all seeds) that raised
an alert, by strategy and scenario.

Quality checks alerted on every corrupted batch (B) and on nothing else
(Figure 1). Drift detection alerted on every changed batch in B, in all three
hours-per-week shifts (C) and in the sudden marital-status shift (D); on 85%
of batches in the gradual age shift (E); and on 5% of batches in the
label-relationship change (F), the same rate as on stable batches (A).

Even the smallest shift, +2 hours per week, was detected in every batch. A
likely explanation is that hours-per-week is strongly concentrated at 40, so
moving every value creates a large step in the cumulative distribution, which
the KS test detects easily; we did not test this explanation directly.

### 4.3 False alerts

In the preliminary test, checking 12 features at p < 0.05 without correction
raised an alert on 37 of 100 stable batches, compared with 3 of 100 with the
Bonferroni correction. In the main experiment, drift detection raised 4 alerts
on 400 unchanged batches (1.0 per 100).

### 4.4 Detection delay

![Figure 4](figures/fig4_detection_over_batches.png)

**Figure 4.** Share of the five runs with a drift alert at each batch.

Sudden changes (B, C, D) were detected at the first changed batch in every
run (Figure 4). The gradual age shift was never detected at +1.25 years
(batch 8); four of five runs alerted at +2.5 years and the fifth at +3.75
years, a mean delay of 1.2 batches. No run missed an input change entirely.

### 4.5 Alerts versus harm

![Figure 2](figures/fig2_alerts_vs_harm.png)

**Figure 2.** For each scenario, the share of changed batches that raised a
drift alert compared with the share where the model was actually harmed.

![Figure 3](figures/fig3_roc_auc_over_batches.png)

**Figure 3.** Mean ROC-AUC across the 16 batches of each run.

Only two scenarios harmed the model (Figures 2 and 3). The marital-status
shift (D) was harmful in 27 of 40 changed batches (mean ROC-AUC 0.870) and
was detected in all 40. The label-relationship change (F) was harmful in 39
of 40 batches (mean ROC-AUC 0.842), but drift alerts appeared on only 2,
matching the false-alert rate. Every other scenario stayed near the baseline
(mean ROC-AUC 0.902-0.907) despite near-constant drift alerts in B, C and E.

### 4.6 Attribution of corrupted batches

On the 40 corrupted batches, drift-only monitoring alerted every time but
reported a distribution change in sex in all 40 (alone in 37, alongside other
features in 3), because the invalid value "Unknown" appeared as a new
category. The combined strategy reported a quality problem in all 40, with
drift also present in the valid rows of 3.

### 4.7 Comparison with Evidently

On the three demonstration batches, our detector and Evidently agreed on 33
of 36 feature decisions. Both were silent on the clean batch and both flagged
only age in the age-shifted batch. All three disagreements were small changes
in the marital-status-shifted batch: our detector flagged occupation and sex
(Jensen-Shannon distances 0.069 and 0.081), both below Evidently's 0.1
threshold, while Evidently flagged hours-per-week at 0.101, which our KS test
did not. With a reference of more than 1,000 rows, Evidently used Wasserstein
distance for numeric and Jensen-Shannon distance for categorical features;
these measure the size of a change, while our p-values measure how certain it
is that a change exists, which explains the disagreements.

### 4.8 Runtime

Quality checks plus drift tests took about 170 ms per 600-row batch on a
personal laptop without a GPU.

## 5. Discussion

**H1: supported.** The quality gate caught every corrupted batch and named
the cause. Drift detection alone also alerted, but for the wrong reason:
corrupted values looked like a shift in one feature. An operator acting on
those alerts would investigate the wrong problem.

**H2: partly supported.** Combining the two methods did not improve detection
or harm-finding; its alert decisions were identical to drift-only. Its value
was diagnostic: it separated corruption from distribution change. We
therefore do not find that combined monitoring wins on detection metrics.

**H3: partly supported.** Shift type mattered: sudden shifts were caught at
once, the gradual shift slightly later, and the label-relationship change not
at all. Severity mattered less than expected for hours-per-week, where even
the smallest shift was always detected, which suggests that detection depends
on a feature's distribution shape as well as on the size of the shift.

**H4: supported.** Most drift alerts did not coincide with harm: 85% fell on
batches where the model performed normally. Feature monitoring could neither
confirm harm, since it alerted equally on D, which was harmful, and on C and
E, which were not, nor detect it, since it missed F. This is consistent with
Podkopaev and Ramdas [2022] and supports measuring performance directly
whenever labels become available.

**Practical implication.** In a pipeline like this one, drift alerts are best
treated as a prompt to check model performance rather than as evidence that
the model has degraded, and quality checks should run first so that corrupted
data is not mistaken for distribution change.

## 6. Limitations

- We use one dataset (the 1994 US census), one model and one family of drift
  tests, and results may not generalise beyond this setting [Ding et al. 2021].
- All changes are synthetic; real shifts are often more complex.
- Each scenario has five seeds. Scenario D samples from the whole monitor
  pool, so its batches can repeat rows from other batches in the same run.
- Harm is defined by a single ROC-AUC rule, and other rules could classify
  batches differently. ROC-AUC itself depends on the mix of people in a batch,
  so part of D's drop may reflect its changed population rather than a less
  accurate model.
- The paired design reuses base batches across scenarios; 13 of the 81 harmful
  batches fall outside D and F, mostly repeats of a few unusual base batches.
- The KS test is conservative for features with many tied values, such as
  capital-gain and capital-loss, which are mostly zero.
- The combined strategy uses no performance signal, since labels are rarely
  available immediately; strategies using delayed labels were not tested.
- The Evidently comparison used only three batches.

## 7. Conclusion and future work

In a controlled experiment on a tabular pipeline, statistical drift detection
reliably detected changes in input data but was a poor indicator of model
harm: most of its alerts were on harmless changes, and the most harmful
change produced no alerts. A data-quality gate added nothing to detection but
correctly separated corrupted data from genuine distribution change.

Future work should repeat the experiment on real temporal and geographic
shifts using Folktables or TableShift, test strategies that use delayed labels
to monitor performance directly, and compare sensitivity across drift
thresholds and multiple-testing corrections.

## Reproducibility

Code, configuration and results are available at
https://github.com/alishba25/driftguard. The experiment configuration
(experiments/configs/main.yaml) was fixed before the first run.
`python experiments/run_experiments.py` reruns all 640 batches, and
`python experiments/summarize.py` and `python experiments/make_figures.py`
rebuild every table and figure. A local rerun and an independent Docker rerun,
with a fresh data download and model training, reproduced all batch results
exactly. Exact library versions are recorded in requirements-lock.txt and
requirements-lock-docker.txt.

## References

Becker, B., & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/2/adult

Breck, E., Zinkevich, M., Polyzotis, N., Whang, S., & Roy, S. (2019). Data validation for machine learning. In *Proceedings of Machine Learning and Systems (SysML)*.

Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021). Retiring Adult: New datasets for fair machine learning. In *Advances in Neural Information Processing Systems (NeurIPS)*.

Evidently AI. (n.d.). Data drift explainer. Retrieved 3 November 2026, from https://docs.evidentlyai.com/metrics/explainer_drift

Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*.

Gardner, J., Popovic, Z., & Schmidt, L. (2023). Benchmarking distribution shift in tabular data with TableShift. In *Advances in Neural Information Processing Systems (NeurIPS)*.

Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2019). Learning under concept drift: A review. *IEEE Transactions on Knowledge and Data Engineering*.

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12.

Podkopaev, A., & Ramdas, A. (2022). Tracking the risk of a deployed model and detecting harmful distribution shifts. In *International Conference on Learning Representations (ICLR)*.

Rabanser, S., Günnemann, S., & Lipton, Z. C. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. In *Advances in Neural Information Processing Systems (NeurIPS)*.

Schelter, S., Lange, D., Schmidt, P., Celikel, M., Biessmann, F., & Grafberger, A. (2018). Automating large-scale data quality verification. *Proceedings of the VLDB Endowment*, 11(12).

Sculley, D., et al. (2015). Hidden technical debt in machine learning systems. In *Advances in Neural Information Processing Systems (NeurIPS)*.

Virtanen, P., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*, 17.