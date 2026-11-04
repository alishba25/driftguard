# DriftGuard: 2-minute demo script

For interviews, or a short screen recording.

## Before you start

- Docker Desktop open, then `docker compose up -d` (or your local setup running)
- Dashboard open at http://localhost:8501 on the Overview page
- A terminal open in the `driftguard` folder, font made large
- Second browser tab: the GitHub repo, scrolled to the architecture diagram
- Third tab: `research/figures/adult_20seeds_alerts_vs_harm_logreg.png`
- Fourth tab: `research/figures/acs_states_alerts_vs_harm_logreg.png`

---

## 1. The problem (0:00-0:20)

**Show:** the GitHub README title.

**Say:**
> "A machine-learning model keeps making predictions whatever data it gets,
> even when that data is broken or has changed. Monitoring tools raise alerts
> when data changes, but an alert doesn't tell you whether the model actually
> got worse. I built DriftGuard to measure how often those two things line up."

## 2. The system (0:20-0:45)

**Show:** the architecture diagram.

**Say:**
> "Each incoming batch goes through a pipeline. First, seven data-quality
> rules: missing values, wrong types, unknown categories, impossible ranges,
> duplicates. Bad rows are set aside with the reason. Then the valid rows are
> compared with the training data using statistical drift tests, one per
> feature. Everything is stored in PostgreSQL and shown on a Streamlit
> dashboard. It's covered by 30 tests that run on every push, and the whole
> thing runs in Docker with one command."

## 3. Live (0:45-1:10)

**Do:** in the terminal, run a deliberately broken batch:

    docker compose exec app python -m driftguard.pipeline data/batches/batch_bad_demo.csv

(Local setup: `python -m driftguard.pipeline data/batches/batch_bad_demo.csv`)

**Say, while it runs:**
> "This batch has forty problems hidden in it: missing ages, an invalid sex
> value, an age of 250, the word 'forty' in a number column, and duplicates."

**Point to the output:** the warning lines and `40 rejected`.

> "It caught all forty and said exactly what was wrong with each."

**Show:** the dashboard, Alerts page.

> "And it raised a quality alert. Rerunning the same batch doesn't create
> duplicates, because each batch is identified by a fingerprint of its contents."

## 4. The finding (1:10-1:50)

**Show:** the Adult alerts-vs-harm figure (third tab).

**Say:**
> "Then I ran experiments: over 4,000 batches, scoring every alert against
> what really happened, for two different models. First, controlled changes
> on census data. Orange is how often drift was flagged; black is how often
> the model actually got worse. Drift detection caught almost every change,
> but only about one alert in eight came on a batch where the model was
> really harmed. And the change that hurt the model most, where the
> relationship between inputs and income changed, raised no more alerts than
> stable data, because the inputs themselves never changed."

**Show:** the ACS figure (fourth tab).

> "Then real shift: models trained on California, monitored on real people
> from four other states. Every state set off an alert on every batch, but
> the harm ranged from one batch in seven for Texas to seven in eight for
> Puerto Rico. The alert looks the same either way."

> "So a drift alert tells you the data changed, not whether the model got
> worse. It's a prompt to check performance, not proof of a problem."

## 5. Next steps (1:50-2:00)

**Say:**
> "Next, I'd test shift over time, comparing survey years, and add
> monitoring that uses labels as they arrive."

---

## Questions you might be asked

**Why logistic regression and gradient boosting?**
The project studies monitoring, not accuracy. Logistic regression is simple
and interpretable; gradient boosting is a strong, very different model. Using
both shows the findings aren't specific to one model, and it showed that the
same alerts can mean harm for one model and not the other.

**Why KS and chi-square?**
They're standard, well understood, and Rabanser et al. (2019) found per-feature
KS tests with a Bonferroni correction perform comparably to more complex methods.

**Why the Bonferroni correction?**
Testing 12 features at p < 0.05 gave false alerts on 37 of 100 stable batches.
Dividing the threshold by the number of features keeps false alerts to about
5 per 100, which is what the experiments measured.

**What did the combined strategy add, if its alerts matched drift-only?**
Explanation, not detection. On corrupted data, drift-only blamed a shift in
one feature; combined correctly reported a data-quality problem.

**How do you know the results aren't tuned to look good?**
Every experiment's configuration was committed before it ran, the results
come with 95% confidence intervals over 20 seeds, and I checked that the
conclusions hold at three different harm thresholds. An independent Docker
rerun reproduced the original experiment exactly.

**Why does 0.03 define "harm"?**
It's about 2.4 standard deviations below normal batch-to-batch variation, and
it was fixed in advance. I also reran the analysis at 0.02 and 0.05: the main
conclusions hold; only whether one moderate shift counts as harmful changes.

**What are the main limitations?**
Both datasets are US census income data, and the real shift is geographic
only, from one year. Harm is defined by a single metric, ROC-AUC.

**What would you do differently?**
Test shift over time as well as between places, and include a strategy that
uses labels as they arrive, since input monitoring alone can't establish harm.