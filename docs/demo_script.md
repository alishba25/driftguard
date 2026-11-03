# DriftGuard: 2-minute demo script

For interviews, or a short screen recording.

## Before you start

- Docker Desktop open, then `docker compose up -d` (or your local setup running)
- Dashboard open at http://localhost:8501 on the Overview page
- A terminal open in the `driftguard` folder, font made large
- Second browser tab: the GitHub repo, scrolled to the architecture diagram
- Third tab: `research/figures/fig2_alerts_vs_harm.png`

---

## 1. The problem (0:00-0:20)

**Show:** the GitHub README title.

**Say:**
> "A machine-learning model keeps making predictions whatever data it gets,
> even when that data is broken or has changed. Monitoring tools raise alerts
> when data changes, but an alert doesn't tell you whether the model actually
> got worse. I built DriftGuard to measure how often those two things line up."

## 2. The system (0:20-0:50)

**Show:** the architecture diagram.

**Say:**
> "Each incoming batch goes through a pipeline. First, seven data-quality
> rules: missing values, wrong types, unknown categories, impossible ranges,
> duplicates. Bad rows are set aside with the reason. Then the valid rows are
> compared with the training data using statistical drift tests, one per
> feature. Everything is stored in PostgreSQL and shown on a Streamlit
> dashboard. It's covered by 29 tests that run on every push, and the whole
> thing runs in Docker with one command."

## 3. Live (0:50-1:20)

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

## 4. The finding (1:20-1:50)

**Show:** Figure 2 (alerts vs harm).

**Say:**
> "Then I ran a controlled experiment: 640 batches, eight scenarios, five
> seeds, scoring every alert against what really happened. Orange is how
> often drift was flagged; black is how often the model actually got worse.
> Drift detection caught almost every change, but only 15% of its alerts
> came on batches where the model was really harmed. And the change that hurt
> the model most, where the relationship between inputs and income changed,
> produced no alerts at all, because the inputs themselves never changed."

> "So a drift alert is a prompt to check performance, not proof of a problem.
> The results reproduced exactly in a clean Docker rerun."

## 5. Next steps (1:50-2:00)

**Say:**
> "Next, I'd test this on real shifts across US states and years using the
> Folktables data, and add monitoring that uses labels as they arrive."

---

## Questions you might be asked

**Why logistic regression, not a stronger model?**
The project studies monitoring, not accuracy. A simple, interpretable model
keeps the focus there, and its ROC-AUC of 0.906 is a solid baseline.

**Why KS and chi-square?**
They're standard, well understood, and Rabanser et al. (2019) found per-feature
KS tests with a Bonferroni correction perform comparably to more complex methods.

**Why the Bonferroni correction?**
Testing 12 features at p < 0.05 gave false alerts on 37 of 100 stable batches.
Dividing the threshold by 12 cut that to 3.

**What did the combined strategy add, if its alerts matched drift-only?**
Explanation, not detection. On corrupted data, drift-only blamed a shift in
one feature; combined correctly reported a data-quality problem.

**How do you know the results aren't tuned to look good?**
The thresholds and experiment configuration were committed before the first
run, and an independent Docker rerun reproduced every result exactly.

**What are the main limitations?**
One dataset from 1994, one model, synthetic shifts and five seeds. That's why
the next step is real shifts with Folktables.

**What would you do differently?**
Define harm in more than one way and compare, and test on a dataset with
naturally occurring shifts from the start.