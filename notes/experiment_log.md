# Experiment log

## E1: False alerts on stable batches (date: 29-10-2026)

- Setup: 100 batches of 1,000 rows sampled from the monitor pool (seeds 0-99);
  reference = training split; KS for 5 numeric, chi-square for 7 categorical features
- Result: no correction 37 / 100 batches alerted; Bonferroni 3 / 100
- Interpretation: Without correction, more than a third of batches where nothing changed raised a drift alert. That's because each batch runs 12 tests at p < 0.05, so one test is likely to cross the threshold by chance. If the features were independent, about 46% of batches would alert (1 - 0.95^12). The observed 37% is lower, probably because correlated features (e.g. relationship and marital-status) tend to raise false alarms together rather
than separately. Bonferroni (p < 0.05/12 ≈ 0.0042 per feature) cut false alerts to 3%, within the 5% per-batch target it is designed to guarantee. With only 100 batches this estimate is rough (±2-3 points), but the gap between 37 and 3 is far larger than that noise. 
- Conclusion: per-feature testing without correction is unusable for monitoring, so Bonferroni is adopted as the default before Phase 4. The expected cost is lower sensitivity to small real shifts, which Phase 4 will measure.
- Limitation: batches are drawn from the same ~9,800-row pool, so they overlap and are not fully independent samples

## E3: Main experiment, four strategies x scenarios A-F (date: 29-10-2026)

- Setup: experiments/configs/main.yaml (committed 29-10-2026, before the first run);
  8 scenarios x 5 seeds x 16 batches of 600 rows = 640 batches; change starts
  at batch 8; harm = ROC-AUC more than 0.03 below baseline (baseline 0.9056,
  so harmful = ROC-AUC < 0.8756)
- Reproducibility: NOT YET VERIFIED (full rerun pending)
- Sanity check: A_stable harmful share = 0.025 after the change (1 of 40 batches).
  Before the change, 8 of 320 batches (2.5%) were "harmful" across all scenarios:
  exactly 1 per scenario, because the paired design reuses the same base batches,
  so this is one unlucky draw counted 8 times. ROC-AUC on unchanged batches:
  mean 0.904, sd 0.011, so the 0.03 rule sits about 2.6 sd below normal

### Results
- Overall, input-change truth: quality P=1.000 R=0.167; drift P=0.983 R=0.975;
  combined P=0.983 R=0.975; false alerts per 100 quiet batches: quality 0,
  drift 1.0, combined 1.0 (4 of 400)
- Overall, harm truth: precision_harmful quality 0.050, drift/combined 0.147
  (35 of 238 alerts); recall_harmful quality 0.025, drift/combined 0.432 (35 of 81)
- Alert rate after change (drift / combined): B 1.00, C_low 1.00, C_mid 1.00,
  C_high 1.00, D 1.00, E 0.85, F 0.05
- Detection delay (drift): C_mid 0, C_high 0, D 0, E 1.2 (missed runs: 0 in all).
  Even C_low (+2 hours) was detected immediately in every run. E was never
  detected at +1.25 years (batch 8); 4 of 5 runs alerted at +2.5 years
- Model impact: harmful scenarios = D (27 of 40 changed batches, mean ROC-AUC
  0.870) and F (39 of 40, mean ROC-AUC 0.842). All other scenarios stayed at
  baseline (ROC-AUC 0.902-0.907), including B, C and E, which drift alerted on
- ROC-AUC after change in F = 0.842; drift alerted on only 2 of 40 F batches,
  the same rate as its false alerts on stable data
- Blame on B: drift-only named "sex" in 40/40 batches (alone in 37, with other
  features in 3); combined named quality in 40/40, plus drift in valid rows in 3
- Runtime: monitoring took about 170 ms per 600-row batch

### Hypotheses
- H1: supported. The quality gate caught all 40 corrupted batches and named the
  cause. Drift-only also alerted on all 40, but misclassified every one as a
  change in the distribution of sex (the invalid "Unknown" values), so a
  person acting on its alerts would chase the wrong problem
- H2: partly supported. Combined and drift-only made identical alert decisions
  on every batch, so combined was no better at detection or at finding harm.
  Its only gain was explanation: it correctly separated corruption (quality)
  from distribution change. Combined does not win on detection metrics
- H3: partly supported. Shift type mattered: sudden shifts were caught at
  once, the gradual age shift a little later (1.2 batches), and the concept
  change (F) not at all. Severity mattered less than expected: all three
  hours-per-week shifts, even +2 hours, were detected in every batch. A likely
  reason is that hours-per-week is heavily concentrated at 40, so shifting
  everyone moves that spike and KS sees a large gap. Detection appears to depend
  on the shape of the feature's distribution, not only the size of the shift;
  this is untested
- H4: supported. Most alerts did not mean harm: only 35 of 238 drift alerts
  (14.7%) fell on harmful batches. C (all severities), E and B triggered
  alerts with no performance loss. D was both detected and harmful. F was
  harmful in 39 of 40 batches yet invisible to every monitor, because the
  inputs did not change. Feature monitoring alone could neither confirm harm
  (D vs C/E) nor detect it (F)

### Limitations
- One dataset, one model, one detector family; synthetic changes
- 5 seeds; D samples from the whole pool, so its batches can repeat rows
- Harm defined by one ROC-AUC rule; other rules could classify differently.
  D's ROC-AUC drop may partly reflect the changed population mix rather than
  a broken model, since ROC-AUC itself depends on who is in the batch
- About 13 of the 81 harmful batches are background noise (one unlucky base
  batch per seed, repeated across scenarios by the paired design), which
  slightly inflates harm counts outside D and F
- Combined uses no performance signal (labels are rarely available at once)
- Only one shift size per feature for E and D; the C result suggests severity
  should also be tested on a smoother feature (e.g. age) before generalising