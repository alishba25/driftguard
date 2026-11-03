# Literature review

*DriftGuard: data-quality checks and drift detection for monitoring a tabular classifier*

## Why I wrote this and how I read

I wanted to know what was already known before I said anything about my own results. My question is how well data-quality checks combined with drift detection can tell harmful changes from harmless ones in the input data of a machine-learning model. Reading around it, I found that most of the individual pieces already exist: per-feature drift tests, data validation, and the idea that drift is not the same as harm. So the honest way to frame DriftGuard is as a small controlled comparison that puts those pieces side by side, not as a new idea.

I searched Google Scholar, arXiv and the NeurIPS, ICLR, VLDB, MLSys and ACM proceedings with queries like "concept drift survey", "detecting dataset shift", "data validation for machine learning", "harmful distribution shift" and "false alarm detection delay". I also followed the Cited-by links from Rabanser et al. and Podkopaev and Ramdas.

Reading depth varies, and I want to be upfront about it. I worked through most of the text of Rabanser, Breck, Schelter, Podkopaev, Dong, Poenaru-Olaru and Lu. For Gama I used the accepted manuscript. For Sculley, Ding et al. and Gardner et al. I read the abstracts, introductions and the parts that matter for my project, not the full papers. Evidently is cited from its documentation. I also skimmed two more papers, a large comparison of supervised drift detectors (Barros and Santos, 2018) and an evaluation of open-source drift tools (Müller et al., 2024), but only at abstract level, so I did not give them entries.

---

## Sculley et al. (2015). Hidden Technical Debt in Machine Learning Systems. NeurIPS.

- Problem: ML systems are quick to build but costly to maintain, and the cost is hidden at the system level instead of in the code.
- Method: argument from experience using the technical-debt metaphor from software engineering, with a catalogue of ML-specific risk factors.
- Data: none (position paper).
- Metrics: none.
- Findings: the risks they list include data dependencies, changes in the external world and difficulty monitoring behaviour. They argue data dependencies can cost more than code dependencies and are harder to detect.
- Limitations: no experiments. It tells you where the risk is, not how well any particular check works. (I only read the abstract and introduction closely.)
- Relation to DriftGuard: this is my motivation. Data that changes underneath a model is a known and expensive problem. It does not tell me which monitoring checks to trust, which is the gap my experiment tries to look at.

## Gama, Žliobaitė, Bifet, Pechenizkiy & Bouchachia (2014). A survey on concept drift adaptation. ACM Computing Surveys.

- Problem: organise what is known about changing data streams and how learners cope with them.
- Method: survey and taxonomy. Drift can be sudden, gradual, incremental or recurring, and outliers are not drift. They also discuss how to evaluate change detectors.
- Data: survey, no new experiments.
- Metrics: for detectors they name the probability of detecting a true change, the probability of false alarms and the delay of detection, and note these pull against each other.
- Findings: faster detection tends to come with more false alarms. Most of the methods covered rely on labels arriving, because they watch the model's error.
- Limitations: mostly stream learning with supervised signals, and says little about data quality or batch pipelines.
- Relation to DriftGuard: I use its vocabulary for my sudden (D) and gradual (E) scenarios, and my metrics (alert precision and recall, false alerts per 100 batches, detection delay) are the standard ones from this line of work. My drift-only strategy has no labels, so I needed other sources too.

## Lu, Liu, Dong, Gu, Gama & Zhang (2019). Learning under concept drift: a review. IEEE TKDE.

- Problem: update the drift literature and give a clearer framework.
- Method: review of more than a hundred publications, organised as detection, understanding and adaptation. Drift is defined as a change in the joint distribution P(X, y), which splits into a change in P(X) alone (virtual drift) and a change in P(y|X) (actual drift).
- Data: survey. It lists benchmark datasets, both synthetic and real.
- Metrics: survey.
- Findings: only a change in P(y|X) should move the decision boundary, so a change in P(X) alone may leave accuracy untouched.
- Limitations: mostly about streams, and little on data quality or on batch-level alerting.
- Relation to DriftGuard: it gives my scenario F its name and logic. Flipping labels changes P(y|X) and leaves the inputs alone, so input monitors have nothing to see. My results fit this, but later papers (Podkopaev, Dong) show the virtual-drift story is too tidy, because where the shifted data lands matters.

## Rabanser, Günnemann & Lipton (2019). Failing loudly: an empirical study of methods for detecting dataset shift. NeurIPS.

- Problem: how to detect dataset shift from samples, without labels, and how to tell whether a shift is a problem.
- Method: two-sample tests between source and target data. They combine dimensionality reductions (none, PCA, random projections, autoencoders, classifier outputs, a domain classifier) with a multivariate test (MMD) or per-dimension Kolmogorov-Smirnov tests combined with a Bonferroni correction.
- Data: image datasets (MNIST, CIFAR-10) with simulated covariate and label shifts of different strengths.
- Metrics: detection accuracy against number of samples, shift type and shift size.
- Findings: per-feature KS with Bonferroni did about as well as the multivariate test. Small shifts needed many more samples to catch. They also point out that many shifts are benign, so "the distributions differ" is not the same as "the model is in trouble".
- Limitations: images, not mixed numeric and categorical tables. One-shot tests, not repeated checks over a stream of batches. No data-quality checks.
- Relation to DriftGuard: my drift module is KS for the numeric columns and chi-square for the categorical ones, with a Bonferroni threshold (0.05 / 12 features, about 0.0042). That is an established choice, not an invention of mine, and I cite it in my methods. I extend the idea to tabular batches, add a quality gate, and score alerts against harm.

## Breck, Zinkevich, Polyzotis, Whang & Roy (2019). Data validation for machine learning. SysML / MLSys.

- Problem: catch data errors before they quietly damage training or serving, in Google's TFX pipelines.
- Method: a schema that records each feature's type, presence, value domain and so on, checked for every batch, plus separate comparisons between batches for skew and distribution change. They use a distance with an interpretable threshold instead of relying on p-values alone.
- Data: Google production pipelines. They also run a simulation where 10,000 of 100 million points (0.01%) were replaced by draws from a slightly wider normal, and a chi-square test at the 1% level flagged it in 7 of 10 trials.
- Metrics: experience reports and the simulation above. No controlled benchmark.
- Findings: a code bug that set a feature to -1 for part of the traffic produced valid-looking values and a slice-level accuracy drop, with no obvious failure. Aggregate model metrics are a weak safety net for data errors. P-value tests flag trivial changes when samples are large.
- Limitations: an industrial report, so no head-to-head comparison of strategies.
- Relation to DriftGuard: it motivates my quality gate (H1) as standard practice, and the large-sample warning matters because my reference set is about 29,000 rows. It also made me report counts next to percentages and treat batch size as something that changes the result.

## Schelter, Lange, Schmidt, Celikel, Biessmann & Grafberger (2018). Automating large-scale data quality verification. PVLDB.

- Problem: let people check data quality at scale without writing a custom validator each time (Amazon's Deequ).
- Method: declarative constraints (completeness, uniqueness, value ranges, types, custom predicates) compiled into aggregation queries on Spark, computed incrementally, with anomaly detection over the history of the quality metrics.
- Data: several datasets for runtime and constraint-suggestion experiments.
- Metrics: runtime, scalability and how good the suggested constraints are.
- Findings: "unit tests for data" is practical and scales.
- Limitations: no evaluation on ML drift or on whether a data problem hurt a model.
- Relation to DriftGuard: its constraint types line up closely with my seven quality rules and with the corruptions in scenario B (missing values, bad categories, wrong types, out-of-range values, duplicates). It is the industrial precedent for rule-based validation, but it never compares those checks with drift detectors.

## Podkopaev & Ramdas (2022). Tracking the risk of a deployed model and detecting harmful distribution shifts. ICLR.

- Problem: a warning system should fire on shifts that hurt the model and stay quiet on harmless ones, and it should be safe to check after every batch.
- Method: a sequential test of whether the model's risk on the new data has risen above its source risk plus a tolerance chosen in advance, using confidence sequences so that checking repeatedly does not inflate false alarms.
- Data: simulated covariate and label shifts, plus real image datasets including corrupted variants.
- Metrics: false-alarm control, detection power and time.
- Findings: simple examples where a model keeps working after a shift, so a raw drift alarm would be a false alarm for practical purposes. Their method needs at least some true labels, possibly delayed.
- Limitations: needs labels. It is a method for performance monitoring, not a comparison of quality and drift checks.
- Relation to DriftGuard: this is the closest work to H4. My finding that drift alerts and harm come apart supports their argument with a concrete pipeline experiment, so I should not claim H4 as new. It also shapes how I define harm: by an observed ROC-AUC drop on labelled data, never by the drift test itself.

## Dong, Wang, Sahri, Palpanas & Srivastava (2024). Efficiently mitigating the impact of data drift on machine learning pipelines. PVLDB.

- Problem: tell harmful covariate drift from benign drift, so a pipeline retrains only when it needs to.
- Method: fit a decision tree on whether the black-box model was right or wrong, treat leaves with below-average accuracy as low-accuracy regions, and flag drift as harmful when the share of serving data in those regions grows.
- Data: UCI Adult (48.8k rows, 14 attributes, split into groups by race and sex), a house-sales dataset, the CER smart-meter series and Mechanical MNIST, over 100 serving batches.
- Metrics: accuracy against retraining cost. They also compare how well harmful and benign drift are told apart.
- Findings: on Adult with logistic regression, overall accuracy was 0.878. One serving batch with fewer low-accuracy regions scored 0.886, while another with more of them fell to 0.793. KS and MMD-based detectors were only about 0.5 accurate at separating benign from harmful drift on Mechanical MNIST.
- Limitations: needs expert-labelled samples from the serving data (a uniform 20% of a batch in the full variant). It reports retraining cost, not alert precision, false alerts per batch or detection delay.
- Relation to DriftGuard: the closest prior work in setup, since it uses the same dataset and model family. The difference is the question. They ask how to retrain efficiently, and I ask how quality-only, drift-only and combined monitoring compare when each alert is scored against two separate truths. I need to position DriftGuard carefully against this paper.

## Poenaru-Olaru, Cruz, van Deursen & Rellermeyer (2022). Are concept drift detectors reliable alarming systems? A comparative study. arXiv.

- Problem: do drift detectors work as alarm systems on batched data, not just as research algorithms?
- Method: error-rate-based detectors (DDM, EDDM, ADWIN, HDDM-A, HDDM-W) with five classifiers, against distribution-based detectors (KS and Mann-Whitney based, and kdq-tree variants with several distance measures), run on a reference set followed by equal test batches.
- Data: three synthetic streams (abrupt and gradual drift, with noise and class imbalance) and two real datasets with known drift points, repeated over 10 seeds.
- Metrics: latency, false-positive rate per batch and miss-detection probability.
- Findings: ADWIN was strong on abrupt drift. Error-based detectors raised alarms on almost every batch once gradual drift got wide. Distribution-based detectors had higher false-positive rates, struggled with one-hot-encoded categorical data and with imbalance, and no distance measure won everywhere. Their conclusion is that the detectors are not fully reliable as alarm systems.
- Limitations: few real datasets with known drift, and it treats only the first drift point.
- Relation to DriftGuard: the closest evaluation protocol to mine (reference set, then batches, scored by false alerts, misses and delay). Adult is mixed-type and imbalanced, so I expected noisy distribution alerts, and that is exactly what my false-alert check (E1) was built to measure. They do not include a data-quality gate or test whether alerts coincide with harm.

## Ding, Hardt, Miller & Schmidt (2021). Retiring Adult: new datasets for fair machine learning. NeurIPS.

- Problem: tabular research leans heavily on UCI Adult, which has problems.
- Method: they trace Adult back to its source (the 1994 Current Population Survey), rebuild a superset from Census records, and release a set of new prediction tasks as Folktables.
- Data: Adult (48,842 rows, 14 attributes) and new tasks on income, employment, health, transportation and housing, spanning multiple years and all US states.
- Metrics: fairness and accuracy criteria across the tasks.
- Findings: the $50K income threshold in Adult is an extreme one that limits external validity, and empirical conclusions can be sensitive to it. The new datasets let researchers study temporal shift and geographic variation with real data.
- Limitations: it is about fairness research, not monitoring. (I read the abstract, introduction and the Adult section.)
- Relation to DriftGuard: it names my biggest limitation, which is one 1994 dataset with synthetic shifts. It also gives me the best future work: rerun the same experiment on Folktables, where the shift is real across years and states.

## Gardner, Popovic & Schmidt (2023). Benchmarking distribution shift in tabular data with TableShift. NeurIPS Datasets and Benchmarks.

- Problem: tabular models are used everywhere, but there was no good benchmark for how they handle distribution shift.
- Method: a benchmark of 15 binary classification tasks, each with a real shift, across domains such as finance, education, public policy and healthcare, plus a large study of tabular models and robustness methods.
- Data: 15 tabular tasks (some from the American Community Survey via folktables).
- Metrics: in-distribution against out-of-distribution accuracy, and the gap between them.
- Findings: a linear trend between in-distribution and out-of-distribution accuracy, domain-robustness methods that reduce shift gaps at a cost to in-distribution accuracy, and a strong relationship between the shift gap and shifts in the label distribution.
- Limitations: it measures how models cope with shift, not how monitoring alerts behave. (I read the abstract and related-work sections.)
- Relation to DriftGuard: future work. It offers real tabular shifts where I can test whether my findings hold beyond synthetic ones. Its finding that label-distribution shift tracks the performance gap is also a useful hint for my scenario F.

## Evidently AI. Data drift: how drift detection works (documentation).

- Problem: not a paper. It describes what a widely used open-source monitoring library does by default.
- Method: compares a reference and a current dataset column by column. With 1,000 reference rows or fewer, it uses KS for numeric columns and chi-square for categorical ones. With more than 1,000, it switches to Wasserstein distance (numeric) and Jensen-Shannon divergence (categorical), with a threshold of 0.1. It states that drift tests ignore empty values by default and suggests separate checks on the share of nulls.
- Data and metrics: not applicable.
- Findings: I rely on the documentation for the defaults above.
- Limitations: documentation is not independent evaluation, and defaults can change between versions.
- Relation to DriftGuard: it confirms what I saw in E2. My reference set is far over 1,000 rows, so Evidently and my SciPy tests are answering different questions (how big is the change, against how sure am I that a change exists). It also supports H1 indirectly, since a drift test that ignores nulls can miss exactly what a quality gate is built to catch.

---

## What the papers say about my hypotheses

- **H1 (quality gate catches what drift misses).** Backed in principle by Breck et al. (the -1 bug), by Schelter et al.'s constraint types and by Evidently's handling of nulls. I found no paper that measures it directly, so my corruption scenario is a real test.
- **H2 (combined is more informative than either alone).** Indirect support only. The checks cover different failures, but Poenaru-Olaru et al. and Breck et al. both show that more detectors bring more false alarms, so I expected a trade-off and not a clean win.
- **H3 (detection depends on shift type, severity, detector and threshold).** Well supported by Rabanser et al. (small shifts are hard), Poenaru-Olaru et al. (abrupt against gradual, no best distance) and Gama et al. (detection speed against false alarms).
- **H4 (drift is not always harm).** Strongly supported by Podkopaev and Ramdas, Dong et al. and Rabanser et al., and by the virtual-drift idea in Lu et al. This is the finding I should not oversell as new.

## What I can honestly claim

The individual ideas in DriftGuard are established: per-feature drift tests with a multiple-testing correction, rule-based data validation, and the gap between drift and harm. What I did not find in the papers I read is a small, fully reproducible, controlled comparison of no monitoring, quality-only, drift-only and combined monitoring on one tabular pipeline, where every alert is scored against two separate truths: whether the inputs changed, and whether the model's ROC-AUC actually fell. That is what DriftGuard contributes, and it is modest. In my experiment drift detection caught 97.5% of input changes with about one false alert per 100 unchanged batches, but only 35 of its 238 alerts landed on batches where the model was actually harmed. A label-relationship change harmed 39 of 40 batches without raising drift alerts above the false-alert rate. And on the corrupted batches, drift-only blamed a distribution change in the sex feature every time, while the combined strategy named the real cause. I am not claiming this is new. My search was small and focused, Dong et al. already work with the same dataset and model family, and there is tabular shift-benchmark work I only skimmed. I would want a wider search before saying anything stronger about novelty.

## Limits of this review

I read ten papers closely, picked for closeness to my design, not for completeness. Three of them I read only in part, and I skimmed two more at abstract level without using them. I did not cover newer label-free performance-estimation or sequential conformal methods in depth, and any of them could weaken my gap statement. The search was a single pass by one person.

## References

1. Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). Hidden technical debt in machine learning systems. *Advances in Neural Information Processing Systems 28*. https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems
2. Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys, 46*(4), Article 44. https://doi.org/10.1145/2523813
3. Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2019). Learning under concept drift: A review. *IEEE Transactions on Knowledge and Data Engineering, 31*(12), 2346-2363. https://doi.org/10.1109/TKDE.2018.2876857
4. Rabanser, S., Günnemann, S., & Lipton, Z. C. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. *Advances in Neural Information Processing Systems 32*. https://arxiv.org/abs/1810.11953
5. Breck, E., Zinkevich, M., Polyzotis, N., Whang, S. E., & Roy, S. (2019). Data validation for machine learning. *Proceedings of Machine Learning and Systems (SysML) 2019*. https://mlsys.org/Conferences/2019/doc/2019/167.pdf
6. Schelter, S., Lange, D., Schmidt, P., Celikel, M., Biessmann, F., & Grafberger, A. (2018). Automating large-scale data quality verification. *Proceedings of the VLDB Endowment, 11*(12), 1781-1794. https://doi.org/10.14778/3229863.3229867
7. Podkopaev, A., & Ramdas, A. (2022). Tracking the risk of a deployed model and detecting harmful distribution shifts. *ICLR 2022*. https://arxiv.org/abs/2110.06177
8. Dong, S., Wang, Q., Sahri, S., Palpanas, T., & Srivastava, D. (2024). Efficiently mitigating the impact of data drift on machine learning pipelines. *Proceedings of the VLDB Endowment, 17*(11), 3072-3081. https://doi.org/10.14778/3681954.3681984
9. Poenaru-Olaru, L., Cruz, L., van Deursen, A., & Rellermeyer, J. S. (2022). Are concept drift detectors reliable alarming systems? A comparative study. arXiv:2211.13098. https://arxiv.org/abs/2211.13098
10. Ding, F., Hardt, M., Miller, J., & Schmidt, L. (2021). Retiring Adult: New datasets for fair machine learning. *Advances in Neural Information Processing Systems 34*. https://arxiv.org/abs/2108.04884
11. Gardner, J., Popovic, Z., & Schmidt, L. (2023). Benchmarking distribution shift in tabular data with TableShift. *Advances in Neural Information Processing Systems 36 (Datasets and Benchmarks)*. https://arxiv.org/abs/2312.07577
12. Evidently AI. Data drift: how data drift detection works (documentation). https://docs.evidentlyai.com/metrics/explainer_drift