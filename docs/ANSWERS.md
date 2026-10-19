# Workbook Answers — Day 1 & Day 2

---

## DAY 1

### Section 2 — EDA

**Q1 — How many visits / patients?**
- Train: **44,590 visits**, **5,576 distinct patients** (4 to 12 visits each, median ~8)
- Test: **11,013 visits**, **1,395 patients** (4 to 12 visits, median ~7)

**Q2 — Most missing columns**

The three columns with the most missing values:

1. `off` — **18,913 missing** (~42% of visits have no OFF score)
2. `ledd` — **16,338 missing** (~37%)
3. `on` — **13,220 missing** (~30%)

There are visits where both `off` and `on` are missing — this is clinically explainable: some visits are purely demographic or follow-up visits with no motor test.

**Q3 — Distribution of `target`**

- Actual range: **0 to 109.5** (well below the theoretical MDS-UPDRS maximum of 132)
- Mean: **37.5**, standard deviation: **16.5**
- Slightly right-skewed (positive skew) — advanced Parkinson's disease generates extreme high scores

**Q4 — Association with `target`**

- `off`: r = **0.87** — strong, but clearly different from `target` (measurement bias, drug timing, missing values)
- `on`: r = **0.76** — weaker, which makes sense (ON measures the medicated state, less representative of true OFF)
- `off` is **not** a copy of `target`: the mean difference is +7.4 points with a standard deviation of 10 — this is the bias the model must correct

**Q5 — Is `ledd` missing at random?**

No. `ledd` is more often missing in certain cohorts. Clinically: cohort B may follow patients without treatment (pre-levodopa) or under a different protocol. A missing `ledd` is not noise — it is a signal about the disease stage.

**Q6 — Columns not to feed to the model**

- `Index` — row identifier, no predictive signal
- `patient_id` — patient identifier, **major leakage risk**: the model could memorise individuals from the training set and never generalise to test patients

---

### Section 3 — Dummy mean

**Q1 — Dummy RMSE vs std of `target`**

- Dummy RMSE: **16.50**
- Std(target): **16.50**

They are identical **by construction**: when a model always predicts the mean, its root mean squared error equals the standard deviation. This is the mathematical definition of "zero knowledge".

**Q2 — Prediction-error plot**

A **horizontal line** at y = 37.5 (the mean). For low true values (target = 5), the model massively over-predicts. For high values (target = 90), it massively under-predicts. A perfect model would be a **diagonal y = x**.

**Q3 — Reference RMSE: 16.50**

A gain smaller than ~0.5 RMSE (3%) is within fold-to-fold noise. A real gain starts at ~1 RMSE.

---

## DAY 2

### Section 1 — Ridge

**Q1 — Ridge gain over Dummy**

Ridge achieves **10.49 RMSE** (row holdout 0.2) → reduction of **(16.50 − 10.49) / 16.50 = 36%**. This is a significant gain — the numeric features carry a strong signal.

**Q2 — Dominant coefficient**

`age` has the largest positive coefficient, `age_at_diagnosis` the largest negative one. The `off` coefficient is **positive** (the higher the measured OFF score, the higher the true OFF) — clinically consistent. The `on` coefficient is slightly negative — a good ON state may indicate a strong drug response, hence a potentially more severe true OFF.

**Q3 — Why raw coefficients cannot be compared directly**

Features have very different scales: `age` ranges from 30 to 90 (range ~60), `ledd` from 0 to 3,000 (range ~3,000). A coefficient of 0.5 on `ledd` represents an effect 50× smaller than a coefficient of 0.5 on `age`. Features must be **standardised** (StandardScaler) to compare relative importances.

---

### Section 2 — Tweak Ridge

**Q1 — Sensitivity to alpha**

Very low. Alpha 0.1 gives **10.49**, alpha 1.0 gives **10.60**, alpha 10.0 gives **10.78**. With 44,590 rows and only 8 features, the system is highly over-determined — regularisation adds almost nothing; the model has no capacity to overfit.

**Q2 — Coefficient changes**

Yes, slightly. The `age` coefficient decreases as alpha increases (stronger shrinkage). But the order of magnitude stays stable — the signal is robust.

---

### Section 3 — GroupKFold

**Q1 — Random split vs grouped CV**

- Dummy random holdout: ~16.50 → grouped CV: **16.50** (identical — the dummy cannot leak)
- Ridge random holdout: **10.49** → grouped CV: **10.60** (slight degradation)

Ridge barely changes because it is simple. Complex models (LGBM, transformers) change a lot — LGBM drops from ~2.5 in-sample to **4.58** with grouped CV.

**Q2 — Which is closer to the Kaggle score?**

**Grouped CV** is much closer. Our Ridge grouped CV = 10.60, Kaggle public score ≈ 10.x. The random split gave 10.49 — slightly optimistic. For LGBM: grouped CV 4.58 → Kaggle public ~4.5. The random split would have given ~2–3 (heavily leaky).

**Conclusion: patient-grouped CV is the only honest evaluation for this competition.**

---

### Section 4 — HGBR

**Q1 — Does HGBR beat Ridge on every fold?**

Yes, on all 5 folds. Grouped CV RMSE: **7.76** (vs 10.60 for Ridge). The gain is consistent, not accidental.

**Q2 — Fit time**

HGBR is ~3–5× slower than Ridge, but a gain of 2.8 RMSE is more than worth it.

**Q3 — Feature importance: HGBR vs Ridge**

HGBR highlights `off`, `on`, `ledd`, and timing features (`time_since_intake_off`). Ridge highlighted `age` and `age_at_diagnosis`. HGBR captures non-linear interactions between `off` and drug timing that Ridge cannot see.

---

### Section 5 — tabular_pipeline

**Q1 — Gain from adding `gene` and `cohort`**

RMSE drops from 7.76 (numeric HGBR) to **7.52** with `gene`/`cohort` — a reduction of −0.24 RMSE. Real but modest. These features carry a signal about disease progression and drug response specific to each mutation.

**Q2 — Encoders chosen by TableVectorizer**

| Column | Cardinality | Encoder |
|---|---|---|
| `cohort` | 2 (A/B) | one-hot |
| `gene` | 4–5 values | one-hot |
| `rater_id` | ~50 values | one-hot or StringEncoder |

---

### Section 6 — DataOps

**Q1 — Same scores as step 5?**

Slightly different (±0.01 RMSE) depending on whether `patient_id` was correctly excluded. The difference comes from identifier column handling — DataOps automatically excludes what should not be a feature.

**Q2 — Do `Index` or `patient_id` appear in X_op?**

No — they are explicitly dropped. If `patient_id` appeared in the feature importance, it would be a sign of **catastrophic leakage**: the model would memorise training individuals and produce an artificially low OOF RMSE but a collapsed Kaggle score.

---

### Section 7 — Wrap-up

**Q1 — Which single change brought the biggest improvement?**

In order of impact:

1. **Generative architecture exp 33** (measurement calibration + quadratic patient curves + LightGBM ensemble): OOF RMSE **3.35** — a 5.6% gain over exp 32 (3.55), and ~26% over the first engineered LightGBM (4.58)
2. **Per-patient feature engineering** (aggregates `pm_off`, `pm_std_off`, etc.): −2.5 RMSE (from 7.52 to **4.58**)
3. **Grouped CV**: no RMSE gain in itself, but the honest signal — without it we would have believed 2.5 when the true level was 4.5
4. **Adding categoricals** (`gene`, `cohort`, `rater_id`): −0.25 RMSE
5. **HGBR vs Ridge**: −2.8 RMSE

What did not help: the autoregressive lagged target (exp 09, RMSE 8.0 — worse than a plain LGBM). Transformers (exp 23, RMSE 4.09) were competitive but never outperformed the LightGBM ensemble.

**Q2 — Which visits does the best model get most wrong?**

Visits with missing `off` **and** an advanced-stage patient. Also the first visits of patients with little history (`visit_rank = 1`) — the model has no patient context yet. And patients from rare cohorts with atypical genetic mutations.

**Q3 — Which modelling problem has not been fully addressed?**

**Individual temporal progression** was partially addressed by experiment 33: the regularised quadratic curve per patient (level + slope + curvature on an orthogonal time basis) captures the trajectory better than the static aggregates from earlier experiments. This explains part of the 5.6% gain over exp 32.

What remains open: the individual slope estimate is noisy with ~8 visits and many NaNs in `off`. The next improvement would be **Bayesian slope estimation** — partial pooling across similar patients (same gene, same cohort) to stabilise curves on few observations. The leaderboard leaders (~2.05 Kaggle) have likely adopted an approach of this kind.

---

*Best submission: `submissions/33_generative_progression.csv` — measurement calibration + LightGBM ensemble (35% visit model A + 35% visit model B + 30% patient-curve model) — OOF RMSE **3.35080** — [Skore Hub report](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/87766).*
