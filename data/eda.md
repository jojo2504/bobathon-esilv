# EDA: Parkinson's True-OFF Score

> Interactive per-table reports:
> - [data/eda_train.html](eda_train.html) — training visits
> - [data/eda_test.html](eda_test.html) — test visits

---

## Dataset at a glance

| Split | Rows | Columns | Unique patients |
|---|---|---|---|
| Train (X_train + y_train merged) | 44,590 visits | 14 | 5,576 patients |
| Test (X_test) | 11,013 visits | 13 | 1,395 patients |

The test set represents ~20% of total patients and **no patient appears in both splits** — the holdout is by `patient_id`, not by visit row. This is the key structural fact for cross-validation.

Each patient contributes 4–12 visits (median 7, mean 8). A random row holdout (`splitter=0.2`) would leak the same patient across train and validation folds, overstating CV performance for any model that exploits individual patient trends. GroupKFold on `patient_id` is the correct splitter for Day 2+.

---

## Per-column findings

| Column | Type | Missing % | Unique | Notes |
|---|---|---|---|---|
| `Index` | Int64 | 0% | 44,590 | Row identifier — exclude from features |
| `patient_id` | String | 0% | 5,576 | Group identifier — exclude from features |
| `cohort` | String | 0% | 2 | Two cohorts — low cardinality, use as categorical |
| `sexM` | Int64 | 0% | 2 | Binary (0/1) |
| `gene` | String | **32.4%** | 4 | Categorical, heavily missing |
| `age_at_diagnosis` | Float | **5.2%** | 563 | Slightly missing |
| `age` | Float | 0% | 1,150 | Always present |
| `ledd` | Float | **36.6%** | 1,320 | Levodopa equivalent daily dose — often missing |
| `time_since_intake_on` | Float | **46.4%** | 64 | Hours since last ON measurement |
| `time_since_intake_off` | Float | **78.8%** | 178 | Hours since last OFF measurement — most missing column |
| `rater_id` | String | 0% | 50 | Clinician who scored — 50 raters, use as categorical |
| `on` | Float | **29.6%** | 83 | Measured ON motor score — significant missingness |
| `off` | Float | **42.4%** | 101 | Measured OFF motor score — significant missingness |
| `target` | Float | 0% | 867 | TRUE OFF score — no missing, always present in train |

---

## Missingness deep-dive

```
time_since_intake_off   78.8%
time_since_intake_on    46.4%
off                     42.4%
ledd                    36.6%
gene                    32.4%
on                      29.6%
age_at_diagnosis         5.2%
```

- **18,913 visits** (42%) have no `off` score — the "uncomfortable exam" bias described in CONTEXT.md.
- **13,220 visits** (30%) have no `on` score.
- **Zero visits** have both `on` and `off` missing — every visit has at least one motor measurement.
- `time_since_intake_off` being 79% missing is consistent with the rarity of OFF exams. When it is present, it is clinically meaningful for debiasing.
- Missingness in `ledd` clusters by cohort (one cohort tracks dosage more carefully than the other) — confirmed by the two-cohort structure.

---

## Target distribution

| Statistic | Value |
|---|---|
| Mean | 37.5 |
| Std | 16.5 |
| Min | 0.0 |
| Max | 109.5 |
| Q25 | 25.6 |
| Median | 37.3 |
| Q75 | 49.3 |

The target covers 0–109.5 (theoretical max 132). Distribution is roughly bell-shaped, slightly right-skewed, centred near 37. The standard deviation (16.5) is the baseline RMSE — a dummy "always predict the mean" model will score approximately RMSE ≈ 16.5. Any real model must beat this.

---

## Associations with the target

| Feature | Pearson r with target | Notes |
|---|---|---|
| `off` | **0.871** | Very strong — `off` is a biased version of `target`. High correlation is signal, not leakage (they measure the same underlying quantity). |
| `on` | **0.669** | Strong — ON and OFF are correlated because both track disease severity |
| `ledd` | 0.298 | Moderate — higher dose linked to higher true OFF (more severe disease) |
| `age` | 0.310 | Moderate — older patients tend to have more severe disease |
| `time_since_intake_on` | ~0 | Near-zero linear correlation; non-linear effect possible |
| `time_since_intake_off` | ~0 | Same |
| `gene`, `cohort` | low Cramér V | Categorical, modest association |

Key finding: `off` (the clinic OFF score) has Pearson r = 0.871 with `target` — they move together but are not identical. That gap (bias) is exactly what the competition asks us to close. `on` is also a strong predictor. `time_since_intake_*` has near-zero *linear* correlation but non-zero Cramér V — non-linear timing effects exist.

The strongest association in the entire dataset is `age_at_diagnosis` ↔ `age` (r = 0.94) — these two columns carry almost the same information. Using both is not harmful but redundant.

---

## Structure signals

- **No datetime columns** detected — there is no explicit visit date. Temporal progression is implicit in visit order per patient, not a parseable timestamp. TimeSeriesSplit is not applicable; GroupKFold is.
- **`Index`** has unique ratio 1.0 — row identifier, must be excluded from features.
- **`patient_id`** has 5,576 unique values — group identifier for GroupKFold. Must not be fed as a feature (would be a data leak and useless on new patients).
- **`rater_id`** has 50 unique values — encodes clinician-specific scoring bias. Worth including as a categorical feature in later models.

---

## Modelling implications

1. **Splitter**: `splitter=0.2` (random row holdout) for the first two experiments to establish a floor quickly. The leak it introduces (same patient on both sides) will inflate CV scores — that is intentional for Day 1. Switch to `GroupKFold(n_splits=5)` on `patient_id` from Day 2 onward for honest evaluation.

2. **Features**: `off` and `on` are the two strongest predictors. A Ridge with `SimpleImputer` on the 8 numeric columns is the natural first non-dummy model (Day 1 step 2). Later models should add `cohort`, `gene`, `rater_id`, and intake timing.

3. **Missing data**: Any model must handle missingness. For the dummy and Ridge: `SimpleImputer(strategy="median")`. For tree-based models later: native NaN support or `HistGradientBoostingRegressor`.

4. **Leakage guards**: Exclude `Index`, `patient_id`, and `target` from features. `rater_id` is not leakage — it is known at test time.

5. **RMSE floor**: std(target) ≈ 16.5. Dummy RMSE ≈ 16.5. Any real model must beat this clearly to be meaningful.

---

## Open questions

- Does `ledd` missingness actually cluster by cohort, or is there another confound? Worth checking per-cohort missingness in the HTML report.
- `time_since_intake_*` has near-zero linear correlation but the biology predicts a pharmacokinetic curve (non-linear decay). A feature like `exp(-k * time_since_intake_off)` might recover that signal.
- Are there patients with only one visit in test? If so, per-patient aggregation features from training cannot be computed for them.
