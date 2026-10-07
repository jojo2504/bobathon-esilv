# JOURNAL

## Status

- **Project / dataset:** Parkinson's Disease True-OFF Score — synthetic multi-cohort visits
- **Goal:** minimize RMSE for the true OFF MDS-UPDRS motor score, on patients held out from training
- **Last experiment:** 06_dataops — done
- **Last result:** RMSE 7.53 (grouped CV) — DataOps graph, same score as 05_tabular

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2025-07-14
  - env manager: pip+venv - recorded: 2025-07-14
  - agent feature: installed - recorded: 2025-07-14
  - optional features: none - recorded: 2025-07-14
  - package name (`src/<pkg>/`): parkinson - recorded: 2025-07-14
  - skore mode: hub - recorded: 2025-07-14
  - skore hub workspace: bobathon - recorded: 2025-07-14
  - skore mlflow tracking uri: n/a - recorded: 2025-07-14
  - student prior: comfortable - recorded: 2025-07-14
  - CV splitter family: row holdout (splitter=0.2) for 01_dummy/02_ridge; GroupKFold(5) on patient_id from 03 onwards - recorded: 2025-07-14

## Data understanding (EDA)

- **Status:** done - 2025-07-14
- **Summary:** 44,590 training visits across 5,576 patients (4–12 visits each); 1,395 held-out test patients. Target (true OFF) mean ≈ 37.5, std ≈ 16.5 → dummy RMSE floor ≈ 16.5. `off` (r=0.87) and `on` (r=0.67) are the strongest predictors. Heavy missingness in `time_since_intake_off` (79%), `off` (42%), `ledd` (37%). No datetime column → GroupKFold on `patient_id` is the correct splitter.
- **Report:** [data/eda.md](../data/eda.md)

## History

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_dummy` | DummyRegressor(mean) — establish RMSE floor | done | RMSE 16.50 (row holdout 0.2) — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/45526) | n/a |
| `02_ridge` | Ridge + median imputer, row holdout, alpha scan | done | RMSE 10.49 (row holdout 0.2, alpha=0.1) — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/estimators/45577) | [02_ridge.py](../experiments/02_ridge.py) |
| `03_grouped_cv` | Honest eval: dummy vs Ridge on GroupKFold-5 | done | Ridge RMSE 10.60 / Dummy 16.50 (grouped CV) — [Ridge Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/45588) | [03_grouped_cv.py](../experiments/03_grouped_cv.py) |
| `04_hgbr` | HGBR numeric-only, NaNs kept as signal (grouped CV) | done | RMSE 7.76 grouped CV — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/45596) | [04_hgbr.py](../experiments/04_hgbr.py) |
| `05_tabular` | TableVectorizer + HGBR, adds cohort/gene/rater_id | done | RMSE 7.52 grouped CV (**best**) — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/45617) | [05_tabular.py](../experiments/05_tabular.py) |
| `06_dataops` | DataOps graph: groups baked in, same model as 05 | done | RMSE 7.53 grouped CV — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/45670) | [06_dataops.py](../experiments/06_dataops.py) |

## Backlog

| # | Item | Source |
|---|---|---|
| B1 | Pharmacokinetic feature: `exp(-k * time_since_intake_off)` to capture non-linear drug-timing signal (near-zero linear correlation but non-zero Cramér V) | `my-pick:05_tabular` |
| B2 | Per-patient visit-order feature: use visit count / rank within patient to capture temporal disease progression | `my-pick:05_tabular` |
| B3 | Tune HGBR hyperparams (max_iter, max_leaf_nodes, learning_rate) with GroupKFold — RMSE plateau at 7.52 suggests underfitting capacity | `my-pick:06_dataops` |
| B4 | Ablation: median-impute NaNs for HGBR (compare vs step 4 where NaNs are kept) — quantifies missingness-as-signal contribution | `my-pick:04_hgbr` |
