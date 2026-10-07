# JOURNAL

## Status

- **Project / dataset:** Parkinson's Disease True-OFF Score — synthetic multi-cohort visits
- **Goal:** minimize RMSE for the true OFF MDS-UPDRS motor score, on patients held out from training
- **Last experiment:** 12_full_ensemble (full-power LGBM + XGB + HGBR stacked) — done
- **Last result:** RMSE **4.58** OOF grouped CV (stacked) — 72.2% below dummy floor

- **Workspace decisions** (immutable unless the user pivots):
  - tabular library: pandas - recorded: 2025-07-14
  - env manager: pip+venv - recorded: 2025-07-14
  - agent feature: installed - recorded: 2025-07-14
  - optional features: lightgbm==4.7.0, xgboost==3.4.1 - recorded: 2025-07-14
  - package name (`src/<pkg>/`): parkinson - recorded: 2025-07-14
  - skore mode: hub - recorded: 2025-07-14
  - skore hub workspace: bobathon - recorded: 2025-07-14
  - skore mlflow tracking uri: n/a - recorded: 2025-07-14
  - student prior: comfortable - recorded: 2025-07-14
  - CV splitter family: row holdout (splitter=0.2) for 01_dummy/02_ridge; GroupKFold(5) on patient_id from 03 onwards - recorded: 2025-07-14

## Data understanding (EDA)

- **Status:** done - 2025-07-14
- **Summary:** 44,590 training visits across 5,576 patients (4–12 visits each); 1,395 held-out test patients. Target (true OFF) mean ≈ 37.5, std ≈ 16.5. `off` (r=0.87) and `on` (r=0.67) are strongest predictors. Per-patient mean(off) has r=0.93 with per-patient mean(target) — the biggest signal. Patient-level aggregates of X-features explain ~79% of target variance.
- **Report:** [data/eda.md](../data/eda.md)

## History

| Stem | Intent (one line) | Status | Headline result | Design note |
|---|---|---|---|---|
| `01_dummy` | DummyRegressor(mean) — establish RMSE floor | done | RMSE 16.50 (row holdout 0.2) — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/45526) | n/a |
| `02_ridge` | Ridge + median imputer, row holdout, alpha scan | done | RMSE 10.49 (row holdout 0.2, alpha=0.1) — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/estimators/45577) | [02_ridge.py](../experiments/02_ridge.py) |
| `03_grouped_cv` | Honest eval: dummy vs Ridge on GroupKFold-5 | done | Ridge RMSE 10.60 / Dummy 16.50 (grouped CV) — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/45588) | [03_grouped_cv.py](../experiments/03_grouped_cv.py) |
| `04_hgbr` | HGBR numeric-only, NaNs kept as signal (grouped CV) | done | RMSE 7.76 grouped CV — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/45596) | [04_hgbr.py](../experiments/04_hgbr.py) |
| `05_tabular` | TableVectorizer + HGBR, adds cohort/gene/rater_id | done | RMSE 7.52 grouped CV — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/45617) | [05_tabular.py](../experiments/05_tabular.py) |
| `06_dataops` | DataOps graph: groups baked in, same model as 05 | done | RMSE 7.53 grouped CV — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/45670) | [06_dataops.py](../experiments/06_dataops.py) |
| `07_lgbm_engineered` | LightGBM + patient-level X-aggregates + disease duration + PK features | done | RMSE **5.17** grouped CV — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/47675) | [07_lgbm_engineered.py](../experiments/07_lgbm_engineered.py) |
| `08_lgbm_v2` | LightGBM extended features (min/max off/on, ledd_per_year, on PK) | done | RMSE **4.96** grouped CV — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/47838) | [08_best.py](../experiments/08_best.py) |
| `08_xgb` | XGBoost same features — best single model | done | RMSE **4.93** grouped CV — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/47856) | [08_best.py](../experiments/08_best.py) |
| `08_blend` | 40% LGBM + 60% XGBoost blend — **best overall** | done | RMSE **4.91** OOF — submit `submissions/08_blend.csv` | [08_best.py](../experiments/08_best.py) |
| `09_lagged_target` | Within-patient autoregressive lag of target (corrected) | done | RMSE **8.02** OOF (autoregressive) — Hub leaky 2.11 confirms signal only exists with real targets; test patients fully held out, autoregression accumulates error — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/48185) | [09_lagged_target.py](../experiments/09_lagged_target.py) |
| `10_target_encoding` | LOO target encoding of patient_id, cohort/gene fallback for test | done | RMSE **10.48** OOF — test patients unseen so fallback to group means gives no gain over existing pat_mean_off feature — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/48263) | [10_target_encoding.py](../experiments/10_target_encoding.py) |
| `11_stacking` | Ridge meta on OOF from LGBM + XGB + HGBR + Ridge (800 trees) | done | RMSE **4.83** OOF stacked — marginal gain; weak base models (800 trees) limited ceiling — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/48263) | [11_stacking.py](../experiments/11_stacking.py) |
| `12_full_ensemble` | Full-power LGBM (3000) + XGB (2000) + HGBR (500) + Ridge meta + extra features — **new best** | done | RMSE **4.58** OOF (stacked) — [Hub](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/48670) | [12_full_ensemble.py](../experiments/12_full_ensemble.py) |

## Backlog

| # | Item | Source |
|---|---|---|
| B1 | Pharmacokinetic feature: `exp(-k * time_since_intake_off)` to capture non-linear drug-timing signal | `my-pick:05_tabular` |
| B3 | Hyperparameter tuning with Optuna (GroupKFold-aware) on LGBM/XGB — could push below 4.0 | `my-pick:08_best` |
| B4 | Neural network with patient embedding layer — captures individual patient baselines; train embedding from X features | `my-pick:08_best` |
| B6 | Deeper feature interactions: LGBM with monotone constraints on off/on (score can only go up with higher off) | `my-pick:12_full_ensemble` |
| B7 | Pseudo-labeling: use 12_stacked test predictions as soft labels, add test rows to training, retrain | `my-pick:12_full_ensemble` |
| B8 | Calibrated confidence: combine OOF residuals with feature uncertainty to weight samples | `my-pick:12_full_ensemble` |
