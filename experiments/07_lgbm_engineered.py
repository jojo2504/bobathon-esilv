# %% [markdown]
# # Experiment 07 — Feature engineering + LightGBM
#
# Goal: beat RMSE 7.52 (05_tabular) by a significant margin.
#
# Key engineered features:
#   1. Disease duration: age - age_at_diagnosis (r=0.54 with target)
#   2. Per-patient aggregates of X-only features (mean off, mean on, ledd)
#      — captures patient baseline severity WITHOUT using target (no leakage)
#   3. Motor gap: off - on, mean_motor, best_motor
#   4. PK-inspired interactions: off * timing weight
#   5. LightGBM native categorical support for cohort, gene, rater_id

# %%
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from skore import evaluate

from parkinson.hub import get_project

# ── Load raw data ─────────────────────────────────────────────────────────────
print("Loading data...")
X_train_raw = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")
X_test_raw = pd.read_csv("data/X_test.csv")

visits = X_train_raw.merge(y_train, on="Index")
print(f"  train: {visits.shape}  test: {X_test_raw.shape}")


# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix. Preserves original row order (no sort_values).

    All features are derived from X columns only — no target used.
    """
    out = df.copy()

    # Encode string columns as pandas category for LightGBM
    for col in ["cohort", "gene", "rater_id"]:
        if col in out.columns:
            out[col] = out[col].astype("category")

    # ── 1. Disease duration ───────────────────────────────────────────────────
    out["years_since_dx"] = out["age"] - out["age_at_diagnosis"]

    # ── 2. Per-patient aggregates of X-only features ──────────────────────────
    # Group by patient, compute stats, join back on patient_id.
    # Row order preserved because we use merge/transform, no reindex.
    pat = (
        out.groupby("patient_id", sort=False)
        .agg(
            pat_mean_off=("off", "mean"),
            pat_mean_on=("on", "mean"),
            pat_std_off=("off", "std"),
            pat_std_on=("on", "std"),
            pat_mean_ledd=("ledd", "mean"),
            pat_median_off=("off", "median"),
            pat_median_on=("on", "median"),
            pat_n_visits=("age", "count"),
            pat_age_range=("age", lambda x: x.max() - x.min()),
        )
        .reset_index()
    )
    out = out.merge(pat, on="patient_id", how="left")

    # ── 3. Motor gap features ─────────────────────────────────────────────────
    out["off_minus_on"] = out["off"] - out["on"]
    out["mean_motor"] = out[["off", "on"]].mean(axis=1)
    out["best_motor"] = out[["off", "on"]].max(axis=1)

    # ── 4. Deviation from patient mean ────────────────────────────────────────
    out["off_vs_pat_mean"] = out["off"] - out["pat_mean_off"]
    out["on_vs_pat_mean"] = out["on"] - out["pat_mean_on"]

    # ── 5. PK-inspired timing interactions ────────────────────────────────────
    # Logistic weight: longer time_since_intake_off → score closer to true OFF
    toff = out["time_since_intake_off"].fillna(out["time_since_intake_off"].median())
    out["off_pk_weight"] = 1.0 / (1.0 + np.exp(-0.3 * (toff - 12.0)))
    out["off_debiased"] = out["off"] * out["off_pk_weight"]

    # ── 6. Drop identifiers ───────────────────────────────────────────────────
    drop_cols = ["Index", "patient_id"]
    if "target" in out.columns:
        drop_cols.append("target")
    out = out.drop(columns=drop_cols)

    return out


# ── Build feature matrices ────────────────────────────────────────────────────
print("\nEngineering features...")
X_eng = engineer_features(visits)
y = visits["target"].values  # numpy array — index-independent
groups = visits["patient_id"]
print(f"  Feature matrix: {X_eng.shape}")
print(f"  Columns: {list(X_eng.columns)}")

# Sanity check alignment
assert len(X_eng) == len(y), "Row count mismatch!"
print(f"  Alignment check: OK ({len(X_eng)} rows)")

# ── LightGBM ─────────────────────────────────────────────────────────────────
lgbm_model = lgb.LGBMRegressor(
    n_estimators=1500,
    num_leaves=127,
    learning_rate=0.03,
    min_child_samples=20,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    reg_alpha=0.05,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

# ── Grouped CV ────────────────────────────────────────────────────────────────
print("\n[Step 7] LightGBM + engineered features (GroupKFold-5)")
cv_splits = list(GroupKFold(n_splits=5).split(X_eng, y, groups=groups))
rep = evaluate(lgbm_model, X_eng, y, splitter=cv_splits)
rmse = float(rep.metrics.rmse().iloc[:, 0].mean())
print(f"  LightGBM RMSE (grouped CV) = {rmse:.4f}")
print(f"  05_tabular baseline        = 7.5238")
print(f"  delta                      = {7.5238 - rmse:+.4f}")
print(f"  vs dummy floor             = {(16.4992 - rmse) / 16.4992 * 100:.1f}% reduction")

# ── Push to Hub ───────────────────────────────────────────────────────────────
print("\n[Push] → Hub key '07_lgbm_engineered'")
project = get_project()
project.put("07_lgbm_engineered", rep)

# ── Submission CSV ────────────────────────────────────────────────────────────
print("\n[Submission] Fitting on ALL training data...")
X_test_eng = engineer_features(X_test_raw)

# Align columns
for col in X_eng.columns:
    if col not in X_test_eng.columns:
        X_test_eng[col] = np.nan
X_test_eng = X_test_eng[X_eng.columns]

# Ensure same category levels (LightGBM needs consistent categories)
for col in ["cohort", "gene", "rater_id"]:
    if col in X_eng.columns and col in X_test_eng.columns:
        X_test_eng[col] = X_test_eng[col].cat.set_categories(X_eng[col].cat.categories)

final_model = lgb.LGBMRegressor(
    n_estimators=1500,
    num_leaves=127,
    learning_rate=0.03,
    min_child_samples=20,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    reg_alpha=0.05,
    reg_lambda=0.5,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)
final_model.fit(X_eng, y)

submission = X_test_raw[["Index"]].copy()
submission["target"] = final_model.predict(X_test_eng)
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/07_lgbm_engineered.csv", index=False)
print(f"Wrote submissions/07_lgbm_engineered.csv  ({len(submission)} rows)")
print("\nDone! Upload submissions/07_lgbm_engineered.csv to Kaggle with the Hub URL above.")
