# %% [markdown]
# # Experiment 27 — Two-level model: patient baseline + visit residual
#
# Diagnostic key finding from exps 23-26:
#   - r(age, target) within-patient = 0.97 → disease is nearly linear in time
#   - In-sample per-patient linear fit RMSE = 1.61 (the theoretical floor)
#   - All transformers stagnate ~3.9 because they can't go below the estimation
#     error on per-patient parameters with only 5576 unique sequences
#
# Architecture (pure sklearn, no GPU, ~3 min):
#
#   Level 1 — patient baseline (Ridge, fit on train patients, applied to val/test):
#     features → pat_mean_off, pm_on, pm_ledd, pm_std_off/on, pm_min/max_off/on,
#                pm_age, age_at_dx, n_visits, sexM, gene, cohort
#     target  → pat_mean_target  (the patient's average true-OFF score)
#     output  → one scalar per patient → broadcasted to all their visits
#
#   Level 2 — visit residual (LGBM, same GroupKFold):
#     input   → full visit-level features + Level-1 residual as a feature
#     target  → target - Level1_pred  (the within-patient deviation)
#     output  → per-visit correction on top of Level 1
#
#   Final prediction = Level1_pred + Level2_correction
#
# Why this beats the transformer:
#   - Level 1 explicitly models the inter-patient variation (79% of total variance)
#     using ALL patient aggregate features — Ridge is perfectly suited here
#   - Level 2 models the within-patient deviation (21% of variance) using
#     visit-level features like off_z_pat, visit_rank, rater_id
#   - No GPU needed, no 2-hour waits

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

print("Loading data...")
X_train_raw = pd.read_csv("data/X_train.csv")
y_train_df  = pd.read_csv("data/y_train.csv")
X_test_raw  = pd.read_csv("data/X_test.csv")
visits = X_train_raw.merge(y_train_df, on="Index")
y      = visits["target"].values
groups = visits["patient_id"]
cv     = GroupKFold(n_splits=5)
cv_splits = list(cv.split(visits, y, groups=groups))
N = len(y)


# ── Feature engineering helpers ───────────────────────────────────────────────
PAT_FEAT_COLS = [
    "pm_off", "pm_on", "pm_ledd",
    "pm_std_off", "pm_std_on",
    "pm_min_off", "pm_max_off",
    "pm_min_on",  "pm_max_on",
    "pm_age", "age_dx", "n_visits", "sexM",
    # gene one-hot
    "gene_GBAp", "gene_LRRK2p", "gene_OTHER", "gene_NoMut",
    # cohort binary
    "cohort_B",
    # per-patient off/on slopes (from X only — valid at test time)
    "pat_off_slope", "pat_on_slope",
]

VISIT_FEAT_COLS_BASE = [
    "age", "age_at_diagnosis", "years_dx",
    "ledd", "time_since_intake_on", "time_since_intake_off",
    "on", "off", "sexM",
    # within-patient deviations
    "off_vs_mean", "on_vs_mean", "off_z", "on_z",
    "off_vs_median",
    # PK timing
    "off_pk", "off_debiased",
    "on_pk",  "on_adjusted",
    # ratios
    "off_on_ratio", "off_minus_on",
    # visit order
    "visit_rank", "visit_rank_pct",
    # dose
    "ledd_per_year",
    # Level-1 prediction (as a feature for Level 2)
    "l1_pred",
]


def compute_pat_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """Per-patient slope of off (and on) vs age — computable from X only."""
    rows = []
    for pid, grp in df.groupby("patient_id"):
        ages = grp["age"].values
        m_age = float(ages.mean())

        def slope(vals: np.ndarray) -> float:
            valid = ~np.isnan(vals)
            if valid.sum() < 2:
                return 0.0
            A = np.column_stack([ages[valid] - m_age, np.ones(valid.sum())])
            p, _, _, _ = np.linalg.lstsq(A, vals[valid], rcond=None)
            return float(p[0])

        rows.append({
            "patient_id": pid,
            "pat_off_slope": slope(grp["off"].values),
            "pat_on_slope":  slope(grp["on"].values),
        })
    return pd.DataFrame(rows)


def build_patient_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build one row per patient from their visit features (no target used)."""
    pat = df.groupby("patient_id").agg(
        pm_off=("off",  "mean"), pm_on=("on",  "mean"), pm_ledd=("ledd", "mean"),
        pm_std_off=("off", "std"), pm_std_on=("on",  "std"),
        pm_min_off=("off", "min"), pm_max_off=("off", "max"),
        pm_min_on=("on",  "min"), pm_max_on=("on",  "max"),
        pm_age=("age", "mean"), age_dx=("age_at_diagnosis", "mean"),
        n_visits=("age", "count"), sexM=("sexM", "first"),
        cohort_B=("cohort", lambda x: (x == "B").mean()),
        gene_GBAp=("gene",  lambda x: (x == "GBA+").mean()),
        gene_LRRK2p=("gene", lambda x: (x == "LRRK2+").mean()),
        gene_OTHER=("gene",  lambda x: (x == "OTHER+").mean()),
        gene_NoMut=("gene",  lambda x: (x == "No Mutation").mean()),
    ).reset_index()
    slopes = compute_pat_slopes(df[["patient_id", "age", "off", "on"]])
    pat = pat.merge(slopes, on="patient_id", how="left")
    return pat


def build_visit_features(df: pd.DataFrame, l1_pred: np.ndarray | None = None) -> pd.DataFrame:
    """Build visit-level feature matrix."""
    out = df.copy()
    for col in ["cohort", "gene", "rater_id"]:
        if col in out.columns:
            out[col] = out[col].astype("category")

    out["years_dx"] = out["age"] - out["age_at_diagnosis"]

    pat = out.groupby("patient_id", sort=False).agg(
        pm_off=("off",  "mean"), pm_on=("on",  "mean"),
        pm_std_off=("off", "std"), pm_std_on=("on",  "std"),
        pm_median_off=("off", "median"), pm_ledd=("ledd", "mean"),
        pm_n_visits=("age", "count"),
    ).reset_index()
    out = out.merge(pat, on="patient_id", how="left")

    out["off_vs_mean"]   = out["off"] - out["pm_off"]
    out["on_vs_mean"]    = out["on"]  - out["pm_on"]
    out["off_z"]         = out["off_vs_mean"] / (out["pm_std_off"] + 1e-6)
    out["on_z"]          = out["on_vs_mean"]  / (out["pm_std_on"]  + 1e-6)
    out["off_vs_median"] = out["off"] - out["pm_median_off"]

    toff = out["time_since_intake_off"].fillna(out["time_since_intake_off"].median())
    out["off_pk"]      = 1.0 / (1.0 + np.exp(-0.3 * (toff - 12.0)))
    out["off_debiased"]= out["off"] * out["off_pk"]

    ton = out["time_since_intake_on"].fillna(out["time_since_intake_on"].median())
    out["on_pk"]       = 1.0 / (1.0 + np.exp(0.5  * (ton  - 2.0)))
    out["on_adjusted"] = out["on"] * out["on_pk"]

    out["off_on_ratio"] = out["off"] / (out["on"] + 1.0)
    out["off_minus_on"] = out["off"] - out["on"]
    out["ledd_per_year"]= out["ledd"] / (out["years_dx"].clip(lower=0.1))

    vr = out.groupby("patient_id")["age"].rank(method="first").astype(float)
    out["visit_rank"]     = vr
    out["visit_rank_pct"] = vr / out["pm_n_visits"]

    if l1_pred is not None:
        out["l1_pred"] = l1_pred

    drop = ["Index", "patient_id"] + (["target"] if "target" in out.columns else [])
    return out.drop(columns=[c for c in drop if c in out.columns])


# ── Cross-validation ───────────────────────────────────────────────────────────
print("\n[27] Two-level model  GroupKFold-5")
oof_l1    = np.zeros(N)  # Level-1 patient baseline
oof_final = np.zeros(N)  # Level-1 + Level-2 residual

lgbm_params = dict(
    n_estimators=2000, num_leaves=255, learning_rate=0.02,
    min_child_samples=15, colsample_bytree=0.8, subsample=0.8,
    subsample_freq=1, reg_alpha=0.05, reg_lambda=0.3, max_depth=-1,
    random_state=42, n_jobs=-1, verbose=-1,
)

for fold_i, (tr, vl) in enumerate(cv_splits):
    tr_vis = visits.iloc[tr]; vl_vis = visits.iloc[vl]

    # ── Level 1: Ridge on patient aggregates ──────────────────────────────────
    tr_pat = build_patient_features(tr_vis)
    vl_pat = build_patient_features(vl_vis)

    # Target: mean(target) per patient in the train fold
    tr_pat_target = tr_vis.groupby("patient_id")["target"].mean().reset_index()
    tr_pat_target.columns = ["patient_id", "pm_target"]
    tr_pat = tr_pat.merge(tr_pat_target, on="patient_id", how="left")

    feat_pat = [c for c in PAT_FEAT_COLS if c in tr_pat.columns]
    pipe1 = Pipeline([
        ("i", SimpleImputer()),
        ("s", StandardScaler()),
        ("r", Ridge(alpha=0.1)),
    ])
    pipe1.fit(tr_pat[feat_pat].values, tr_pat["pm_target"].values)

    vl_l1_pat = pipe1.predict(vl_pat[feat_pat].fillna(0).values)
    vl_l1_map = dict(zip(vl_pat["patient_id"], vl_l1_pat))
    oof_l1[vl] = vl_vis["patient_id"].map(vl_l1_map).values

    # ── Level 2: LGBM on visit residuals ──────────────────────────────────────
    # Build visit features with Level-1 prediction as a feature
    tr_l1_map = dict(zip(tr_pat.merge(tr_pat_target, on="patient_id", how="left")["patient_id"],
                         pipe1.predict(tr_pat[feat_pat].fillna(0).values)))
    tr_l1_visit = tr_vis["patient_id"].map(tr_l1_map).values

    X_tr_visit = build_visit_features(tr_vis, l1_pred=tr_l1_visit)
    X_vl_visit = build_visit_features(vl_vis, l1_pred=oof_l1[vl])

    residual_tr = y[tr] - tr_l1_visit

    # Align categorical columns
    for col in ["cohort", "gene", "rater_id"]:
        if col in X_tr_visit.columns and col in X_vl_visit.columns:
            X_vl_visit[col] = X_vl_visit[col].cat.set_categories(
                X_tr_visit[col].cat.categories
            )

    m2 = lgb.LGBMRegressor(**lgbm_params)
    m2.fit(X_tr_visit, residual_tr)
    oof_final[vl] = oof_l1[vl] + m2.predict(X_vl_visit)

    r1 = float(np.sqrt(np.mean((y[vl] - oof_l1[vl]) ** 2)))
    r2 = float(np.sqrt(np.mean((y[vl] - oof_final[vl]) ** 2)))
    print(f"  Fold {fold_i+1}: L1={r1:.4f}  L1+L2={r2:.4f}")

rmse_l1    = float(np.sqrt(np.mean((y - oof_l1) ** 2)))
rmse_final = float(np.sqrt(np.mean((y - oof_final) ** 2)))
print(f"\n  Level-1 only RMSE:  {rmse_l1:.4f}")
print(f"  Level-1+2  RMSE:    {rmse_final:.4f}")

# ── Blend with transformer OOF (exp 25) if available ──────────────────────────
oof_25_path = Path("scratch/25_oof_visit.csv")
if oof_25_path.exists():
    oof_25_df = pd.read_csv(oof_25_path)
    oof_25 = X_train_raw[["Index"]].merge(oof_25_df, on="Index", how="left")["pred_25"].values
    valid = ~np.isnan(oof_25)
    # Simple average blend
    blend_avg = (oof_final[valid] + oof_25[valid]) / 2
    rmse_avg  = float(np.sqrt(np.mean((y[valid] - blend_avg) ** 2)))
    # Ridge meta blend
    from sklearn.linear_model import Ridge as R
    meta = R(alpha=0.1)
    stack = np.column_stack([oof_final[valid], oof_25[valid]])
    meta.fit(stack, y[valid])
    rmse_meta = float(np.sqrt(np.mean((y[valid] - meta.predict(stack)) ** 2)))
    print(f"  Blend 27+25 avg:    {rmse_avg:.4f}")
    print(f"  Blend 27+25 Ridge:  {rmse_meta:.4f}  (coef: L2={meta.coef_[0]:.3f}, T25={meta.coef_[1]:.3f})")
else:
    print("  (scratch/25_oof_visit.csv not found — skipping blend with transformer)")
    meta = None

# ── Save OOF for blending ──────────────────────────────────────────────────────
Path("scratch").mkdir(exist_ok=True)
oof_df = visits[["Index"]].copy()
oof_df["pred_27"] = oof_final
oof_df.to_csv("scratch/27_oof.csv", index=False)
print("\n  Saved scratch/27_oof.csv")

# ── Final models on ALL training data ────────────────────────────────────────
print("\n[Final] Fitting on full training data...")
full_pat = build_patient_features(visits)
full_pat_target = visits.groupby("patient_id")["target"].mean().reset_index()
full_pat_target.columns = ["patient_id", "pm_target"]
full_pat = full_pat.merge(full_pat_target, on="patient_id", how="left")

feat_pat = [c for c in PAT_FEAT_COLS if c in full_pat.columns]
pipe1_final = Pipeline([
    ("i", SimpleImputer()), ("s", StandardScaler()), ("r", Ridge(alpha=0.1)),
])
pipe1_final.fit(full_pat[feat_pat].values, full_pat["pm_target"].values)

full_l1_map = dict(zip(full_pat["patient_id"],
                       pipe1_final.predict(full_pat[feat_pat].fillna(0).values)))
full_l1_visit = visits["patient_id"].map(full_l1_map).values
residual_full = y - full_l1_visit

X_full_visit = build_visit_features(visits, l1_pred=full_l1_visit)
m2_final = lgb.LGBMRegressor(**lgbm_params)
m2_final.fit(X_full_visit, residual_full)

# ── Test predictions ───────────────────────────────────────────────────────────
print("\n[Submission] Building test predictions...")
test_pat = build_patient_features(X_test_raw)
feat_pat_test = [c for c in feat_pat if c in test_pat.columns]
test_l1_preds = pipe1_final.predict(test_pat[feat_pat_test].fillna(0).values)
test_l1_map   = dict(zip(test_pat["patient_id"], test_l1_preds))
test_l1_visit = X_test_raw["patient_id"].map(test_l1_map).values

X_test_visit = build_visit_features(X_test_raw, l1_pred=test_l1_visit)
for col in ["cohort", "gene", "rater_id"]:
    if col in X_test_visit.columns and col in X_full_visit.columns:
        X_test_visit[col] = X_test_visit[col].cat.set_categories(
            X_full_visit[col].cat.categories
        )
X_test_visit = X_test_visit.reindex(columns=X_full_visit.columns, fill_value=np.nan)

test_l2_preds  = m2_final.predict(X_test_visit)
test_final     = test_l1_visit + test_l2_preds

Path("submissions").mkdir(exist_ok=True)
sub = X_test_raw[["Index"]].copy()
sub["target"] = test_final
sub.to_csv("submissions/27_two_level.csv", index=False)
print("  Wrote submissions/27_two_level.csv")

# Save test preds for blending
test_df = X_test_raw[["Index"]].copy()
test_df["pred_27"] = test_final
test_df.to_csv("scratch/27_test.csv", index=False)

# Blend with transformer test preds
test_25_path = Path("scratch/25_test_visit.csv")
if test_25_path.exists() and meta is not None:
    test_25_df = pd.read_csv(test_25_path)
    test_25 = X_test_raw[["Index"]].merge(test_25_df, on="Index", how="left")["pred_25"].values
    nan_mask = np.isnan(test_25)
    if nan_mask.any():
        test_25[nan_mask] = test_final[nan_mask]
    test_blend = meta.predict(np.column_stack([test_final, test_25]))
    sub_blend = X_test_raw[["Index"]].copy()
    sub_blend["target"] = test_blend
    sub_blend.to_csv("submissions/27_blend_with_transformer.csv", index=False)
    print("  Wrote submissions/27_blend_with_transformer.csv  ← BEST SUBMISSION")

print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"  12 GBM (baseline):          4.58")
print(f"  26 best blend (T25+GBM):    3.94")
print(f"  27 Level-1 only:            {rmse_l1:.4f}")
print(f"  27 Level-1 + Level-2:       {rmse_final:.4f}  ← TARGET: <= 2.00")
if oof_25_path.exists():
    print(f"  27 + 25 blend:              {rmse_meta:.4f}")
print(f"  Floor (per-pat linear):     1.61")
print(f"{'='*60}")
