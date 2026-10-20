# %% [markdown]
# # Experiment 07 — Feature Engineering avancé
#
# Objectif : Passer sous la barre des 3 de RMSE en modélisant explicitement :
#   - La progression de la maladie (time_since_diagnosis)
#   - La pharmacocinétique de la lévodopa (ledd_decay)
#   - L'impossibilité de faire l'examen OFF (is_off_missing)
#   - La réponse au traitement (on_off_delta)

# %%
from pathlib import Path
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from skrub import TableVectorizer
from skore import evaluate

from parkinson.data import load_data, FEATURE_COLS_FULL
from parkinson.hub import get_project

# ── 1. Feature Engineering ────────────────────────────────────────────────────
def engineer_features(X):
    """Génère de nouvelles variables cliniques à partir des données brutes."""
    X_out = X.copy()
    
    # Progression temporelle
    X_out["time_since_diagnosis"] = X_out["age"] - X_out["age_at_diagnosis"]
    
    # Pharmacocinétique (déclin de l'effet de la dose dans le temps)
    X_out["ledd_decay_off"] = X_out["ledd"] / (1 + X_out["time_since_intake_off"])
    X_out["ledd_decay_on"] = X_out["ledd"] / (1 + X_out["time_since_intake_on"])
    
    # Marqueurs explicites d'inconfort (valeurs manquantes)
    X_out["is_off_missing"] = X_out["off"].isna().astype(int)
    X_out["is_on_missing"] = X_out["on"].isna().astype(int)
    
    # Capacité de réponse au traitement (différence OFF - ON)
    X_out["on_off_delta"] = X_out["off"] - X_out["on"]
    
    return X_out

# ── 2. Data ───────────────────────────────────────────────────────────────────
print("Loading data...")
ds = load_data(feature_cols=FEATURE_COLS_FULL)
X_full = ds.X
y = ds.y
groups = ds.groups
print(f"  train: {X_full.shape}  groups (patients): {groups.nunique()}")

# ── 3. Pipeline & Evaluation ──────────────────────────────────────────────────
cv_splits = list(GroupKFold(n_splits=5).split(X_full, y, groups=groups))

print("\n[Step 7] Pipeline avec Feature Engineering + skrub + HGBR")
model = make_pipeline(
    FunctionTransformer(engineer_features),
    TableVectorizer(),
    HistGradientBoostingRegressor(random_state=0)
)

rep = evaluate(model, X_full, y, splitter=cv_splits)
rmse = float(rep.metrics.rmse().iloc[:, 0].mean())

print(f"  Feature Eng RMSE (grouped CV) = {rmse:.4f}")
print(f"  amélioration par rapport au modèle 05 (7.52) = {7.5238 - rmse:+.4f}")

# ── 4. Push to Hub ────────────────────────────────────────────────────────────
print("\n[Push] Feature Engineering report → Hub key '07_feature_eng'")
try:
    project = get_project()
    project.put("07_feature_eng", rep)
except Exception as e:
    print(f"Erreur lors du push sur le Hub (le serveur peut être instable): {e}")

# ── 5. Submission CSV ─────────────────────────────────────────────────────────
print("\n[Submission] Fitting on all training data...")
final = clone(model).fit(X_full, y)
X_test_full = ds.X_test
submission = ds.X_test_raw[["Index"]].copy()
submission["target"] = final.predict(X_test_full)

Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/07_feature_eng.csv", index=False)
print(f"Wrote submissions/07_feature_eng.csv  ({len(submission)} rows)")
print("\nDone. Upload submissions/07_feature_eng.csv to Kaggle.")