# %% [markdown]
# # Expérience 32 — Ensemble d'arbres et trajectoires par patient
#
# Idée : exploiter toutes les visites observées d'un patient pour estimer
# sa progression et réduire le bruit des scores ON/OFF.
# Les courbes linéaires/quadratiques sont calculées depuis X uniquement.
#
# Stratégie :
#   1. Deux petits LightGBM apprennent les corrections target - off / on.
#      Ils utilisent les données cliniques, rater_id et les délais de prise.
#      Ils sont réentraînés dans chaque fold, sur les patients train seuls.
#   2. Construction de 121 variables : agrégats patient, trajectoires
#      régularisées, observations voisines et indicateurs pharmacocinétiques.
#   3. Moyenne 50/50 de LightGBM (2 200 arbres, 63 feuilles) et
#      XGBoost (1 760 arbres, profondeur 5), sans réseau de neurones.
#   4. Lissage quadratique par patient : 85 % courbe + 15 % prédiction brute.
#   5. Validation GroupKFold-5 via skore.evaluate, entraînement final,
#      génération du CSV et publication du rapport avec --push.
#
# Résultats mesurés :
#   - RMSE OOF global : 3.54836.
#   - Folds : 3.56730 / 3.59380 / 3.46349 / 3.47063 / 3.64308.
#   - Anciennes prédictions disponibles (T17) : 3.93637, gain de 9.86 %.
#   - Le lissage a été choisi sur le premier fold : ce score n'est donc
#     pas une évaluation sur un jeu de test indépendant de toute sélection.
#
# Livrables : submissions/32_final_trajectory.csv (11 013 visites),
# modèle et rapport dans scratch/, documentation journal/32_final_trajectory.md.
# Skore : https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/86150

# %%
"""Final tree ensemble: fold-local motor correction and patient trajectories.

Run from the repository root. No neural network or old predictions are used.
All learned corrections are fitted exclusively on training-fold patients.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bobathon-matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "6")

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_is_fitted
from skore import evaluate
from xgboost import XGBRegressor

KEY = "32_final_trajectory"
CATS = ["cohort", "gene", "rater_id"]


# %% [markdown]
# ## Lissage des prédictions au sein de chaque patient
#
# Une courbe quadratique régularisée décrit la progression avec l'âge.
# La régularisation limite les courbures excessives lorsque les visites sont rares.

# %%
def smooth(x, pred, strength=1.0):
    """Project visit predictions onto regularized quadratic patient curves."""
    result = np.array(pred, dtype=float, copy=True)
    for idx in x.groupby("patient_id", sort=False).indices.values():
        age = x.iloc[idx].age.to_numpy()
        age = age - age.mean()
        design = np.column_stack([np.ones(len(idx)), age, age ** 2])
        penalty = np.diag([0.0, 0.02, 1.0])
        fitted = design @ np.linalg.solve(design.T @ design + penalty, design.T @ pred[idx])
        result[idx] = strength * fitted + (1 - strength) * pred[idx]
    return result


# %% [markdown]
# ## Estimateur : corrections ON/OFF, variables temporelles et ensemble
#
# fit() apprend les corrections et les deux modèles de prédiction.
# predict() calcule les variables depuis les visites fournies, puis applique
# la moyenne des arbres, le lissage et les bornes du score [0, 132].
# Fournir ensemble toutes les visites disponibles d'un patient permet
# de reconstruire ses agrégats et sa trajectoire comme en validation.

# %%
class TrajectoryRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, trees=2200, smoothing=0.85, seed=2026):
        self.trees = trees
        self.smoothing = smoothing
        self.seed = seed

    def raw_features(self, x):
        f = x.drop(columns=["Index", "patient_id"], errors="ignore").copy()
        for c in CATS:
            f[c] = pd.Categorical(f[c], categories=self.categories_[c])
        f["duration"] = f.age - f.age_at_diagnosis
        f["gap"] = f.off - f.on
        for c in ["on", "off"]:
            t = f[f"time_since_intake_{c}"]
            for decay in [0.2, 0.5, 1.0]:
                f[f"{c}_pk_{decay}"] = np.exp(-decay * t)
                f[f"{c}_dose_{decay}"] = f.ledd * np.exp(-decay * t)
        return f

    def features(self, x):
        x = x.reset_index(drop=True)
        f = self.raw_features(x)
        for c in ["on", "off"]:
            f[f"corrected_{c}"] = x[c] + self.corrections_[c].predict(self.raw_features(x))
            f.loc[x[c].isna(), f"corrected_{c}"] = np.nan
        work = pd.concat([x[["patient_id", "age"]], f.drop(columns="age")], axis=1)
        groups = work.groupby("patient_id", sort=False)
        additions = {}
        for c in ["off", "on", "ledd", "age", "duration", "corrected_on", "corrected_off"]:
            for agg in ["mean", "std", "min", "max", "count"]:
                additions[f"pat_{c}_{agg}"] = groups[c].transform(agg).to_numpy()
            additions[f"delta_{c}"] = work[c].to_numpy() - additions[f"pat_{c}_mean"]
        additions["rank"] = groups.age.rank().to_numpy()
        additions["relative_age"] = x.age.to_numpy() - additions["pat_age_mean"]
        # Each trajectory is estimated from observed X values of this patient.
        trajectory_cols = ["off", "on", "ledd", "corrected_off", "corrected_on"]
        for c in trajectory_cols:
            for name in ["slope", "linear", "quadratic", "resid_std", "nearest", "nearest_distance"]:
                additions[f"{c}_{name}"] = np.full(len(x), np.nan)
        for idx in groups.indices.values():
            age = x.iloc[idx].age.to_numpy()
            centered = age - age.mean()
            for c in trajectory_cols:
                values = work.iloc[idx][c].to_numpy(dtype=float)
                ok = np.isfinite(values)
                if not ok.any():
                    continue
                a, v = centered[ok], values[ok]
                linear = np.column_stack([np.ones(ok.sum()), a])
                coef = np.linalg.solve(linear.T @ linear + np.diag([0, 1.0]), linear.T @ v)
                all_linear = np.column_stack([np.ones(len(idx)), centered]) @ coef
                quad = np.column_stack([np.ones(ok.sum()), a, a*a])
                qcoef = np.linalg.solve(quad.T @ quad + np.diag([0, 1, 20]), quad.T @ v)
                all_quad = np.column_stack([np.ones(len(idx)), centered, centered**2]) @ qcoef
                distance = np.abs(centered[:, None] - a[None, :])
                nearest = distance.argmin(axis=1)
                additions[f"{c}_slope"][idx] = coef[1]
                additions[f"{c}_linear"][idx] = all_linear
                additions[f"{c}_quadratic"][idx] = all_quad
                additions[f"{c}_resid_std"][idx] = np.sqrt(np.mean((v - linear @ coef)**2))
                additions[f"{c}_nearest"][idx] = v[nearest]
                additions[f"{c}_nearest_distance"][idx] = distance.min(axis=1)
        f = pd.concat([f, pd.DataFrame(additions)], axis=1)
        # Provide every neighboring observation, preserving temporal distances.
        ordered = work.sort_values(["patient_id", "age"])
        for offset in [-2, -1, 1, 2]:
            shifted = ordered.groupby("patient_id")[
                ["age", "off", "on", "corrected_off", "corrected_on"]
            ].shift(offset).reindex(work.index)
            for c in shifted:
                f[f"neighbor_{offset}_{c}"] = shifted[c].to_numpy() - (x.age.to_numpy() if c == "age" else 0)
        return f.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def numeric(f):
        f = f.copy()
        for c in CATS:
            f[c] = f[c].cat.codes.astype(float)
        return f

    def fit(self, X, y):
        start = time.monotonic()
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        self.categories_ = {c: sorted(X[c].dropna().unique()) for c in CATS}
        raw = self.raw_features(X)
        self.corrections_ = {}
        for c in ["off", "on"]:
            mask = X[c].notna().to_numpy()
            model = lgb.LGBMRegressor(n_estimators=350, learning_rate=0.035,
                num_leaves=15, min_child_samples=100, reg_lambda=20,
                n_jobs=6, verbosity=-1, random_state=self.seed)
            model.fit(raw.loc[mask], y[mask] - X.loc[mask, c].to_numpy())
            self.corrections_[c] = model
        f = self.features(X)
        self.columns_ = f.columns.tolist()
        self.models_ = [
            lgb.LGBMRegressor(n_estimators=self.trees, learning_rate=0.025,
                num_leaves=63, min_child_samples=25, colsample_bytree=0.85,
                subsample=0.85, subsample_freq=1, reg_lambda=3,
                reg_alpha=0.05, n_jobs=6, verbosity=-1, random_state=self.seed),
            XGBRegressor(n_estimators=int(self.trees * 0.8), learning_rate=0.035,
                max_depth=5, min_child_weight=15, subsample=0.85,
                colsample_bytree=0.85, reg_lambda=10, reg_alpha=0.05,
                tree_method="hist", n_jobs=6, random_state=self.seed),
        ]
        self.models_[0].fit(f, y)
        self.models_[1].fit(self.numeric(f), y)
        print(f"Fitted {len(X)} visits, {f.shape[1]} features in {time.monotonic()-start:.1f}s", flush=True)
        return self

    def predict(self, X):
        check_is_fitted(self, "models_")
        f = self.features(X).reindex(columns=self.columns_)
        pred = 0.5 * self.models_[0].predict(f) + 0.5 * self.models_[1].predict(self.numeric(f))
        return np.clip(smooth(X.reset_index(drop=True), pred, self.smoothing), 0, 132)


# %% [markdown]
# ## Validation, entraînement final et publication Skore
#
# Les labels sont alignés sur Index et les folds séparent les patient_id.
# --probe compare le lissage sur le premier fold ; --push publie le rapport.
# Les fichiers du modèle et du rapport sont réutilisés s'ils existent déjà.
# Après modification du modèle ou des données, les déplacer ou les supprimer
# explicitement avant un nouvel entraînement pour éviter un résultat obsolète.

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    X = pd.read_csv("data/X_train.csv")
    labels = pd.read_csv("data/y_train.csv")
    y = X[["Index"]].merge(labels, on="Index", validate="one_to_one").target.to_numpy()
    test = pd.read_csv("data/X_test.csv")
    assert set(X.patient_id).isdisjoint(test.patient_id)
    splits = list(GroupKFold(5).split(X, y, groups=X.patient_id))
    if args.probe:
        tr, vl = splits[0]
        model = TrajectoryRegressor().fit(X.iloc[tr], y[tr])
        for strength in [0, 0.5, 0.85, 1]:
            model.smoothing = strength
            pred = model.predict(X.iloc[vl])
            print("Probe smoothing", strength, "RMSE", np.sqrt(np.mean((pred-y[vl])**2)), flush=True)
        return
    report_path = Path(f"scratch/{KEY}_report.joblib")
    if report_path.exists():
        report = joblib.load(report_path)
    else:
        report = evaluate(TrajectoryRegressor(), X, y, splitter=splits, n_jobs=1)
        joblib.dump(report, report_path)
    oof = np.full(len(X), np.nan)
    fold_scores = []
    for (_, vl), pred in zip(splits, report.get_predictions(data_source="test"), strict=True):
        oof[vl] = pred
        fold_scores.append(float(np.sqrt(np.mean((pred-y[vl])**2))))
    assert np.isfinite(oof).all()
    rmse = float(np.sqrt(np.mean((oof-y)**2)))
    print("Fold RMSE:", fold_scores, "Pooled OOF RMSE:", rmse, flush=True)
    pd.DataFrame({"Index": X.Index, "prediction": oof}).to_csv(f"scratch/{KEY}_oof.csv", index=False)
    model_path = Path(f"scratch/{KEY}_model.joblib")
    if model_path.exists():
        final = joblib.load(model_path)
    else:
        final = TrajectoryRegressor().fit(X, y)
        joblib.dump(final, model_path)
    submission = pd.DataFrame({"Index": test.Index, "target": final.predict(test)})
    sample = pd.read_csv("data/sample_submission.csv")
    assert submission.Index.is_unique and set(submission.Index) == set(sample.Index)
    submission = sample[["Index"]].merge(submission, on="Index", validate="one_to_one")
    assert np.isfinite(submission.target).all()
    submission.to_csv(f"submissions/{KEY}.csv", index=False)
    metrics = {"key": KEY, "rmse_oof": rmse, "fold_rmse": fold_scores,
               "n_train": len(X), "n_test": len(test), "neural_network": False,
               "validation": "GroupKFold(5), no patient overlap, fold-local corrections"}
    if args.push:
        from parkinson.hub import get_project
        project = get_project()
        project.put(KEY, report)
        summary = project.summarize().frame()
        matches = summary.loc[summary["key"] == KEY]
        if matches.empty:
            raise RuntimeError("Uploaded report is missing from the Hub summary")
        metrics["hub_report_id"] = str(matches.index.get_level_values("id")[-1])
    Path(f"scratch/{KEY}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
