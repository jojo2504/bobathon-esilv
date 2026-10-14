# %% [markdown]
# # Expérience 33 — Calibration des mesures et progression par patient
#
# Nouvelle architecture construite depuis les CSV bruts :
#   1. Ridge apprend la relation score observé / cible selon le contexte.
#   2. Son inversion permet d'estimer une trajectoire initiale par patient.
#   3. Deux LightGBM corrigent cette trajectoire au niveau des visites.
#   4. Trois LightGBM prédisent niveau, pente et courbure au niveau patient.
#   5. Ensemble 35 / 35 / 30 et validation par groupes de patients.
#
# RMSE OOF : 3.35080 contre 3.54836 pour le 32, soit un gain de 5.57 %.
# Le 32 est utilisé uniquement comme référence de comparaison.
# Recherche sur le premier fold, puis évaluation des paramètres figés sur
# cinq folds. Validation non imbriquée ; aucun réseau de neurones.
# Documentation détaillée : journal/33_generative_progression.md.
# Skore : https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/87766

# %%
"""Independent non-neural model: forward measurement calibration and trajectories."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bobathon-matplotlib")
import argparse
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

KEY = "33_generative_progression"

CATS = ["cohort", "gene", "rater_id"]


class MeasurementCalibration:
    """Fit observed motor = target * slope(context) + intercept(context)."""
    def design(self, X, mode, fit=False):
        t = X[f"time_since_intake_{mode}"].fillna(1.8 if mode == "on" else 14).to_numpy()
        if fit:
            self.splines_[mode] = SplineTransformer(n_knots=6, degree=3, include_bias=False).fit(t[:, None])
        basis = self.splines_[mode].transform(t[:, None])
        rater = pd.Categorical(X.rater_id, categories=self.raters_).codes
        r = np.zeros((len(X), len(self.raters_)))
        valid = rater >= 0
        r[np.arange(len(X))[valid], rater[valid]] = 1
        missing = np.column_stack([X[c].isna().astype(float) for c in
            ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off"]])
        dose = X.ledd.fillna(600).to_numpy()/600
        base = np.column_stack([np.ones(len(X)), basis, r, missing, dose,
                               (X.cohort == "B").astype(float)])
        # A modest additive offset; most calibration is proportional to severity.
        offset = np.column_stack([np.ones(len(X)), basis, missing, dose])
        return base, offset

    def fit(self, X, y):
        self.raters_ = sorted(X.rater_id.unique())
        self.splines_, self.models_, self.noise_ = {}, {}, {}
        for mode in ["off", "on"]:
            base, offset = self.design(X, mode, True)
            mask = X[mode].notna().to_numpy() & (y > 3)
            A = np.column_stack([base*y[:, None], offset])
            model = Ridge(alpha=20, fit_intercept=False)
            model.fit(A[mask], X.loc[mask, mode], sample_weight=1/np.maximum(y[mask], 10)**2)
            self.models_[mode] = model
            residual = (model.predict(A[mask])-X.loc[mask, mode].to_numpy()) / np.maximum(y[mask],10)
            self.noise_[mode] = float(np.sqrt(np.mean(residual**2)))
        return self

    def transform(self, X):
        out = {}
        for mode in ["off", "on"]:
            base, offset = self.design(X, mode)
            c = self.models_[mode].coef_
            slope = np.clip(base@c[:base.shape[1]], 0.12, 1.5)
            intercept = offset@c[base.shape[1]:]
            out[f"latent_{mode}"] = np.clip((X[mode].to_numpy()-intercept)/slope, -20, 150)
            out[f"reliability_{mode}"] = slope**2/self.noise_[mode]**2
        return out


def patient_smooth(X, pred, strength=1):
    result = np.asarray(pred).copy()
    for idx in X.groupby("patient_id", sort=False).indices.values():
        t = X.iloc[idx].age.to_numpy(); t = t - t.mean()
        a = np.column_stack([np.ones(len(t)),t,t*t])
        curve = a@np.linalg.solve(a.T@a + np.diag([0,.01,.3]), a.T@pred[idx])
        result[idx] = strength*curve + (1-strength)*pred[idx]
    return np.clip(result, 0, 132)


class GenerativeProgression(RegressorMixin, BaseEstimator):
    def __init__(self, trees=2600, seed=1337, variant="lgb", smoothing=1.0):
        self.trees, self.seed, self.variant, self.smoothing = trees, seed, variant, smoothing

    def features(self, X):
        X = X.reset_index(drop=True)
        z = X.copy()
        latent = self.calibration_.transform(X)
        for c,v in latent.items(): z[c] = v
        z["duration"] = z.age-z.age_at_diagnosis
        z["gap"] = z.off-z.on
        for c in CATS: z[c] = pd.Categorical(z[c], categories=self.categories_[c]).codes.astype(float)
        numeric = ["age", "on", "off", "ledd", "duration", "latent_on", "latent_off",
                   "time_since_intake_on", "time_since_intake_off"]
        features = z.drop(columns=["Index", "patient_id"])
        extra = {}
        grouped = z.groupby("patient_id", sort=False)
        for col in numeric:
            for agg in ["mean", "std", "min", "max", "count", "median"]:
                extra[f"{col}_{agg}"] = grouped[col].transform(agg).to_numpy()
            extra[f"{col}_centered"] = z[col].to_numpy()-extra[f"{col}_mean"]
        keys = []
        for c in ["on", "off", "ledd", "latent_on", "latent_off", "joint"]:
            keys += [f"{c}_{s}" for s in ["linear", "slope", "quad", "curvature", "noise", "first", "last", "first_age", "last_age"]]
            for scale in [1.5, 4, 10]: keys += [f"{c}_kernel_{scale}"]
        for c in ["on", "off", "ledd", "latent_on", "latent_off"]:
            keys += [f"{c}_visit_{i}" for i in range(12)]
        keys += [f"age_visit_{i}" for i in range(12)]
        keys += ["relative_age", "age_span", "n_visits", "visit_fraction", "on_first_age", "off_last_age"]
        for key in keys: extra[key] = np.full(len(X), np.nan)
        for idx in grouped.indices.values():
            order = idx[np.argsort(X.iloc[idx].age.to_numpy())]
            ages = X.iloc[order].age.to_numpy(); center = ages.mean(); t = ages-center
            extra["relative_age"][order] = t
            extra["age_span"][order] = np.ptp(ages)
            extra["n_visits"][order] = len(order)
            extra["visit_fraction"][order] = np.arange(len(order))/max(len(order)-1,1)
            for i,a in enumerate(ages): extra[f"age_visit_{i}"][order] = a-ages
            for c in ["on", "off", "ledd", "latent_on", "latent_off", "joint"]:
                if c == "joint":
                    values = np.r_[z.iloc[order].latent_off, z.iloc[order].latent_on]
                    times = np.r_[t,t]
                    weights = np.r_[z.iloc[order].reliability_off, z.iloc[order].reliability_on]
                    weights = weights/np.nanmean(weights)
                else:
                    values = z.iloc[order][c].to_numpy(dtype=float); times=t
                    weights=np.ones(len(t))
                    for i,v in enumerate(values): extra[f"{c}_visit_{i}"][order] = v
                ok = np.isfinite(values)
                if not ok.any(): continue
                a,v,w = times[ok], values[ok], weights[ok]
                for degree,penalty,name in [(1,2,"linear"),(2,20,"quad")]:
                    design = np.column_stack([a**j for j in range(degree+1)])
                    full = np.column_stack([t**j for j in range(degree+1)])
                    reg = np.diag([0,2]+([penalty] if degree==2 else []))
                    coef = np.linalg.solve(design.T@(w[:,None]*design)+reg,design.T@(w*v))
                    extra[f"{c}_{name}"][order] = full@coef
                    extra[f"{c}_{'slope' if degree==1 else 'curvature'}"][order] = coef[degree]
                    if degree==2: extra[f"{c}_noise"][order] = np.sqrt(np.mean((v-design@coef)**2))
                imin,imax = np.argmin(a),np.argmax(a)
                for tag,i in [("first",imin),("last",imax)]:
                    extra[f"{c}_{tag}"][order]=v[i]
                    extra[f"{c}_{tag}_age"][order]=a[i]-t
                for scale in [1.5,4,10]:
                    weight=np.exp(-.5*((t[:,None]-a[None,:])/scale)**2)*w[None,:]
                    extra[f"{c}_kernel_{scale}"][order] = (weight@v)/weight.sum(axis=1)
        return pd.concat([features,pd.DataFrame(extra)],axis=1).replace([np.inf,-np.inf],np.nan)

    def fit(self, X, y):
        start=time.monotonic(); X=X.reset_index(drop=True); y=np.asarray(y)
        self.categories_={c:sorted(X[c].dropna().unique()) for c in CATS}
        self.calibration_=MeasurementCalibration().fit(X,y)
        f=self.features(X)
        self.feature_names_=f.columns.tolist()
        # Learn a residual around a physically calibrated trajectory.
        baseline=f.joint_quad.fillna(35).to_numpy()
        if self.variant == "xgb":
            self.model_=XGBRegressor(n_estimators=self.trees,max_depth=5,learning_rate=.025,
                min_child_weight=12,reg_lambda=15,subsample=.85,colsample_bytree=.9,
                n_jobs=6,tree_method="hist",random_state=self.seed)
        else:
            self.model_=lgb.LGBMRegressor(n_estimators=self.trees,num_leaves=31 if self.variant=="small" else 63,
                learning_rate=.025,min_child_samples=30,colsample_bytree=.85,
                subsample=.85,subsample_freq=1,reg_lambda=5,reg_alpha=.1,
                n_jobs=6,verbosity=-1,random_state=self.seed)
        self.model_.fit(f,y-baseline)
        print(f"{self.variant}: fitted {len(X)} visits/{f.shape[1]} features in {time.monotonic()-start:.1f}s",flush=True)
        return self

    def predict(self,X):
        f=self.features(X)
        pred=f.joint_quad.fillna(35).to_numpy()+self.model_.predict(f)
        return patient_smooth(X.reset_index(drop=True),pred,self.smoothing)


def patient_basis(X, f, residual=None):
    """Patient feature means and orthogonal temporal bases, in original row order."""
    rows, targets, bases, indices = [], [], [], []
    for idx in X.reset_index(drop=True).groupby('patient_id',sort=False).indices.values():
        t=X.iloc[idx].age.to_numpy(); t=t-t.mean()
        t=t/max(np.sqrt(np.mean(t*t)),.1)
        q=t*t-np.mean(t*t); q-=t*(t@q)/(t@t+1e-8)
        q=q/max(np.sqrt(np.mean(q*q)),.1)
        a=np.column_stack([np.ones(len(t)),t,q])
        rows.append(f.iloc[idx].mean().to_numpy()); bases.append(a); indices.append(idx)
        if residual is not None:
            targets.append(np.linalg.lstsq(a,residual[idx],rcond=None)[0])
    return np.array(rows),np.array(targets),bases,indices


class ProgressionEnsemble(GenerativeProgression):
    """35% leaf-15 GBM + 35% leaf-31 GBM + 30% patient-curve GBMs.

    Hyperparameters fixed after exploration on the first grouped fold only.
    The other four outer validation folds are not consulted during fitting.
    """
    def __init__(self, seed=333):
        super().__init__(seed=seed)

    def fit(self, X, y):
        start=time.monotonic(); X=X.reset_index(drop=True); y=np.asarray(y)
        digest=hashlib.sha256(pd.util.hash_pandas_object(X,index=False).values.tobytes()+y.tobytes()+str(self.seed).encode()).hexdigest()[:16]
        cache=Path(f'scratch/{KEY}_fit_v1_{digest}.joblib')
        if cache.exists():
            self.__dict__.update(joblib.load(cache).__dict__)
            print('Reused fitted ensemble',digest,flush=True)
            return self
        self.categories_={c:sorted(X[c].dropna().unique()) for c in CATS}
        self.calibration_=MeasurementCalibration().fit(X,y)
        f=self.features(X); baseline=f.joint_quad.fillna(35).to_numpy()
        self.feature_names_=f.columns.tolist(); self.models_=[]
        for leaves,trees in [(15,6499),(31,6497)]:
            gb=lgb.LGBMRegressor(n_estimators=trees,num_leaves=leaves,learning_rate=.025,
                min_child_samples=35,colsample_bytree=.9,subsample=.9,subsample_freq=1,
                reg_lambda=8,reg_alpha=.1,n_jobs=6,verbosity=-1,random_state=self.seed)
            gb.fit(f,y-baseline); self.models_.append(gb)
            print(f'  visit GBM ({leaves} leaves) trained',flush=True)
        F,Y,_,_=patient_basis(X,f,y-baseline)
        self.patient_models_=[]
        for i,trees in enumerate([2998,2333,2067]):
            gb=lgb.LGBMRegressor(n_estimators=trees,num_leaves=15,learning_rate=.02,
                min_child_samples=25,colsample_bytree=.9,reg_lambda=10,n_jobs=6,
                verbosity=-1,random_state=345)
            gb.fit(F,Y[:,i]); self.patient_models_.append(gb)
        joblib.dump(self,cache)
        print(f'Ensemble fitted on {len(X)} visits in {time.monotonic()-start:.1f}s',flush=True)
        return self

    def predict(self,X):
        # Canonical ordering makes floating-point reductions and patient curve
        # features deterministic even if callers shuffle the input visits.
        X=X.reset_index(drop=True)
        order=X.sort_values(['patient_id','age','Index'],kind='stable').index.to_numpy()
        X=X.iloc[order].reset_index(drop=True); f=self.features(X)
        baseline=f.joint_quad.fillna(35).to_numpy()
        result=np.zeros(len(X))
        for gb in self.models_:
            result+=.35*patient_smooth(X,baseline+gb.predict(f))
        F,_,bases,indices=patient_basis(X,f)
        coef=np.column_stack([gb.predict(F) for gb in self.patient_models_])
        patient_pred=baseline.copy()
        for i,(a,idx) in enumerate(zip(bases,indices)):
            patient_pred[idx]+=a@coef[i]
        restored=np.empty(len(X))
        restored[order]=np.clip(result+.3*patient_pred,0,132)
        return restored


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--push',action='store_true')
    args=parser.parse_args()
    X=pd.read_csv('data/X_train.csv')
    y=X[['Index']].merge(pd.read_csv('data/y_train.csv'),on='Index',validate='one_to_one').target.to_numpy()
    test=pd.read_csv('data/X_test.csv')
    assert set(X.patient_id).isdisjoint(test.patient_id)
    splits=list(GroupKFold(5).split(X,y,groups=X.patient_id))
    if args.probe:
        tr,vl=splits[0]
        for variant in ['small','lgb','xgb']:
            model=GenerativeProgression(variant=variant).fit(X.iloc[tr],y[tr])
            pred=model.predict(X.iloc[vl]); rmse=np.sqrt(np.mean((pred-y[vl])**2))
            print('PROBE',variant,'RMSE',rmse,flush=True)
            joblib.dump(model,f'scratch/33_probe_{variant}.joblib')
            np.save(f'scratch/33_probe_{variant}.npy',pred)
        return
    from skore import evaluate
    path=Path(f'scratch/{KEY}_report.joblib')
    if path.exists():
        report=joblib.load(path)
        # Recompute actual held-out predictions with the loaded estimator code,
        # including canonical ordering, while retaining original training times.
        report._clear_cache()
    else:
        report=evaluate(ProgressionEnsemble(),X,y,splitter=splits,n_jobs=1)
        joblib.dump(report,path)
    oof=np.full(len(X),np.nan); fold_rmse=[]
    for (_,vl),pred in zip(splits,report.get_predictions(data_source='test'),strict=True):
        oof[vl]=pred; fold_rmse.append(float(np.sqrt(np.mean((pred-y[vl])**2))))
    joblib.dump(report,path)
    assert np.isfinite(oof).all()
    pd.DataFrame({'Index':X.Index,'prediction':oof}).to_csv(f'scratch/{KEY}_oof.csv',index=False)
    previous=X[['Index']].merge(pd.read_csv('scratch/32_final_trajectory_oof.csv'),on='Index',validate='one_to_one').prediction.to_numpy()
    confirm=np.ones(len(X),dtype=bool); confirm[splits[0][1]]=False
    metrics={'key':KEY,'rmse_oof':float(np.sqrt(np.mean((oof-y)**2))),
        'fold_rmse':fold_rmse,'baseline_32_rmse':float(np.sqrt(np.mean((previous-y)**2))),
        'confirmation_rmse_folds_2_to_5':float(np.sqrt(np.mean((oof[confirm]-y[confirm])**2))),
        'baseline_32_confirmation_rmse':float(np.sqrt(np.mean((previous[confirm]-y[confirm])**2))),
        'selection':'Tuning validation uses fold 1; folds 2-5 are additional validation. Non-nested CV.',
        'n_train':len(X),'n_test':len(test),'neural_network':False}
    metrics['relative_improvement_percent']=100*(1-metrics['rmse_oof']/metrics['baseline_32_rmse'])
    print(json.dumps(metrics,indent=2),flush=True)
    Path(f'scratch/{KEY}_metrics.json').write_text(json.dumps(metrics,indent=2))
    if metrics['confirmation_rmse_folds_2_to_5']>=metrics['baseline_32_confirmation_rmse']:
        raise RuntimeError('Candidate does not improve on 32 on confirmation folds; do not publish.')
    final=ProgressionEnsemble().fit(X,y)
    joblib.dump(final,f'scratch/{KEY}_model.joblib')
    submission=pd.DataFrame({'Index':test.Index,'target':final.predict(test)})
    sample=pd.read_csv('data/sample_submission.csv')
    assert submission.Index.is_unique and set(submission.Index)==set(sample.Index)
    submission=sample[['Index']].merge(submission,on='Index',validate='one_to_one')
    assert np.isfinite(submission.target).all() and submission.target.between(0,132).all()
    submission.to_csv(f'submissions/{KEY}.csv',index=False)
    if args.push:
        from parkinson.hub import get_project
        project=get_project(); project.put(KEY,report)
        summary=project.summarize().frame(); matching=summary.loc[summary['key']==KEY]
        assert not matching.empty
        metrics['hub_report_id']=str(matching.index.get_level_values('id')[-1])
        Path(f'scratch/{KEY}_metrics.json').write_text(json.dumps(metrics,indent=2))
    print('Completed',KEY,flush=True)


if __name__=='__main__': main()
