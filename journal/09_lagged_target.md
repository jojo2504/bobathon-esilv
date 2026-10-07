# Experiment 09 — Within-Patient Lagged Target Features

## Question / hypothesis

Can within-patient expanding mean of the true target (sorted by age as time proxy) cut RMSE below 4 by capturing individual disease progression in a leak-free way?

## Motivation

Patient-level mean(off) already has r=0.93 with patient-level mean(target), driving the 5.17→4.91 improvement. But `off` is a noisy, biased proxy. The true target itself, averaged across a patient's *earlier* visits, should be a much stronger predictor of the *current* visit's target. 

In GroupKFold, **all visits of a validation patient are together in the val fold** — so we can compute each visit's "expanding mean of the patient's prior targets (older visits)" without leakage. For a patient with visits at ages 62, 64, 66, the visit at 66 can see the mean target of ages 62 and 64. The visit at 62 (first) gets `NaN`, filled with the patient's own `pat_mean_off` (X-only proxy). This is "temporal progression" — the clearest individual baseline.

At test time there is no `target` in `X_test`, but the test patients' `off`/`on` columns are proxies with r≈0.87. We use a `WithinPatientLagTransformer` sklearn transformer that:
- During `fit`: stores a global fallback (median of train-fold expanding-mean targets).
- During `transform`: for **train** rows, computes the correct fold-local expanding mean. For **test** rows, falls back to: (1) the patient's own mean(off) scaled by a factor learned at fit time, or (2) the global fallback.

The key: the transformer is wrapped around the model inside CV, so each fold's train-only targets are used.

## Method

1. Sort each patient's visits by `age` to establish temporal order.
2. For each visit, compute `lag_target_mean` = expanding mean of all prior visits' `target` within that patient (shift by 1 so no self-leakage). First visit = NaN.
3. Also compute: `lag_target_std`, `lag_target_last` (most recent prior), `lag_target_count` (how many prior visits exist).
4. Fill NaN with the patient's `pat_mean_off` scaled by the train-fold `target / off` ratio. This is the best X-only proxy when no prior target exists.
5. Wrap in a custom sklearn transformer (`WithinPatientLagTransformer`) that takes `patient_id` and `age` as extra context columns alongside X.
6. Build on top of experiment 08's feature set + LightGBM (tuned params).
7. Evaluate with the same `GroupKFold(5)` on `patient_id`.

## Risks

- **Implementation complexity**: The transformer needs careful fold-aware logic. Verified: GroupKFold ensures all visits of a patient land in the same fold, so there is no cross-patient leakage. Within-patient temporal split is safe.
- **Age ties**: Some patients may have multiple visits at the same age. We break ties by row order (stable sort).
- **Test leakage check**: At test time, `target` is not available. The fallback to X-only features means some information loss. The test RMSE improvement may be smaller than OOF RMSE improvement.
- **Overfitting to patient count**: Patients with many visits get good lag estimates; patients with few visits fall back to the proxy.

## Status

- **State**: done
- **Approved by user on**: 2025-07-15
- **Headline result**: RMSE 8.02 OOF (autoregressive, no leakage). Hub leaky version 2.11.
- **Implication for next iteration**: Target leakage unavailable at test time — test patients entirely held out. Autoregressive prediction accumulates errors. Confirms the signal ceiling is reachable only with real targets.

**Sourcing strategy**: `user` (explicit strategy from conversation)
