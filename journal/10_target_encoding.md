# Experiment 10 — Leave-One-Out Target Encoding of patient_id

## Question / hypothesis

Does leave-one-out (LOO) target encoding of `patient_id` within each CV fold — with demographic fallback for test patients — materially reduce RMSE by directly capturing individual patient baseline severity?

## Motivation

`patient_id` explains ~79% of target variance (patient-level mean(target) vs per-visit target). The current models discard patient identity entirely after computing hand-crafted X-aggregates. LOO target encoding computes, for each training visit, the mean target of that patient's *other* training visits (not the current one), then joins this as a feature. This is leak-free within a fold and avoids the overfitting that raw target encoding causes.

For test patients (unseen at train time), we fall back to cohort × gene group mean, then cohort mean, then global mean. This is a principled degradation.

## Method

1. Implement `LooPatiendTargetEncoder` sklearn transformer that:
   - `fit(X, y)`: computes per-patient LOO mean = (sum(y_patient) - y_i) / (n_patient - 1) for each row; stores patient → mean(y) map for test fallback.
   - `transform(X, y=None)`: for train rows returns the precomputed LOO value; for test (patient not in fit set) falls back to cohort × gene → cohort → global mean (computed at fit time).
2. Add `patient_id` as a passthrough column (not dropped) up to the encoding step.
3. Add the encoded value as `patient_target_enc` feature alongside all 08 features.
4. Use LightGBM (same params as 08_lgbm_v2) for fair comparison.
5. Evaluate `GroupKFold(5)` on `patient_id`.

## Risks

- **Test fallback quality**: test patients with rare gene/cohort combos may fall back to global mean, providing less signal.
- **LOO and small patient groups**: patients with only 1 visit have an undefined LOO mean (division by zero). We handle this by using the patient's own observed value (no leave-out possible) or the group mean.
- **Interaction with lagged features**: experiment 09 already captures patient history through the target. Running this as a standalone baseline experiment (not stacked on 09) keeps the comparison clean.

## Status

- **State**: done
- **Approved by user on**: 2025-07-15
- **Headline result**: RMSE 10.48 OOF. Test patients fully unseen → cohort/gene fallback adds no signal beyond existing pat_mean_off.
- **Implication for next iteration**: LOO encoding only helps if test patients are seen at train time. For this competition structure, it is not applicable.

**Sourcing strategy**: `user` (explicit strategy from conversation)
