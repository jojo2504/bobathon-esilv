# Experiment 14 — Transformer v2: enriched features + early stopping

## Question / hypothesis

Can injecting the engineered features from experiment 12 (patient aggregates, PK
timing, motor gap, z-scores, visit rank) into the Transformer's per-visit token
vector push the CV RMSE below 3.91 — and ideally below 2.09 (the leaky ceiling)?

## Motivation

Experiment 13 plateaus around RMSE 3.91 with only the 21 raw features per visit.
Experiment 12 achieves RMSE 4.58 with gradient boosting using a richer feature set
(~35 features: patient aggregates, PK sigmoid weights, motor gap, within-patient
z-scores, visit rank). The Transformer sees each visit as a token — the richer the
token, the better the attention can exploit cross-visit context.

The RMSE 2.09 from experiment 09 is leaky (uses true targets as lag features at
prediction time). The honest ceiling is the Transformer with all available X-only
information.

Additional fix vs. 13: **early stopping** (patience=15 epochs on val ensemble MSE)
stops training before the model starts memorizing, which was the SKD001 overfitting
signal visible in the skore checks.

## Method

**Features per visit (~35 total):**
- All 21 features from experiment 13 (raw + flags)
- Patient-level aggregates merged per visit: `pat_mean_off`, `pat_mean_on`,
  `pat_std_off`, `pat_std_on`, `pat_median_off`, `pat_min_off`, `pat_max_off`,
  `pat_n_visits`, `pat_age_range`
- PK timing features: `off_pk_weight`, `on_pk_weight`, `off_debiased`, `on_adjusted`
- Motor gap: `off_minus_on`, `off_on_ratio`, `off_z_pat`, `on_z_pat`
- Visit rank: `visit_rank`, `visit_rank_pct`

**Architecture:** identical to experiment 13 (hidden_dim=64, n_head=4, num_layers=4,
n_ensemble=5). Only the input dimension changes (n_features per visit ≈ 35 vs 21).

**Training changes vs. 13:**
- Early stopping: if val ensemble MSE does not improve by >0.01 for 15 consecutive
  checks (every 5 epochs), stop early and restore best weights.
- Max epochs: 150 (was 100) — early stopping will trigger before if needed.
- Everything else identical (AdamW lr=1e-3, StepLR, GroupKFold-5, ensemble=5).

## Risks

- Patient aggregates computed over the entire patient (not fold-local) — minor
  leakage, same as experiment 12 which already accepted this trade-off.
- Richer features may need more regularisation; dropout already at 0.1.

## Status

- **State**: approved
- **Approved by user on**: 2025-07-15
- **Headline result**: —
- **Implication for next iteration**: —

**Sourcing strategy**: `user` (direct follow-up on 13 + 12 enrichment)
