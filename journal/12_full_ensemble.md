# Experiment 12 — Full-Power Ensemble: LGBM + XGB + HGBR + Ridge Stacking

## Question / hypothesis

Does using full-power base models (3000/2000/500 trees, deeper leaves, extra features) with a Ridge meta-learner beat the 4.91 hand-tuned blend?

## Motivation

Experiment 11 showed stacking works but was limited by 800-tree base models (RMSE 4.83). The key lever is model quality — more trees, deeper leaves, more features. This experiment runs the full intended configuration from the start: LGBM at 3000 trees with 255 leaves, XGB at 2000 trees with depth 8, HGBR at 500 trees with depth 8, plus additional features (within-patient z-scores of off/on, visit rank percentile, ledd × off interaction, age × off interaction).

## Method

1. Feature set: 08 features + within-patient z-scores (`off_z_pat`, `on_z_pat`), visit rank (`visit_rank`, `visit_rank_pct`), cross-feature interactions (`ledd_x_off`, `age_x_off`), `pat_mean_dx_age` → 49 total features.
2. Three base models: LightGBM (n=3000, leaves=255, lr=0.015), XGBoost (n=2000, depth=8, lr=0.02), HGBR (n=500, depth=8, lr=0.05).
3. Collect OOF predictions with GroupKFold(5).
4. Ridge(alpha=0.1) meta-learner on 3-column OOF stack.
5. Final submission: refit all 3 on full train, Ridge meta combines test predictions.
6. Also save LGBM-only submission for comparison.

## Risks

- Compute: ~25 min per model × 3 = 75 min for OOF + 25 min for Hub report + 3 final fits ≈ 2h total.
- Stacking on OOF is slightly optimistic (meta sees all OOF, not nested OOF). The true test RMSE may be slightly higher.

## Status

- **State**: done
- **Approved by user on**: 2025-07-15
- **Headline result**: RMSE **4.58** OOF (Ridge-stacked) — LGBM 4.62, XGB 4.65, HGBR 4.69. New best. Hub: https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/48670
- **Implication for next iteration**: Full-power models improve by ~0.33 over 08. Next: Optuna hyperparameter search on LGBM/XGB to push towards RMSE 4.0, or pseudo-labeling with test predictions as soft labels.

**Sourcing strategy**: `user` (learnings from 09/10/11 + explicit prior strategy)
