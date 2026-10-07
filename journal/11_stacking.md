# Experiment 11 — Stacking: OOF Predictions + Ridge Meta-Learner

## Question / hypothesis

Does a Ridge meta-learner trained on out-of-fold (OOF) predictions from diverse base models (LightGBM, XGBoost, HGBR, Ridge-engineered) reduce RMSE beyond the best single model's 4.91?

## Motivation

Experiments 08's blend (40% LGBM + 60% XGB = 4.91) uses hand-tuned weights. A Ridge meta-learner can learn optimal weights from OOF predictions, and also combine predictions from models with different inductive biases (boosting vs. linear). The diverse errors of these models are partially independent, and Ridge's L2 regularization prevents the meta-learner from overfitting to the OOF folds.

This is the standard Level-2 stacking approach: for each fold, train each base model on the train fold and collect predictions on the val fold. This gives N_train OOF predictions. The meta-learner is then fit on these. At test time, each base model is refit on all training data and predicts on test; the meta-learner combines those.

## Method

1. Define 4 base learners (same as 08 features):
   - LightGBM (3000 est, num_leaves=255, lr=0.015)
   - XGBoost (2000 est, max_depth=7, lr=0.02)
   - HGBR (500 est, max_depth=6)
   - Ridge (alpha=1.0, on the same engineered features + StandardScaler)
2. Collect OOF predictions for each base learner using `GroupKFold(5)`.
3. Stack the 4 OOF arrays as columns → 44,590 × 4 matrix.
4. Fit `Ridge(alpha=0.1)` meta-learner on this matrix (no intercept-less is fine).
5. Evaluate: meta-learner OOF RMSE (using the same 5-fold structure, i.e., the meta-learner's "OOF" is approximated by nested CV or by the fact that each row's OOF predictions already came from fold-withheld models).
6. For submission: refit each base model on all train data, predict test; meta-learner predicts from these 4 test predictions.

## Risks

- **Nested evaluation**: strictly speaking, the Ridge meta is also trained on OOF, so its own generalization is one level removed. We report the OOF RMSE of each base model plus the meta, and note that the meta RMSE is slightly optimistic (it sees all OOF predictions, not a nested OOF).
- **Base model diversity**: LGBM + XGB are correlated (~same features). HGBR and Ridge add diversity. If diversity is too low, stacking won't beat blending significantly.
- **Compute**: training 4 models × 5 folds = 20 fits. At current params, ~15 min total.

## Status

- **State**: done
- **Approved by user on**: 2025-07-15
- **Headline result**: RMSE 4.83 OOF (stacked meta). Base models capped at 800 trees — improvement over 4.91 but limited by model quality, not the stacking approach.
- **Implication for next iteration**: Full-power base models (3000/2000 trees) needed. Exp 12 confirmed this — full-power stacking reached 4.58.

**Sourcing strategy**: `user` (explicit strategy from conversation)
