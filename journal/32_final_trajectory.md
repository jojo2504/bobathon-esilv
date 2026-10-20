# Experiment 32 — Tree ensemble with patient trajectories

## Idea

Predict the true OFF score by combining each visit's data with the progression of the same patient's observations over time. Linear and quadratic trajectories help capture how the disease evolves with age and reduce the noise in individual ON/OFF measurements.

The final model uses LightGBM and XGBoost. It does not depend on any prediction from a previous model and contains no neural network.

## Strategy

1. **Correct the ON/OFF measurements.** Two small LightGBMs learn `target - on` and `target - off` respectively, on visits where the score is available. They use visit-level features including `rater_id`, treatment dose, and time-since-intake delays. Missing scores stay missing after correction.
2. **Build 121 features.** Add patient-level aggregates, deviations from the patient mean, visit rank, regularised trajectories of scores and LEDD, neighbouring observations, and their time distances. Pharmacokinetic features use `exp(-k × delay)` for `k ∈ {0.2, 0.5, 1.0}`, alone and multiplied by LEDD.
3. **Blend two models.** Average the LightGBM and XGBoost predictions with equal weights.
4. **Smooth by patient.** Mix 85% of a quadratic curve fitted to the predictions with 15% of the raw predictions, then clip the score to `[0, 132]`.
5. **Validate and publish.** Evaluate with `skore.evaluate` on five patient groups, retrain on all data, and publish the report to Skore.

## Hyperparameters

| Parameter | LightGBM final | XGBoost final |
|---|---|---|
| Number of trees | 2,200 | 1,760 |
| Learning rate | 0.025 | 0.035 |
| Capacity | 63 leaves | Depth 5 |
| Min samples per leaf | 25 | `min_child_weight=15` |
| Row subsampling | 0.85, each iteration | 0.85 |
| Column subsampling | 0.85 | 0.85 |
| L1 / L2 regularisation | 0.05 / 3 | 0.05 / 10 |
| Seed | 2026 | 2026 |

Each correction model uses 350 trees, 15 leaves, a learning rate of 0.035, at least 100 samples per leaf, and L2 regularisation of 20.

## Validation

`GroupKFold(5)` splits by patient: no visit from a validation patient is used to train models in its fold. The supervised corrections are also learned only on training patients. The train/test patient split is verified.

Patient aggregates and trajectories for a validation or test patient are computed from that patient's own X observations only, without their labels. Inference therefore assumes that all available visits for a patient are provided together, as in the competition files — this is not a strictly past-only forecast.

Fold 1 was used to compare smoothing strengths of 0, 0.5, 0.85, and 1. The overall result therefore includes this selection on one fold and is not a score on a set fully independent of model choice.

## Results

| Measure | RMSE |
|---|---|
| Fold 1 | 3.56730 |
| Fold 2 | 3.59380 |
| Fold 3 | 3.46349 |
| Fold 4 | 3.47063 |
| Fold 5 | 3.64308 |
| **Overall pooled OOF** | **3.54836** |
| Previous T17 predictions (reference) | 3.93637 |

The relative improvement is **9.86%** over the saved T17 predictions. The overall RMSE is computed over all OOF visits pooled together, not as an average of the five fold RMSEs. No Kaggle score was measured for this experiment.

The final model is trained on **44,590 visits** and produces **11,013 predictions**. The CSV is verified: unique indices, same order as the sample submission, no missing or infinite values.

Published report: [32_final_trajectory on Skore](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/86150).

## Generated files

| File | Content |
|---|---|
| `experiments/32_final_trajectory.py` | Estimator, validation, full-data training, and publishing |
| `scratch/32_final_trajectory_report.joblib` | Skore report for the five real folds |
| `scratch/32_final_trajectory_model.joblib` | Ensemble trained on all data |
| `scratch/32_final_trajectory_oof.csv` | Validation predictions aligned on `Index` |
| `scratch/32_final_trajectory_metrics.json` | Scores, data sizes, and Skore report ID |
| `submissions/32_final_trajectory.csv` | Submission in `Index,target` format |

## How to run

From the repo root:

```sh
# Compare smoothing strengths on the first fold
.venv/bin/python -u experiments/32_final_trajectory.py --probe

# Validate, train the final model, and generate the CSV
.venv/bin/python -u experiments/32_final_trajectory.py

# Publish the report to Skore
.venv/bin/python -u experiments/32_final_trajectory.py --push
```

Later runs reuse the existing `.joblib` files. After changing the data or the model, move or delete those two files explicitly before re-running to force a full retrain. The `--push` command publishes to Skore; it does not submit the CSV to Kaggle.
