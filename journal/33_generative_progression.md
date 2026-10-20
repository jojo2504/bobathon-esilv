# Experiment 33 — Measurement calibration and patient progression

## New approach

This experiment starts fresh from the raw CSVs. It does not import model 32 and does not use any of its predictions as an input or ensemble component. Model 32's predictions are only used as a benchmark to compare performance.

Looking at the training data reveals three clear signals:

- The gap between the observed score and the target changes depending on the rater and the time since the last dose.
- ON measurements and LEDD are almost always missing when the target is low. The start and end of a patient's observations therefore carry information about their trajectory.
- A patient's visits follow a progression that is regular enough to jointly learn their level, slope, and curvature.

## Architecture

1. **Direct measurement model.** For ON and OFF separately, a Ridge regression learns `observed_score = target × scale(context) + offset(context)`. Context includes splines of the time delay, the rater, the dose, the cohort, and missingness indicators. Sample weights based on the target reduce the influence of high scores.
2. **Inversion and initial trajectory.** Inverting this relationship gives a severity estimate from each available measurement. ON and OFF observations are combined into a regularised quadratic curve, with weights based on the calibration scale factor and estimated noise.
3. **Visit features.** Build 228 features: visit data, aggregates, curve values, local means at three time scales, first and last observations, and all of the patient's measurements in order, up to 12 visits as in this dataset.
4. **Visit-level learning.** Two LightGBMs correct the errors of the initial trajectory. Their predictions are smoothed within each patient.
5. **Patient-level learning.** Three more LightGBMs predict correction coefficients for the curve: level, slope, and curvature. These coefficients are defined on an orthogonal time basis.
6. **Final ensemble.** Combine 35% of the first visit model, 35% of the second, and 30% of the patient model. Clip the score to `[0, 132]`.

Calibration and all supervised models are retrained within each fold. Validation patients' transformations use only their X observations and the parameters learned from training patients. All models are Ridge and LightGBM — no neural networks.

## Model selection

Exploration uses only the first fold of `GroupKFold(5)`:

| Candidate | RMSE on the exploration fold |
|---|---|
| Model 32, reference | 3.56730 |
| LightGBM 31 leaves, 2,600 trees | 3.41357 |
| LightGBM 63 leaves, 2,600 trees | 3.41796 |
| XGBoost depth 5, 2,600 trees | 3.44401 |
| LightGBM 15 leaves, extended training | 3.40006 |
| LightGBM 31 leaves, extended training | 3.39618 |
| Patient coefficient model | 3.52569 |
| **Chosen ensemble 35 / 35 / 30** | **3.35755** |

The patient model performs worse on its own but its errors are complementary to the visit models. XGBoost is not included in the final ensemble.

Hyperparameters and weights are fixed before validating on the other four folds. Fold 1 took part in selection, so its score is optimistic. Folds 2 to 5 provide a check on unseen patient groups. This is not nested cross-validation: patients in those groups also appeared in training during the exploratory model search. There is no fully held-out labelled test set here that is independent of model search.

Visits are reordered canonically before inference to stabilise the numerical reductions of patient features. A check verifies that permuting the input rows gives the same predictions.

## Chosen hyperparameters

| Component | Trees | Leaves | Learning rate | L2 |
|---|---|---|---|---|
| LightGBM visit A | 6,499 | 15 | 0.025 | 8 |
| LightGBM visit B | 6,497 | 31 | 0.025 | 8 |
| LightGBM patient: level | 2,998 | 15 | 0.02 | 10 |
| LightGBM patient: slope | 2,333 | 15 | 0.02 | 10 |
| LightGBM patient: curvature | 2,067 | 15 | 0.02 | 10 |

Tree counts come from early-stopping runs on the exploration fold. The full validation uses these fixed counts without consulting the labels of other folds to stop training.

## Full validation results

| Fold | Model 32 | Model 33 |
|---|---|---|
| 1 — used for exploration | 3.56730 | 3.35490 |
| 2 | 3.59380 | 3.33285 |
| 3 | 3.46349 | 3.38568 |
| 4 | 3.47063 | 3.33698 |
| 5 | 3.64308 | 3.34332 |
| **Overall pooled OOF RMSE** | **3.54836** | **3.35080** |

The gain is positive on all five folds: **5.57% reduction in overall RMSE**. On folds 2 to 5 only, RMSE goes from 3.54361 to approximately 3.34977. A paired bootstrap with 2,000 resamples on those four folds gives a 95% confidence interval for the absolute RMSE gain of **[0.137, 0.250]**. This interval is conditional on the chosen model and does not account for the uncertainty introduced by the full hyperparameter search.

Inference checks verify that row reordering and processing complete patients separately preserve the predictions.

The final model was trained on 44,590 visits. The 11,013 predictions in the CSV were checked: format, unique indices, expected order, finite values, and scores in `[0, 132]`.

**Skore publication confirmed:** [33_generative_progression](https://skore.probabl.ai/degensunite/bobathon-esilv/cross-validations/87766).

## How to reproduce

```sh
.venv/bin/python -u experiments/33_generative_progression.py
.venv/bin/python -u experiments/33_generative_progression.py --push
```

The script saves the Skore report, the final model, the OOF predictions, and the metrics under the prefix `scratch/33_generative_progression`. The final CSV is `submissions/33_generative_progression.csv`. Intermediate models are cached based on fold data and seed. After changing the algorithm, update the cache version and move any old artefacts before re-running.

The protocol assumes that all available visits for a patient are provided together, including visits later in their trajectory. This matches the competition setup and does not measure a strictly past-only forecast. The required publication is on Skore.
