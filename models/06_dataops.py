# %% [markdown]
# # Experiment 06 — skrub DataOps: groups baked into the graph
#
# Step 6 of the Day 2 guide:
#   - Same model as step 5 (TableVectorizer + HGBR) but expressed as a DataOps graph
#   - GroupKFold + groups are attached to the X node — no side list that can drift
#   - SkrubLearner + evaluate(learner, data={"visits": visits}) reads cv/groups from graph
#   - Push "06_dataops"

# %%
from pathlib import Path
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from skrub import TableVectorizer
from skore import evaluate

from parkinson.data import load_data, FEATURE_COLS_FULL
from parkinson.hub import get_project

# ── Data ──────────────────────────────────────────────────────────────────────
print("Loading data...")
ds = load_data(feature_cols=FEATURE_COLS_FULL)
visits = ds.visits   # full merged training table (features + target)
X_test_raw = ds.X_test_raw
print(f"  visits: {visits.shape}")

# ── Build the DataOps graph ───────────────────────────────────────────────────
print("\n[Step 6] Building skrub DataOps graph")
data = skrub.var("visits", visits)

# groups live on the graph — no separate cv_splits list can drift
groups_op = data["patient_id"]

# X: drop identifier columns and the target; attach GroupKFold + groups
X_op = (
    data.drop(["Index", "patient_id", "target"], axis=1)
    .skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": groups_op},
    )
)
y_op = data["target"].skb.mark_as_y()

# Apply the same pipeline as step 5
pred = (
    X_op.skb.apply(TableVectorizer())
    .skb.apply(HistGradientBoostingRegressor(random_state=0), y=y_op)
)

# ── Freeze into a SkrubLearner and evaluate ───────────────────────────────────
print("  Making learner from graph...")
learner = pred.skb.make_learner()

print("  Evaluating (GroupKFold baked into graph)...")
# splitter is omitted → skore reads GroupKFold + groups from the DataOp
rep = evaluate(learner, data={"visits": visits})
rmse = float(rep.metrics.rmse().iloc[:, 0].mean())
print(f"  dataops RMSE (grouped CV) = {rmse:.4f}")
print(f"  step-5 tabular was        = 7.5238")
print(f"  delta                     = {rmse - 7.5238:+.4f}")

# ── Push to Hub ───────────────────────────────────────────────────────────────
print("\n[Push] DataOps report → Hub key '06_dataops'")
project = get_project()
project.put("06_dataops", rep)

# ── Submission CSV via SkrubLearner ───────────────────────────────────────────
print("\n[Submission] Fitting learner on all training data...")
learner.fit({"visits": visits})

# The graph drops "target" — add a placeholder column so the graph can drop it
# without error; it is never used in prediction (the estimator step ignores y).
X_test_for_learner = X_test_raw.copy()
X_test_for_learner["target"] = float("nan")

pred_test = learner.predict({"visits": X_test_for_learner})

submission = X_test_raw[["Index"]].copy()
submission["target"] = pred_test
Path("submissions").mkdir(exist_ok=True)
submission.to_csv("submissions/06_dataops.csv", index=False)
print(f"Wrote submissions/06_dataops.csv  ({len(submission)} rows)")
print("\nDone. Upload submissions/06_dataops.csv to Kaggle with the Hub URL above.")
