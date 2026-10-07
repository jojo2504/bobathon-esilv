"""Dummy baseline — end-to-end test of the full submission pipeline.

This script exercises every piece of the API:
  1. load_data()          — reads X_train / y_train / X_test from data/
  2. Experiment.run()     — grouped-CV evaluation, prints RMSE
  3. Experiment.push()    — pushes report to Skore Hub, prints URL
  4. make_submission()    — fits on all train, writes submissions/01_dummy.csv

Run it to confirm the whole chain works before building real models:

    python main.py

The resulting submissions/01_dummy.csv is a valid Kaggle upload.
Paste the Hub URL printed above into the Kaggle Submission Description.
"""

from sklearn.dummy import DummyRegressor

from parkinson.data import load_data
from parkinson.evaluate import Experiment
from parkinson.submission import make_submission

KEY = "01_dummy"

# ── 1. Data ──────────────────────────────────────────────────────────────────
print("Loading data...")
ds = load_data()
print(f"  train: {ds.X.shape}  test: {ds.X_test.shape}")

# ── 2. Model ─────────────────────────────────────────────────────────────────
model = DummyRegressor(strategy="mean")

# ── 3. Evaluate (GroupKFold, 5 folds) ────────────────────────────────────────
exp = Experiment(KEY, model)
report = exp.run(ds)

# ── 4. Push to Skore Hub ─────────────────────────────────────────────────────
# Prints: Consult your report at https://skore.probabl.ai/…
# Copy that URL → paste into Kaggle Submission Description.
exp.push(report)

# ── 5. Write submission CSV ───────────────────────────────────────────────────
# submissions/01_dummy.csv  →  upload to Kaggle
make_submission(KEY, model, ds)

print("\nDone. Upload submissions/01_dummy.csv to Kaggle and paste the Hub URL above.")
