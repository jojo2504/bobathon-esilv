"""Entry point — run a model by name.

Usage:
    python main.py --model dummy_01
    python main.py --model yes_02

Each model lives in models/<name>/__init__.py and must expose:
    KEY   : str           — unique submission key, e.g. "01_dummy"
    model : sklearn estimator
"""

import argparse
import importlib

from src.parkinson.data import load_data
from src.parkinson.evaluate import Experiment
from src.parkinson.submission import make_submission

parser = argparse.ArgumentParser(description="Train and submit a model.")
parser.add_argument("--model", required=True, help="Model folder name under models/")
args = parser.parse_args()

# ── Load model module ─────────────────────────────────────────────────────────
mod = importlib.import_module(f"models.{args.model}")
KEY = mod.KEY
model = mod.model

# ── 1. Data ───────────────────────────────────────────────────────────────────
print("Loading data...")
ds = load_data()
print(f"  train: {ds.X.shape}  test: {ds.X_test.shape}")

# ── 2. Evaluate (GroupKFold, 5 folds) ─────────────────────────────────────────
exp = Experiment(KEY, model)
report = exp.run(ds)

# ── 3. Push to Skore Hub ──────────────────────────────────────────────────────
exp.push(report)

# ── 4. Write submission CSV ───────────────────────────────────────────────────
make_submission(KEY, model, ds)

print(f"\nDone. Upload submissions/{KEY}.csv to Kaggle and paste the Hub URL above.")
