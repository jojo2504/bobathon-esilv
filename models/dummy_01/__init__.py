"""Dummy baseline — DummyRegressor(strategy="mean")."""

from sklearn.dummy import DummyRegressor

KEY = "01_dummy"
model = DummyRegressor(strategy="mean")
