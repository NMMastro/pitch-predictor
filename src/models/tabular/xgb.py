"""XGBoost: the main tabular model.

Trains on a random TRAIN_FRACTION of the train pitches: all 5.4M swap heavily
on an 8 GB machine. The table reaches XGBoost a chunk at a time, so only its
compressed form (about 1 byte per value) is ever in memory, not rows of floats. Trees are added until val log loss has not improved for
EARLY_STOPPING_ROUNDS rounds; predictions use the trees up to the best round.

Category features go in as their integer codes, and missing values as missing:
XGBoost learns which way to send them at each split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import xgboost

TRAIN_FRACTION = 0.3  # about 1.6M of the 5.4M train pitches; the full set swaps on 8 GB
PARAMS = {
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "max_depth": 8,
    "eta": 0.1,               # learning rate
    "min_child_weight": 10,
    "subsample": 0.8,         # share of pitches each tree sees
    "colsample_bytree": 0.8,  # share of features each tree sees
    "max_bin": 256,
    "seed": 0,
}
MAX_ROUNDS = 3000
EARLY_STOPPING_ROUNDS = 50


def fit(train_chunks: Callable[[], Iterator], val: tuple, n_classes: int) -> tuple[xgboost.Booster, dict]:
    """Fit on the train chunks, stopping early on ``val``; labels are class positions 0..n_classes-1."""
    dtrain = xgboost.QuantileDMatrix(Chunks(train_chunks), max_bin=PARAMS["max_bin"])
    X_val, y_val = val
    dval = xgboost.QuantileDMatrix(X_val, label=y_val, ref=dtrain)
    booster = xgboost.train(
        {**PARAMS, "num_class": n_classes}, dtrain,
        num_boost_round=MAX_ROUNDS,
        evals=[(dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=50,
    )
    return booster, {"n_train": dtrain.num_row(), "best_iteration": booster.best_iteration}


def predict_proba(booster: xgboost.Booster, X: pd.DataFrame) -> np.ndarray:
    """Class probabilities from the trees up to the best round, columns in class-position order."""
    return booster.predict(xgboost.DMatrix(X), iteration_range=(0, booster.best_iteration + 1))


def save(booster: xgboost.Booster, path: Path) -> Path:
    path = path.with_suffix(".ubj")
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(path)
    return path


class Chunks(xgboost.DataIter):
    """Hands XGBoost one (features, labels) chunk at a time. XGBoost may go through them more than once."""

    def __init__(self, train_chunks: Callable[[], Iterator]):
        self.train_chunks = train_chunks
        self.chunks = None
        super().__init__()

    def next(self, input_data: Callable) -> bool:
        if self.chunks is None:
            self.chunks = self.train_chunks()
        chunk = next(self.chunks, None)
        if chunk is None:
            return False
        X, y = chunk
        input_data(data=X, label=y)
        return True

    def reset(self) -> None:
        self.chunks = None
