"""Multinomial logistic regression: the linear reference point for XGBoost.

Inputs, prepared in this order:
  mixes      every p_type_*, p_zone_*, b_type_*, b_zone_* share as its log: the
             model adds up its inputs in log-probability space, so a weight of 1
             on log(share) can rebuild a mix exactly, where the raw share can't
  numbers    missing values filled with the median (plus a column saying which
             were missing), then standardised
  categories one-hot, with missing as its own category

A linear model gains little from millions of rows, so it trains on a random
TRAIN_FRACTION of the train pitches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features.tabular_features import CATEGORICAL_FEATURES

TRAIN_FRACTION = 0.2  # about 1M of the 5.4M train pitches
C = 1.0               # inverse regularisation strength
MAX_ITER = 1000
MIX_PREFIXES = ("p_type_", "p_zone_", "b_type_", "b_zone_")
MIN_SHARE = 1e-6      # floor before the log; shrinkage keeps shares above 0, float32 may not


def fit(train_chunks: Callable[[], Iterator], val: tuple, n_classes: int) -> tuple[Pipeline, dict]:
    """Fit on the train chunks (val is not used); labels are class positions 0..n_classes-1."""
    chunks = list(train_chunks())
    X = pd.concat([X for X, _ in chunks], ignore_index=True)
    y = np.concatenate([y for _, y in chunks])
    model = make_pipeline(
        ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
             numeric_columns(X)),
            ("category", OneHotEncoder(handle_unknown="ignore", dtype=np.float32), CATEGORICAL_FEATURES),
        ]),
        LogisticRegression(C=C, max_iter=MAX_ITER),
    )
    model.fit(prepare(X), y)
    if list(model.classes_) != list(range(n_classes)):
        raise ValueError(f"train pitches cover {len(model.classes_)} of {n_classes} classes")
    return model, {"n_train": len(y)}


def predict_proba(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Class probabilities, columns in class-position order."""
    return model.predict_proba(prepare(X))


def save(model: Pipeline, path: Path) -> Path:
    path = path.with_suffix(".joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def numeric_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in CATEGORICAL_FEATURES]


def prepare(X: pd.DataFrame) -> pd.DataFrame:
    """Every column as float32 with NaN for missing; mixes as logs; categories get -1 for missing."""
    X = X.astype("float32")
    mixes = [c for c in X.columns if c.startswith(MIX_PREFIXES)]
    X[mixes] = np.log(X[mixes].clip(lower=MIN_SHARE))
    X[CATEGORICAL_FEATURES] = X[CATEGORICAL_FEATURES].fillna(-1)
    return X
