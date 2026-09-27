"""Shared scoring for both pitch-prediction models.

Every model reports through ``evaluate``, which takes predicted probabilities
and true labels and knows nothing about the model that produced them. 

Example:
-------
    from evaluation import align_proba, evaluate, log_run

    proba = align_proba(clf.predict_proba(X_test), clf.classes_)
    metrics = evaluate(y_test, proba)
    log_run("results/scoreboard.jsonl", metrics, model="lgbm", target="pitch_type")

"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)

PITCH_CLASSES: list[str] = [
    "FF", "SI", "SL", "CH", "FC", "ST", "CU", "FS", "KC", "SV", "OTHER",
]
OTHER_CODES: frozenset[str] = frozenset({"EP", "FO", "CS", "KN", "SC"})
DROP_CODES: frozenset[str] = frozenset({"PO", "IN", "UN", "FA"})
ZONE_CLASSES: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]
KEEP_GAME_TYPES: frozenset[str] = frozenset({"R", "F", "D", "L", "W"})


def map_pitch_classes(pitch_type: pd.Series) -> pd.Series:
    """Map raw Statcast ``pitch_type`` codes onto :data:`PITCH_CLASSES`.

    Rare codes collapse to ``OTHER``; non-pitch codes and nulls become NaN.
    """
    s = pitch_type.astype("object")
    out = s.where(~s.isin(OTHER_CODES), "OTHER").where(~s.isin(DROP_CODES), np.nan)
    unknown = set(out.dropna().unique()) - set(PITCH_CLASSES)
    if unknown:
        raise ValueError(
            f"pitch_type codes not covered by the taxonomy: {sorted(unknown)}. "
            "Add them to PITCH_CLASSES, OTHER_CODES or DROP_CODES."
        )
    return out


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the design-doc row filters and attach target columns.

    Drops spring training, null ``pitch_type`` and the non-pitch codes, then
    adds ``pitch_class``. If a ``zone`` column is present, adds ``zone_class``
    as a nullable integer; callers training the zone model should drop its
    nulls themselves.

    The 25-prior-pitch rule is not applied here: it needs pitch history, so it
    belongs in the feature pipeline after the running averages are computed.
    """
    out = df[df["game_type"].isin(KEEP_GAME_TYPES)].copy()
    out["pitch_class"] = map_pitch_classes(out["pitch_type"])
    out = out.dropna(subset=["pitch_class"])
    if "zone" in out.columns:
        out["zone_class"] = out["zone"].astype("Int64")
    return out


def align_proba(
    proba: np.ndarray, model_classes: Sequence, classes: Sequence = PITCH_CLASSES
) -> np.ndarray:
    """Reorder a model's probability columns into ``classes`` order.

    Parameters
    ----------
    proba:
        The model's probability array, shape ``(n_samples, n_classes)``.
    model_classes:
        What the model's columns currently mean, in order. Usually
        ``clf.classes_``. For a raw Booster or a PyTorch model there is no such
        attribute, so pass your own label-encoding order.
    classes:
        What the columns should mean, in order. :data:`PITCH_CLASSES` by
        default; pass :data:`ZONE_CLASSES` for the zone model.

    Returns
    -------
    np.ndarray
        ``proba`` with columns reordered from ``model_classes`` into
        ``classes`` order.
    """
    model_classes = list(model_classes)
    if set(model_classes) != set(classes):
        raise ValueError(
            f"model_classes and classes describe different label sets; "
            f"missing from model: {sorted(set(classes) - set(model_classes))}, "
            f"unexpected: {sorted(set(model_classes) - set(classes))}"
        )
    proba = np.asarray(proba, dtype=float)
    if proba.shape[1] != len(classes):
        raise ValueError(f"proba has {proba.shape[1]} columns, expected {len(classes)}")
    pos = {c: i for i, c in enumerate(model_classes)}
    return proba[:, [pos[c] for c in classes]]


def _validate(y_true: np.ndarray, proba: np.ndarray, classes: Sequence) -> None:
    if proba.ndim != 2:
        raise ValueError(f"proba must be 2-D (n_samples, n_classes), got shape {proba.shape}")
    if proba.shape[0] != len(y_true):
        raise ValueError(f"proba has {proba.shape[0]} rows but y_true has {len(y_true)}")
    if proba.shape[1] != len(classes):
        raise ValueError(
            f"proba has {proba.shape[1]} columns but {len(classes)} classes were given; "
            "columns must be in `classes` order (see align_proba)"
        )
    if not np.isfinite(proba).all():
        raise ValueError("proba contains NaN or inf")
    if (proba < 0).any():
        raise ValueError(
            "proba contains negative values; pass probabilities, not logits "
            "(apply a softmax first)"
        )
    sums = proba.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        bad = int(np.argmax(np.abs(sums - 1.0)))
        raise ValueError(
            f"proba rows must sum to 1; row {bad} sums to {sums[bad]:.6f}. "
            "Logits and decision-function scores need a softmax first."
        )


def evaluate(
    y_true: Iterable,
    proba: np.ndarray,
    classes: Sequence = PITCH_CLASSES,
    *,
    per_class: bool = True,
) -> dict[str, Any]:
    """Score predicted probabilities against the fixed metric set.

    Parameters
    ----------
    y_true:
        True labels. Assumed already mapped onto ``classes`` by preprocessing.
        For sequence models, flatten across timesteps and drop padded
        positions first.
    proba:
        ``(n_samples, n_classes)`` probabilities with columns in ``classes``
        order. Use :func:`align_proba` if the model reports another order.
    classes:
        :data:`PITCH_CLASSES`, :data:`ZONE_CLASSES`, or any label list.
    per_class:
        Include per-class F1 and support.
    """
    y_true = np.asarray(list(y_true))
    proba = np.asarray(proba, dtype=float)
    classes = list(classes)
    _validate(y_true, proba, classes)

    class_arr = np.asarray(classes)
    y_pred = class_arr[proba.argmax(axis=1)]
    kw = {"labels": classes, "average": "macro", "zero_division": 0}

    # sklearn's probability metrics assume proba columns are in sorted class
    # order whatever is passed as `labels`, so permute to match.
    order = sorted(range(len(classes)), key=lambda i: classes[i])
    sorted_classes = [classes[i] for i in order]
    proba_sorted = proba[:, order]

    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "log_loss": float(log_loss(y_true, proba_sorted, labels=sorted_classes)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "top2_accuracy": float(
            top_k_accuracy_score(y_true, proba_sorted, k=2, labels=sorted_classes)
        ),
        "macro_precision": float(precision_score(y_true, y_pred, **kw)),
        "macro_recall": float(recall_score(y_true, y_pred, **kw)),
        "macro_f1": float(f1_score(y_true, y_pred, **kw)),
    }

    support = pd.Series(y_true).value_counts().reindex(classes, fill_value=0)
    missing = [c for c in classes if support[c] == 0]
    if missing:
        out["zero_support_classes"] = [_json_safe(c) for c in missing]
    if per_class:
        f1s = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
        out["per_class_f1"] = {str(c): float(v) for c, v in zip(classes, f1s)}
        out["support"] = {str(c): int(support[c]) for c in classes}
    return out


def _json_safe(v: Any) -> Any:
    return v.item() if isinstance(v, np.generic) else v


def log_run(path: str | Path, metrics: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """Append one run to a JSONL scoreboard and return the row written.

    ``fields`` identifies the run: at minimum ``model``, ``target`` and
    ``features_version``. The last matters most -- when the feature table
    changes, earlier rows stop being comparable, and the stamp is the only way
    to tell which.
    """
    row = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields, **metrics}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, default=_json_safe) + "\n")
    return row


def load_scoreboard(path: str | Path) -> pd.DataFrame:
    """Read a JSONL scoreboard back as a DataFrame, newest run last."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in path.read_text().splitlines() if l.strip()])
