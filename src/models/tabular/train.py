"""Train and score a tabular model on the feature table.

Input:  data/features/tabular.parquet (from tabular_features.py).
Output: one row on results/scoreboard.jsonl, and the fitted model in results/models/.

Steps, in the order run() does them:
    1. load      val pitches, and the split being scored; train pitches are read
                 a chunk at a time (a sample, if the model asks for one)
    2. fit       the model's own fit(), on train; XGBoost also watches val to
                 decide when to stop (logreg.py, xgb.py)
    3. score     its probabilities through evaluate()
    4. log       to the scoreboard, and save the model

Zone is trained in the batter's view, like the features: a left-handed batter's
zones are mirrored, so "inside" means the same thing for every batter. The
probabilities are mirrored back to Statcast's zones before scoring.

Usage:
    python src/models/tabular/train.py --model logreg --target pitch_type
    python src/models/tabular/train.py --model xgb --target pitch_type
    python src/models/tabular/train.py --model logreg --target zone --dry-run
    python src/models/tabular/train.py --model logreg --target zone --split test   # once, at the end
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation import evaluate, log_run  # noqa: E402
from features.schema import MIRROR_ZONE, PITCH_CLASSES, ZONE_CLASSES  # noqa: E402
from features.tabular_features import FEATURE_COLUMNS, OUT_PATH as FEATURES_PATH  # noqa: E402
from models.tabular import logreg, xgb  # noqa: E402

RESULTS = Path("results")
MODELS = {"logreg": logreg, "xgb": xgb}

# target -> label column and its classes
TARGETS = {
    "pitch_type": ("pitch_class", PITCH_CLASSES),
    "zone": ("zone_class", ZONE_CLASSES),
}


def run(model_name: str, target: str, split: str, dry_run: bool) -> None:
    """Train one model for one target, score it on ``split``, log and save it."""
    model = MODELS[model_name]
    label, classes = TARGETS[target]
    columns = [label, *FEATURE_COLUMNS]

    def train_chunks() -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
        """The train pitches as (features, labels) chunks; each call starts over, same sample."""
        for chunk in read_pitches("train", columns, model.TRAIN_FRACTION):
            yield chunk[FEATURE_COLUMNS], train_labels(chunk, target)

    val = pd.concat(read_pitches("val", columns), ignore_index=True)
    rows = val if split == "val" else pd.concat(read_pitches(split, columns), ignore_index=True)
    print(f"\n{model_name} / {target}: scoring {len(rows):,} {split} pitches")

    start = time.time()
    fitted, info = model.fit(train_chunks, (val[FEATURE_COLUMNS], train_labels(val, target)), len(classes))
    train_seconds = round(time.time() - start)

    proba = model.predict_proba(fitted, rows[FEATURE_COLUMNS]).astype(np.float64)
    proba /= proba.sum(axis=1, keepdims=True)  # float32 models; sum to 1 at float64
    if target == "zone":
        proba = to_statcast_zones(proba, rows["is_LHB"].to_numpy() == 1)
    metrics = evaluate(labels(rows[label]), proba, classes)
    print(f"  log_loss={metrics['log_loss']:.4f}  acc={metrics['accuracy']:.4f}  "
          f"top2={metrics['top2_accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
          f"(trained on {info['n_train']:,} pitches in {train_seconds}s)")

    if not dry_run:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = model.save(fitted, RESULTS / "models" / f"{stamp}_{model_name}_{target}")
        log_run(RESULTS / "scoreboard.jsonl", metrics, model=model_name, target=target, split=split,
                train_seconds=train_seconds, model_path=str(path), **info)
        print(f"  saved {path}")


# --- loading ------------------------------------------------------------------

def read_pitches(split: str, columns: list[str], fraction: float = 1.0, seed: int = 0) -> Iterator[pd.DataFrame]:
    """Eligible, labelled pitches of ``split``, a chunk at a time, optionally a random ``fraction``."""
    rng = np.random.default_rng(seed)
    label = columns[0]
    for batch in pq.ParquetFile(FEATURES_PATH).iter_batches(columns=["split", "eligible", *columns]):
        chunk = batch.to_pandas()
        keep = (chunk["split"] == split) & chunk["eligible"] & chunk[label].notna()
        if fraction < 1.0:
            keep &= rng.random(len(chunk)) < fraction
        if keep.any():
            yield chunk.loc[keep, columns]


# --- labels -------------------------------------------------------------------

def labels(values: pd.Series) -> np.ndarray:
    """Label column as plain values: pitch-class strings or zone ints."""
    return values.to_numpy(dtype=object if isinstance(values.dtype, pd.CategoricalDtype) else int)


def train_labels(rows: pd.DataFrame, target: str) -> np.ndarray:
    """Labels as class positions 0..k-1, with zones in the batter's view."""
    label, classes = TARGETS[target]
    values = pd.Series(labels(rows[label]))
    if target == "zone":
        lhb = rows["is_LHB"].to_numpy() == 1
        values = values.where(~lhb, values.map(MIRROR_ZONE))
    return values.map({c: i for i, c in enumerate(classes)}).to_numpy()


# For a left-handed batter, Statcast zone z is batter-view zone MIRROR_ZONE[z].
LHB_ZONE_ORDER = np.array([ZONE_CLASSES.index(MIRROR_ZONE[z]) for z in ZONE_CLASSES])


def to_statcast_zones(proba: np.ndarray, lhb: np.ndarray) -> np.ndarray:
    """Batter-view zone probabilities -> columns in Statcast's zone order."""
    proba = proba.copy()
    proba[lhb] = proba[lhb][:, LHB_ZONE_ORDER]
    return proba


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--target", choices=sorted(TARGETS), required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--dry-run", action="store_true", help="print the score without logging or saving")
    args = parser.parse_args()
    run(args.model, args.target, args.split, args.dry_run)


if __name__ == "__main__":
    main()
