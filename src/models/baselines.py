"""Baselines: simple predictions every real model has to beat.

Input:  data/features/tabular.parquet (from tabular_features.py).
Output: one row per baseline and target, appended to results/scoreboard.jsonl.

Nothing is trained. Each baseline gives every class a probability, so its log
loss is a fair score:
  league            share of each class over eligible train pitches, the same for every pitch
  pitcher_career    the pitcher's mix over all his earlier pitches
  pitcher_last200   his mix over his last 200 pitches
  pitcher_vs_hand   his mix against batters of this batter's hand
  pitcher_count     his mix in this ball-strike count

The pitcher mixes are the feature table's p_type_* / p_zone_* columns: built from
earlier pitches only and shrunk toward the league, so every pitch has one.

Usage:
    python src/models/baselines.py                # score on val
    python src/models/baselines.py --split test   # once, at the end
    python src/models/baselines.py --dry-run      # print, don't log
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import evaluate, log_run  # noqa: E402
from features.schema import MIRROR_ZONE, PITCH_CLASSES, ZONE_CLASSES, ZONE_LABELS  # noqa: E402
from features.tabular_features import OUT_PATH as FEATURES_PATH  # noqa: E402

SCOREBOARD = Path("results/scoreboard.jsonl")

# Each pitcher baseline and the slice of his mix it reads.
PITCHER_SLICES = {
    "pitcher_career": "career",
    "pitcher_last200": "last200",
    "pitcher_vs_hand": "vs_hand",
    "pitcher_count": "count",
}

# target -> label column and its classes
TARGETS = {
    "pitch_type": ("pitch_class", PITCH_CLASSES),
    "zone": ("zone_class", ZONE_CLASSES),
}


def run(split: str, dry_run: bool) -> None:
    """Score every baseline on ``split`` for both targets."""
    for target, (label, classes) in TARGETS.items():
        mix_cols = [c for s in PITCHER_SLICES.values() for c in mix_columns(target, s)]
        rows = pd.read_parquet(FEATURES_PATH, columns=[label, "is_LHB", *mix_cols],
                               filters=[("split", "==", split), ("eligible", "==", True)]).dropna(subset=[label])
        train = pd.read_parquet(FEATURES_PATH, columns=[label],
                                filters=[("split", "==", "train"), ("eligible", "==", True)]).dropna()
        y = labels(rows[label])
        print(f"\n{target} on {split}: {len(rows):,} pitches")

        predictions = {"league": np.tile(league_mix(labels(train[label]), classes), (len(rows), 1))}
        for name, slice_name in PITCHER_SLICES.items():
            predictions[name] = pitcher_mix(rows, target, slice_name)

        for name, proba in predictions.items():
            metrics = evaluate(y, proba, classes)
            print(f"  {name:<16} log_loss={metrics['log_loss']:.4f}  acc={metrics['accuracy']:.4f}  "
                  f"top2={metrics['top2_accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}")
            if not dry_run:
                log_run(SCOREBOARD, metrics, model=f"baseline_{name}", target=target, split=split)


def league_mix(y_train: np.ndarray, classes: list) -> np.ndarray:
    """Share of each class in ``y_train``, in ``classes`` order."""
    counts = pd.Series(y_train).value_counts().reindex(classes, fill_value=0).to_numpy(float)
    return counts / counts.sum()


def pitcher_mix(rows: pd.DataFrame, target: str, slice_name: str) -> np.ndarray:
    """The pitcher's mix for one slice, as probabilities in the target's class order."""
    proba = rows[mix_columns(target, slice_name)].to_numpy(np.float64)
    proba = proba / proba.sum(axis=1, keepdims=True)  # stored as float32; sum to 1 at float64
    if target == "zone":
        # The table's zones are the batter's view; mirror left-handed batters
        # back so columns line up with zone_class (Statcast's zones).
        lhb = rows["is_LHB"].to_numpy() == 1
        proba[lhb] = proba[lhb][:, LHB_ZONE_ORDER]
    return proba


def mix_columns(target: str, slice_name: str) -> list[str]:
    """Feature-table columns holding one slice of the pitcher's mix, in class order."""
    if target == "pitch_type":
        return [f"p_type_{slice_name}_{c}" for c in PITCH_CLASSES]
    return [f"p_zone_{slice_name}_{z}" for z in ZONE_LABELS]


# For a left-handed batter, Statcast zone z is batter-view zone MIRROR_ZONE[z].
LHB_ZONE_ORDER = np.array([ZONE_CLASSES.index(MIRROR_ZONE[z]) for z in ZONE_CLASSES])


def labels(values: pd.Series) -> np.ndarray:
    """Label column as plain values: pitch-class strings or zone ints."""
    return values.to_numpy(dtype=object if isinstance(values.dtype, pd.CategoricalDtype) else int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--dry-run", action="store_true", help="print scores without logging them")
    args = parser.parse_args()
    run(args.split, args.dry_run)


if __name__ == "__main__":
    main()
