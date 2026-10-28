"""Sequence features: clean pitch table -> transformer inputs, kept in memory.

Input:  data/clean/pitches.parquet (from clean.py), optionally one pitcher.
Output: nothing on disk. build() returns the encoded, scaled table and its
        scaler; build_sequences() turns that table into the arrays below.
        Row i of each array is row i of the table, which holds the labels
        (pitch_class, zone_class) and the split / eligible columns.

Every input for a pitch uses only its pre-pitch state and earlier pitches.
History never resets at plate appearances, games, seasons or splits, so every
split stays in one table.

Steps, in the order they run:

  build()
    1. load              the clean table, optionally one pitcher (the v1 scope)
    2. encode            every pitch as numbers (encode.py)
    3. encode_ids        embedding ids -> contiguous codes, rare batters -> UNK
    4. scale             standardise CONTINUOUS_INPUTS, fit on train rows only

  build_sequences()
    5. X_seq, X_ctx      the arrays below

Arrays
------
  X_seq  (n, WINDOW, len(PREV_PITCH_LAGGED))
         the pitcher's previous WINDOW pitches, oldest first, padded with UNK
  X_ctx  (n, len(MODEL_INPUTS))
         the current pitch: pre-pitch state, handedness and inning-half flags,
         and the embedding ids
  cols   names of X_seq's last axis (PREV_PITCH_LAGGED)

Defaults for two open decisions (DESIGN_DECISIONS.md Section 6):
  WINDOW = 5             previous pitches in each sequence
  MIN_BATTER_COUNT = 5   batters seen fewer times become UNK

Usage:
    python src/features/sequence_features.py               # Scherzer
    python src/features/sequence_features.py --pitcher 0   # every pitcher
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.encode import encode  # noqa: E402
from features.schema import (  # noqa: E402
    CLEAN_PATH,
    ENCODED_PRE_PITCH_COLUMNS,
    PITCHER_CHRONO_KEYS,
)

# ==============================================================================
# 1. MODEL INPUTS (DESIGN_DECISIONS.md Section 6)
# ==============================================================================

# --- 6a. Current-pitch inputs: known before the pitch, fed directly ---
INPUT_EMBED = ["batter", "pitcher", "fielder_2"]  # fielder_2 = catcher

# --- 6b. Lagged previous-pitch features (described for PRIOR pitches only) ---
PREV_PITCH_LAGGED = [
    "pitch_class", "release_speed", "plate_x", "plate_z", "zone_class", "description",
]
# Which of the lagged columns are categorical (encoded to int codes, not scaled).
LAGGED_CATEGORICAL = ["pitch_class", "zone_class", "description"]

# The current-pitch (6a) columns in X_ctx, in order.
MODEL_INPUTS = ENCODED_PRE_PITCH_COLUMNS + INPUT_EMBED

# Continuous 6a inputs that get scaled. Counts/ordinals and booleans are left
# alone; ids are embedded, not scaled.
CONTINUOUS_INPUTS = [
    "bat_score", "fld_score", "home_score_diff", "bat_score_diff",
    "pitcher_days_since_prev_game", "batter_days_since_prev_game",
    "age_pit", "age_bat", "sz_top", "sz_bot",
]

# Open-decision defaults (DESIGN_DECISIONS.md Section 6 "Open decisions").
WINDOW = 5            # previous-pitch window length for the 6b lags
MIN_BATTER_COUNT = 5  # batters seen fewer times than this collapse to UNK
UNK = -1             # reserved code for the UNK / padding bucket


# ==============================================================================
# 2. LOAD
# ==============================================================================

def load(path: str | Path = CLEAN_PATH, pitcher: int | None = None) -> pd.DataFrame:
    """The clean table, optionally one pitcher's rows."""
    filters = [("pitcher", "==", pitcher)] if pitcher is not None else None
    return pd.read_parquet(path, filters=filters)


# ==============================================================================
# 3. EMBEDDING IDS (stateless, no train/test leakage)
# ==============================================================================

def encode_ids(df: pd.DataFrame, min_batter_count: int = MIN_BATTER_COUNT) -> pd.DataFrame:
    """Embedding ids -> contiguous codes with an UNK bucket (Open Decision #3).

    Stateless (no fitting). The rest of the encoding is shared and done by
    encode.py; scaling is NOT done here — it is stateful and must be fit on
    train only.
    """
    for col in INPUT_EMBED:
        df = _encode_id(df, col, min_batter_count if col == "batter" else 1)
    return df


def _encode_id(df: pd.DataFrame, col: str, min_count: int) -> pd.DataFrame:
    """Map an id column to contiguous int codes, rare ids -> UNK.

    Rarity is judged on the frame passed in. For a leakage-clean pipeline this
    should be the full dataset's vocabulary decided once and reused; for v1
    single-pitcher data it is computed here. Codes start at 0; UNK is -1.
    """
    counts = df[col].value_counts()
    keep = set(counts[counts >= min_count].index)
    codes = {v: i for i, v in enumerate(sorted(keep))}
    df[col] = df[col].map(lambda v: codes.get(v, UNK)).astype("int64")
    return df


# ==============================================================================
# 4. SCALE (fit on train rows only)
# ==============================================================================

def fit_scaler(train: pd.DataFrame, cols: list[str] = CONTINUOUS_INPUTS) -> StandardScaler:
    """Fit a scaler on TRAIN ONLY (the fix for scaler leakage)."""
    scaler = StandardScaler().fit(train[cols])
    scaler.feature_names_in_ = np.array(cols)  # remember the column order
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a pre-fit scaler to every row. Does not fit."""
    cols = list(scaler.feature_names_in_)
    out = df.copy()
    out[cols] = scaler.transform(out[cols])
    return out


# ==============================================================================
# 5. SEQUENCES
# ==============================================================================

def _as_numbers(values: pd.Series) -> pd.Series:
    """Categoricals -> their codes (-1 = missing); other columns unchanged."""
    return values.cat.codes if isinstance(values.dtype, pd.CategoricalDtype) else values


def build_sequences(
    df: pd.DataFrame, window: int = WINDOW
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build ``(n, window, n_feat)`` previous-pitch tensors for the transformer.

    For each pitch t, stacks the feature vectors of pitches t-window .. t-1 (same
    pitcher, real time order) into a window. Positions with no prior pitch (start
    of a pitcher's history) are padded with UNK/0. Returns ``(X_seq, X_ctx, cols)``
    where X_seq is the lagged sequence, X_ctx is the current-pitch 6a inputs, and
    cols names the per-step sequence features.

    Note: this returns the SEQUENCE and the current-pitch CONTEXT separately,
    because the transformer consumes them on different paths (attention over the
    sequence; the 6a context either prepended as a token or concatenated at the
    head). The label is not returned here — callers pair with the frame's
    ``pitch_class``.
    """
    out = df.sort_values(PITCHER_CHRONO_KEYS).reset_index(drop=True)

    seq_frame = pd.DataFrame({col: _as_numbers(out[col]) for col in PREV_PITCH_LAGGED})
    seq_frame[LAGGED_CATEGORICAL] = seq_frame[LAGGED_CATEGORICAL].fillna(UNK)

    n_feat = len(PREV_PITCH_LAGGED)
    pitcher = out["pitcher"].to_numpy()
    values = seq_frame.to_numpy(dtype="float64", na_value=np.nan)
    n = len(out)

    X_seq = np.full((n, window, n_feat), float(UNK), dtype="float64")
    for t in range(n):
        # Walk backward while still the same pitcher.
        for k in range(1, window + 1):
            j = t - k
            if j < 0 or pitcher[j] != pitcher[t]:
                break
            X_seq[t, window - k, :] = values[j]  # oldest at index 0

    X_ctx = out[MODEL_INPUTS].apply(_as_numbers).to_numpy(dtype="float64", na_value=np.nan)
    return X_seq, X_ctx, list(PREV_PITCH_LAGGED)


# ==============================================================================
# 6. ORCHESTRATION
# ==============================================================================

def build(path: str | Path = CLEAN_PATH, pitcher: int | None = None, window: int = WINDOW) -> dict:
    """Full pipeline: load -> encode -> encode_ids -> scale.

    Returns every row in one encoded+scaled table, in ``PITCHER_CHRONO_KEYS``
    order so its rows line up with ``build_sequences``, and the scaler fit on
    its train rows.
    """
    encoded = encode(load(path, pitcher)).sort_values(PITCHER_CHRONO_KEYS, ignore_index=True)
    encoded = encode_ids(encoded)
    scaler = fit_scaler(encoded[encoded["split"] == "train"])
    table = apply_scaler(encoded, scaler)

    print(table.groupby("split", observed=True).agg(rows=("rid", "size"), eligible=("eligible", "sum")).to_string())
    return {"table": table, "scaler": scaler, "window": window}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean", type=Path, default=CLEAN_PATH)
    parser.add_argument("--pitcher", type=int, default=453286, help="MLBAM id (default: Scherzer); 0 = every pitcher")
    parser.add_argument("--window", type=int, default=WINDOW)
    args = parser.parse_args()

    start = time.time()
    built = build(args.clean, args.pitcher or None, args.window)
    X_seq, X_ctx, cols = build_sequences(built["table"], args.window)
    print(f"\nBuilt {len(X_seq):,} sequences x {args.window} pitches x {len(cols)} features "
          f"+ {X_ctx.shape[1]} context inputs in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
