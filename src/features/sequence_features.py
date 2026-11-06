"""Sequence features: clean pitch table -> transformer inputs, kept in memory.

Input:  data/clean/pitches.parquet (from clean.py), optionally one pitcher.
Output: nothing on disk. ``build_dataset()`` returns the arrays below plus the
        table they came from, so row i of every array is row i of the table.

Every input for a pitch uses only its pre-pitch state and earlier pitches.
History never resets at plate appearances, games, seasons or splits, so every
split stays in one table (DESIGN_DECISIONS.md Section 4).

Steps, in the order they run:

  build()
    1. load            the clean table, optionally one pitcher (the v1 scope)
    2. encode          every pitch as numbers (encode.py)
    3. add_codes       nullable categoricals -> plain int code columns, -1 = none
    4. encode_ids      embedding ids -> contiguous codes, rare batters -> UNK
    5. missing_flags   a 0/1 column per input that can be null
    6. scale           standardise the continuous inputs, fit on train rows only,
                       then fill what is left with 0 (the train mean)

  build_sequences()
    7. the arrays below

Arrays
------
``build_sequences`` returns a dict. ``n`` is the number of table rows and ``W``
the window length. Nothing contains NaN.

  seq_cat   (n, W, 3) int64    previous pitches: type, zone, result
  seq_num   (n, W, 3) float32  previous pitches: release_speed, plate_x_rel, plate_z
  pad_mask  (n, W)    bool     True where the slot is padding (PyTorch's
                              "ignore this position" convention)
  ctx_cat   (n, 6)    int64    this pitch: batter, pitcher, catcher, the two
                              fielding alignments, inning
  ctx_num   (n, 26)   float32  this pitch: count, base/out state, score, zone
                              size, rest, ages and the 0/1 flags
  y         (n,)      int64    the label: index into schema.PITCH_CLASSES

Slot W-1 of each window is the pitcher's immediately previous pitch and slot 0
the oldest. Windows at the start of a pitcher's history are padded on the left.

Every categorical is stored as ``code + CODE_SHIFT``, leaving 0 for padding and
1 for missing / unknown; see ``schema.CODE_SHIFT``. ``vocab_sizes`` gives the
number of embedding rows each one needs.

Defaults for two open decisions (DESIGN_DECISIONS.md Section 6):
  WINDOW = 64            previous pitches in each sequence
  MIN_BATTER_COUNT = 5   batters seen fewer times become UNK

Usage:
    python src/features/sequence_features.py                 # Scherzer, window 64
    python src/features/sequence_features.py --window 400
    python src/features/sequence_features.py --pitcher 0     # every pitcher
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
    CODE_SHIFT,
    ENCODED_PRE_PITCH_COLUMNS,
    INNING_CAP,
    PITCH_CLASSES,
    PITCH_RESULTS,
    PITCHER_CHRONO_KEYS,
    ZONE_CLASSES,
)

# ==============================================================================
# 1. MODEL INPUTS (DESIGN_DECISIONS.md Section 6)
# ==============================================================================

# --- 6b. Lagged previous-pitch features (describe PRIOR pitches only) ---
# encode.py's codes, not the raw columns: zone_code is mirrored for left-handed
# batters, result_code groups the 30-odd descriptions into 6 outcomes, and
# plate_x_rel is negative = inside for either hand.
SEQ_CATEGORICAL = ["type_code", "zone_code", "result_code"]
SEQ_NUMERIC = ["release_speed", "plate_x_rel", "plate_z"]

# --- 6a. Current-pitch inputs: known before the pitch, fed directly ---
INPUT_EMBED = ["batter", "pitcher", "fielder_2"]  # fielder_2 = catcher

# Pre-pitch categoricals. The alignments are text in the clean table and the
# inning is small and non-linear, so all three are embedded rather than scaled
# ("bucket rather than feed raw", Section 6a).
CTX_EMBED_OTHER = ["if_align_code", "of_align_code", "inning_code"]
CTX_CATEGORICAL = INPUT_EMBED + CTX_EMBED_OTHER

# Continuous 6a inputs that get standardised.
CONTINUOUS_INPUTS = [
    "bat_score", "fld_score", "home_score_diff", "bat_score_diff",
    "pitcher_days_since_prev_game", "batter_days_since_prev_game",
    "age_pit", "age_bat", "sz_top", "sz_bot",
]

# Inputs that can be null; each gets a companion 0/1 "was missing" flag, because
# filling with the mean would otherwise look like an average amount of rest.
NULLABLE_INPUTS = ["pitcher_days_since_prev_game", "batter_days_since_prev_game"]
MISSING_FLAGS = [f"{col}_missing" for col in NULLABLE_INPUTS]

# Everything else from 6a, in ctx_num order. The embedded columns drop out:
# `inning` is replaced by `inning_code`, the alignments by their code columns.
_CTX_EMBED_SOURCES = ["inning", "if_fielding_alignment", "of_fielding_alignment"]
CTX_NUMERIC = [c for c in ENCODED_PRE_PITCH_COLUMNS if c not in _CTX_EMBED_SOURCES] + MISSING_FLAGS

# The label.
LABEL = "type_code"

# How many real values each categorical can take, before CODE_SHIFT. Taken from
# the taxonomy, not the data, so one pitcher's arrays stay compatible with
# everyone's. Id and alignment vocabularies depend on the rows and are measured.
TAXONOMY_SIZES = {
    "type_code": len(PITCH_CLASSES),
    "zone_code": len(ZONE_CLASSES),
    "result_code": len(PITCH_RESULTS),
    "inning_code": INNING_CAP,
}

# Open-decision defaults (DESIGN_DECISIONS.md Section 6 "Open decisions").
WINDOW = 64           # previous-pitch window length for the 6b lags
MIN_BATTER_COUNT = 5  # batters seen fewer times than this collapse to UNK
UNK = -1              # encode.py's "none" code; CODE_SHIFT maps it to UNK_INDEX


# ==============================================================================
# 2. LOAD
# ==============================================================================

def load(path: str | Path = CLEAN_PATH, pitcher: int | None = None) -> pd.DataFrame:
    """The clean table, optionally one pitcher's rows."""
    filters = [("pitcher", "==", pitcher)] if pitcher is not None else None
    return pd.read_parquet(path, filters=filters)


# ==============================================================================
# 3. CODES (stateless)
# ==============================================================================

def add_codes(df: pd.DataFrame) -> pd.DataFrame:
    """The remaining categoricals as plain int code columns, -1 = none.

    ``encode.py`` already produced the pitch codes; these are the pre-pitch
    ones. The alignment vocabularies come from the column's dtype rather than
    the values present, so filtering to one pitcher cannot shift them.
    """
    return df.assign(
        if_align_code=_category_codes(df["if_fielding_alignment"]),
        of_align_code=_category_codes(df["of_fielding_alignment"]),
        # inning is 1-based and capped at INNING_CAP by encode.py.
        inning_code=(df["inning"].astype("int64") - 1).clip(0, INNING_CAP - 1),
    )


def _category_codes(values: pd.Series) -> np.ndarray:
    """A category column's codes as int64; -1 where null."""
    return values.cat.codes.to_numpy().astype(np.int64)


def encode_ids(df: pd.DataFrame, min_batter_count: int = MIN_BATTER_COUNT) -> pd.DataFrame:
    """Embedding ids -> contiguous codes with an UNK bucket (Open Decision #3).

    Stateless (no fitting). Rarity is judged on the frame passed in, which for a
    leakage-clean pipeline should be a vocabulary decided once and reused; for
    v1 single-pitcher data it is computed here. Codes start at 0; UNK is -1.
    """
    for col in INPUT_EMBED:
        df = _encode_id(df, col, min_batter_count if col == "batter" else 1)
    return df


def _encode_id(df: pd.DataFrame, col: str, min_count: int) -> pd.DataFrame:
    """Map an id column to contiguous int codes, rare ids -> UNK."""
    counts = df[col].value_counts()
    keep = set(counts[counts >= min_count].index)
    codes = {v: i for i, v in enumerate(sorted(keep))}
    df[col] = df[col].map(lambda v: codes.get(v, UNK)).astype("int64")
    return df


def vocab_sizes(df: pd.DataFrame) -> dict[str, int]:
    """Embedding rows needed per categorical: real values + padding + UNK.

    Fixed vocabularies come from :data:`TAXONOMY_SIZES`; id and alignment
    vocabularies are measured from ``df``, whose codes run 0..n-1.
    """
    sizes = {}
    for col in SEQ_CATEGORICAL + CTX_CATEGORICAL:
        if col in TAXONOMY_SIZES:
            n_values = TAXONOMY_SIZES[col]
        else:
            n_values = int(df[col].max()) + 1  # codes are 0..n-1, or -1 for UNK only
        sizes[col] = max(n_values, 0) + CODE_SHIFT
    return sizes


# ==============================================================================
# 4. MISSING VALUES AND SCALING (fit on train rows only)
# ==============================================================================

def missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    """A 0/1 column per nullable input, recorded before anything is filled."""
    return df.assign(**{
        f"{col}_missing": df[col].isna().astype(np.int8) for col in NULLABLE_INPUTS
    })


def fit_scaler(train: pd.DataFrame, cols: list[str]) -> StandardScaler:
    """Fit a scaler on TRAIN ONLY (the fix for scaler leakage)."""
    scaler = StandardScaler().fit(train[cols].astype("float64"))
    scaler.feature_names_in_ = np.array(cols)  # remember the column order
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a pre-fit scaler to every row, then fill the gaps with 0.

    StandardScaler passes NaN through, and 0 is the train mean of a
    standardised column, so a missing value becomes "average" -- which is why
    :func:`missing_flags` records that it happened.
    """
    cols = list(scaler.feature_names_in_)
    out = df.copy()
    scaled = scaler.transform(out[cols].astype("float64"))
    out[cols] = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return out


# ==============================================================================
# 5. SEQUENCES
# ==============================================================================

def _gather_index(table: pd.DataFrame, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of each pitch's previous ``window`` pitches, and a padding mask.

    Vectorised, because the Python version was O(n x window) and the window can
    be hundreds of pitches long.

    ``table`` must already be sorted by :data:`PITCHER_CHRONO_KEYS`, so one
    pitcher's rows are contiguous and in real-time order. Within a pitcher,
    pitch ``p`` wants positions ``p-window .. p-1``; adding the index of that
    pitcher's first row turns those into table row numbers.

    Returns ``(rows, pad)``, both ``(n, window)``. Where ``pad`` is True the
    slot has no earlier pitch and ``rows`` holds a harmless 0.
    """
    n = len(table)
    position = table.groupby("pitcher", sort=False).cumcount().to_numpy()
    first_row = np.arange(n) - position  # table index of this pitcher's first pitch

    offsets = np.arange(-window, 0)                       # (window,) -window .. -1
    relative = position[:, None] + offsets[None, :]       # (n, window)
    pad = relative < 0                                    # before the pitcher's first pitch
    rows = np.where(pad, 0, first_row[:, None] + relative)
    return rows, pad


def build_sequences(table: pd.DataFrame, window: int = WINDOW) -> dict[str, np.ndarray]:
    """The transformer's input arrays for every row of ``table``.

    ``table`` is the output of :func:`build`, already encoded, scaled and
    sorted, so row i of each array describes row i of the table. See the module
    docstring for the shapes. Nothing is written to disk and nothing is fit.
    """
    table = table.sort_values(PITCHER_CHRONO_KEYS).reset_index(drop=True)
    rows, pad = _gather_index(table, window)

    # Previous pitches: read the per-pitch values, then gather them per slot.
    seq_cat_src = _int_matrix(table, SEQ_CATEGORICAL)      # (n, 3), shifted
    seq_num_src = _float_matrix(table, SEQ_NUMERIC)        # (n, 3)

    # Indexing then overwriting beats np.where here: one (n, window, 3) array
    # instead of two, which matters at window 400.
    seq_cat = seq_cat_src[rows]
    seq_cat[pad] = 0  # PAD_INDEX
    seq_num = seq_num_src[rows]
    seq_num[pad] = 0.0

    out = {
        "seq_cat": seq_cat,
        "seq_num": seq_num,
        "pad_mask": np.ascontiguousarray(pad),
        "ctx_cat": _int_matrix(table, CTX_CATEGORICAL),
        "ctx_num": _float_matrix(table, CTX_NUMERIC),
        "y": table[LABEL].to_numpy().astype(np.int64),
    }
    _check(out)
    return out


def _int_matrix(table: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """``(n, len(columns))`` int64 of the given code columns, shifted past PAD/UNK."""
    values = table[columns].to_numpy(dtype=np.int64)
    return np.ascontiguousarray(values + CODE_SHIFT)


def _float_matrix(table: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """``(n, len(columns))`` float32 of the given columns, nulls as 0."""
    values = table[columns].to_numpy(dtype=np.float64, na_value=np.nan)
    return np.ascontiguousarray(np.nan_to_num(values, nan=0.0), dtype=np.float32)


def _check(arrays: dict[str, np.ndarray]) -> None:
    """Fail loudly on the mistakes that would surface as a NaN loss or a crash."""
    for name in ("seq_num", "ctx_num"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} contains NaN or inf; the model would train to NaN")
    for name in ("seq_cat", "ctx_cat"):
        if arrays[name].min() < 0:
            raise ValueError(f"{name} has negative codes; nn.Embedding needs indices >= 0")
    if arrays["y"].min() < 0:
        raise ValueError("y has negative labels; every row needs a pitch_class")


# ==============================================================================
# 6. ORCHESTRATION
# ==============================================================================

def build(path: str | Path = CLEAN_PATH, pitcher: int | None = None, verbose: bool = True) -> dict:
    """Load -> encode -> code -> ids -> missing flags -> scale.

    Returns the encoded, scaled table in :data:`PITCHER_CHRONO_KEYS` order so
    its rows line up with :func:`build_sequences`, the two scalers fit on its
    train rows, and the embedding vocabulary sizes.
    """
    table = encode(load(path, pitcher)).sort_values(PITCHER_CHRONO_KEYS, ignore_index=True)
    table = add_codes(table)
    table = encode_ids(table)
    table = missing_flags(table)

    train = table[table["split"] == "train"]
    if train.empty:
        raise ValueError("no train rows; the scalers have nothing to fit on")
    scalers = {
        "ctx": fit_scaler(train, CONTINUOUS_INPUTS),
        "seq": fit_scaler(train, SEQ_NUMERIC),
    }
    for scaler in scalers.values():
        table = apply_scaler(table, scaler)

    if verbose:
        counts = table.groupby("split", observed=True).agg(
            rows=("rid", "size"), eligible=("eligible", "sum")
        )
        print(counts.to_string())
    return {"table": table, "scalers": scalers, "vocab_sizes": vocab_sizes(table)}


def build_dataset(
    path: str | Path = CLEAN_PATH,
    pitcher: int | None = None,
    window: int = WINDOW,
    verbose: bool = True,
) -> dict:
    """Everything a model needs: the arrays, the table behind them, and metadata.

    The one call ``src/model/dataset.py`` makes. Keys: the arrays from
    :func:`build_sequences`, plus ``table``, ``scalers``, ``vocab_sizes`` and
    ``window``.
    """
    built = build(path, pitcher, verbose)
    arrays = build_sequences(built["table"], window)
    return {**arrays, **built, "window": window}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean", type=Path, default=CLEAN_PATH)
    parser.add_argument("--pitcher", type=int, default=453286, help="MLBAM id (default: Scherzer); 0 = every pitcher")
    parser.add_argument("--window", type=int, default=WINDOW)
    args = parser.parse_args()

    start = time.time()
    data = build_dataset(args.clean, args.pitcher or None, args.window)

    print(f"\nArrays (window {args.window}, {time.time() - start:.0f}s)")
    nans = 0
    for name in ("seq_cat", "seq_num", "pad_mask", "ctx_cat", "ctx_num", "y"):
        array = data[name]
        print(f"  {name:9} {str(array.shape):18} {array.dtype}")
        if array.dtype.kind == "f":
            nans += int(np.isnan(array).sum())
    print(f"  {nans} NaNs")

    print("\nEmbedding vocabularies")
    for col, size in data["vocab_sizes"].items():
        print(f"  {col:24} {size}")

    pad_share = data["pad_mask"].mean()
    print(f"\n{pad_share:.1%} of sequence slots are padding")


if __name__ == "__main__":
    main()
