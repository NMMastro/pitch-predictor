"""Sequence features: clean pitch table -> transformer inputs, kept in memory.

Input:  data/clean/pitches.parquet (from clean.py), optionally one pitcher.
Output: nothing on disk. Training stores one feature row per pitch and gathers
        the windows below per batch. ``build_dataset()`` materializes all the
        windows for small diagnostics and tests.

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
1 for missing / unknown; see :data:`CODE_SHIFT`. ``vocab_sizes`` gives the
number of embedding rows each one needs.

Defaults (DESIGN_DECISIONS.md Section 6, "v1 scope" and "Sequence-model encoding"):
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
    ENCODED_PRE_PITCH_COLUMNS,
    INNING_CAP,
    PITCH_CLASSES,
    PITCH_RESULTS,
    PITCHER_CHRONO_KEYS,
    PRE_PITCH_COLUMNS,
    ZONE_CLASSES,
)

# ==============================================================================
# 1. MODEL INPUTS (DESIGN_DECISIONS.md Section 6)
# ==============================================================================

# --- Embedding index convention ---
# Every categorical fed to an nn.Embedding is stored as ``code + CODE_SHIFT``,
# so index 0 is free for padding and index 1 for missing / unknown. A code of
# -1 (encode.py's "none") lands on UNK_INDEX by construction. Transformer-only:
# the tabular pipeline keeps encode.py's plain codes.
PAD_INDEX = 0
UNK_INDEX = 1
CODE_SHIFT = 2

# --- 6b. Lagged previous-pitch features (describe PRIOR pitches only) ---
# encode.py's codes, not the raw columns: zone_code is mirrored for left-handed
# batters, result_code groups the 30-odd descriptions into 6 outcomes, and
# plate_x_rel is negative = inside for either hand.
SEQ_CATEGORICAL = ["type_code", "zone_code", "result_code"]
SEQ_NUMERIC = ["release_speed", "plate_x_rel", "plate_z"]

# --- 6a. Current-pitch inputs: known before the pitch, fed directly ---
INPUT_EMBED = ["batter", "pitcher", "fielder_2"]  # fielder_2 = catcher

# Keep the real pitcher ID for window grouping and the existing "pitcher"
# embedding key for model configs and checkpoints.
_ID_CODE_COLUMNS = {"pitcher": "pitcher_code"}

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

# The previous-pitch window never resets, so these flags let the model discount
# a "previous pitch" that came from another at-bat or plate appearance. encode.py
# computes them for every pitch; only the sequence model feeds them as inputs.
FIRST_PITCH_FLAGS = ["is_first_pitch_of_ab", "is_first_pitch_of_pa"]

# Everything else from 6a, in ctx_num order. The embedded columns drop out:
# `inning` is replaced by `inning_code`, the alignments by their code columns.
_CTX_EMBED_SOURCES = ["inning", "if_fielding_alignment", "of_fielding_alignment"]
CTX_NUMERIC = (
    [c for c in ENCODED_PRE_PITCH_COLUMNS if c not in _CTX_EMBED_SOURCES]
    + FIRST_PITCH_FLAGS
    + MISSING_FLAGS
)

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

# Defaults (DESIGN_DECISIONS.md Section 6, "v1 scope" and "Sequence-model encoding").
WINDOW = 64           # previous-pitch window length for the 6b lags
MIN_BATTER_COUNT = 5  # batters seen fewer times than this collapse to UNK
UNK = -1              # encode.py's "none" code; CODE_SHIFT maps it to UNK_INDEX


# ==============================================================================
# 2. LOAD
# ==============================================================================

_INPUT_COLUMNS = [
    "rid", "split", "eligible", "game_pk", "at_bat_number", "pitch_number",
    *INPUT_EMBED, *PRE_PITCH_COLUMNS,
    "pitch_class", "zone_class", "description", "events", "release_speed", "plate_x", "plate_z",
]


def load(
    path: str | Path = CLEAN_PATH, pitcher: int | None = None,
    *, columns: list[str] | None = None,
) -> pd.DataFrame:
    """The clean table, optionally one pitcher's rows."""
    filters = [("pitcher", "==", pitcher)] if pitcher is not None else None
    return pd.read_parquet(path, columns=columns, filters=filters)


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
    """Embedding ids -> contiguous codes with an UNK bucket for rare or unseen ids.

    The vocabulary is built from train rows only: an id gets its own code if it
    appears in train at least the minimum number of times. Rare and unseen IDs
    share the UNK code instead of receiving their own embedding rows. Codes
    start at 0; UNK is -1. The raw pitcher ID stays in ``pitcher``.
    """
    for col in INPUT_EMBED:
        code_col = _ID_CODE_COLUMNS.get(col, col)
        min_count = min_batter_count if col == "batter" else 1
        df[code_col] = _encode_id(df, col, min_count)
    return df


def _encode_id(df: pd.DataFrame, col: str, min_count: int) -> pd.Series:
    """Map an id column to contiguous int codes; ids rare or absent in train -> UNK."""
    counts = df.loc[df["split"] == "train", col].value_counts()
    keep = set(counts[counts >= min_count].index)
    codes = {v: i for i, v in enumerate(sorted(keep))}
    return df[col].map(codes).fillna(UNK).astype("int32")


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
            code_col = _ID_CODE_COLUMNS.get(col, col)
            n_values = int(df[code_col].max()) + 1  # 0..n-1, or -1 for UNK only
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


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, *, copy: bool = True) -> pd.DataFrame:
    """Apply a pre-fit scaler to every row, then fill the gaps with 0.

    StandardScaler passes NaN through, and 0 is the train mean of a
    standardised column, so a missing value becomes "average" -- which is why
    :func:`missing_flags` records that it happened.
    """
    cols = list(scaler.feature_names_in_)
    out = df.copy() if copy else df
    scaled = scaler.transform(out[cols].astype("float64"))
    out[cols] = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return out


# ==============================================================================
# 5. SEQUENCES
# ==============================================================================

def _history_starts(table: pd.DataFrame) -> np.ndarray:
    """For each row, find its pitcher's first row in the pitcher-sorted table."""
    # Use raw IDs here since unrelated pitchers can share the UNK model code.
    pitcher = table["pitcher"].to_numpy()
    boundaries = np.flatnonzero(pitcher[1:] != pitcher[:-1]) + 1
    starts = np.zeros(len(table), dtype=np.int64)
    starts[boundaries] = boundaries
    np.maximum.accumulate(starts, out=starts)
    return starts


def _window_index(
    rows: np.ndarray, starts: np.ndarray, offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Every row looks backward. Positions before this pitcher's first row
    # point to a safe index and are masked after gathering.
    previous = rows[:, None] + offsets
    padding = previous < starts[:, None]
    previous[padding] = 0
    return previous, padding


def _gather_index(table: pd.DataFrame, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Eager history index for a small, pitcher-sorted table."""
    return _window_index(np.arange(len(table)), _history_starts(table), np.arange(-window, 0))


def build_sequence_sources(table: pd.DataFrame) -> dict[str, np.ndarray]:
    """Store one feature row per pitch in the order returned by :func:`build`."""
    ctx_code_columns = [_ID_CODE_COLUMNS.get(col, col) for col in CTX_CATEGORICAL]
    arrays = {
        "seq_cat": _int_matrix(table, SEQ_CATEGORICAL),
        "seq_num": _float_matrix(table, SEQ_NUMERIC),
        "ctx_cat": _int_matrix(table, ctx_code_columns),
        "ctx_num": _float_matrix(table, CTX_NUMERIC),
        "y": table[LABEL].to_numpy(dtype=np.int64),
        "history_start": _history_starts(table),
    }
    _check(arrays)
    return arrays


def gather_history(
    arrays: dict[str, np.ndarray], rows: np.ndarray, offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build only the history windows requested by the caller."""
    previous, padding = _window_index(rows, arrays["history_start"][rows], offsets)
    seq_cat = arrays["seq_cat"][previous].astype(np.int64, copy=False)
    seq_num = arrays["seq_num"][previous]
    seq_cat[padding] = PAD_INDEX
    seq_num[padding] = 0.0
    return seq_cat, seq_num, padding


def build_sequences(table: pd.DataFrame, window: int = WINDOW) -> dict[str, np.ndarray]:
    """The transformer's input arrays for every row of ``table``.

    ``table`` is the output of :func:`build`, already encoded, scaled and
    sorted, so row i of each array describes row i of the table. See the module
    docstring for the shapes. Nothing is written to disk and nothing is fit.
    """
    table = table.sort_values(PITCHER_CHRONO_KEYS).reset_index(drop=True)
    arrays = build_sequence_sources(table)
    seq_cat, seq_num, pad = gather_history(arrays, np.arange(len(table)), np.arange(-window, 0))
    return {
        "seq_cat": seq_cat,
        "seq_num": seq_num,
        "pad_mask": pad,
        "ctx_cat": arrays["ctx_cat"].astype(np.int64),
        "ctx_num": arrays["ctx_num"],
        "y": arrays["y"],
    }


def _int_matrix(table: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Store shifted integer codes compactly. Batches convert them to int64."""
    values = table[columns].to_numpy(dtype=np.int32)
    return np.ascontiguousarray(values + CODE_SHIFT)


def _float_matrix(table: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """``(n, len(columns))`` float32 of the given columns, nulls as 0."""
    values = table[columns].to_numpy(dtype=np.float32, na_value=np.nan, copy=True)
    values[np.isnan(values)] = 0.0
    return np.ascontiguousarray(values)


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

def build(
    path: str | Path = CLEAN_PATH, pitcher: int | None = None, verbose: bool = True,
    *, compact: bool = False,
) -> dict:
    """Load -> encode -> code -> ids -> missing flags -> scale.

    Returns the encoded, scaled table in :data:`PITCHER_CHRONO_KEYS` order so
    its rows line up with :func:`build_sequences`, the two scalers fit on its
    train rows, and the embedding vocabulary sizes.
    """
    columns = _INPUT_COLUMNS if compact else None
    table = encode(load(path, pitcher, columns=columns)).sort_values(PITCHER_CHRONO_KEYS, ignore_index=True)
    table = add_codes(table)
    table = encode_ids(table)
    table = missing_flags(table)

    train = table["split"] == "train"
    if not train.any():
        raise ValueError("no train rows; the scalers have nothing to fit on")
    scalers = {
        "ctx": fit_scaler(table.loc[train, CONTINUOUS_INPUTS], CONTINUOUS_INPUTS),
        "seq": fit_scaler(table.loc[train, SEQ_NUMERIC], SEQ_NUMERIC),
    }
    for scaler in scalers.values():
        table = apply_scaler(table, scaler, copy=False)

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

    Use this eager helper on small inputs. Training uses the per-pitch arrays
    from :func:`build_sequence_sources` instead.
    """
    built = build(path, pitcher, verbose)
    arrays = build_sequences(built["table"], window)
    return {**arrays, **built, "window": window}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean", type=Path, default=CLEAN_PATH)
    parser.add_argument("--pitcher", type=int, default=453286, help="MLBAM id (default: Scherzer); 0 = every pitcher")
    parser.add_argument("--window", type=int, default=WINDOW, help="history length for the sample batch")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be at least 1")

    start = time.time()
    data = build(args.clean, args.pitcher or None, compact=True)
    arrays = build_sequence_sources(data["table"])
    rows = np.arange(min(128, len(data["table"])))
    seq_cat, seq_num, pad = gather_history(arrays, rows, np.arange(-args.window, 0))

    print(f"\nSample batch (window {args.window}, {time.time() - start:.0f}s)")
    for name, array in (
        ("seq_cat", seq_cat), ("seq_num", seq_num), ("pad_mask", pad),
        ("ctx_cat", arrays["ctx_cat"][rows].astype(np.int64)),
        ("ctx_num", arrays["ctx_num"][rows]), ("y", arrays["y"][rows]),
    ):
        print(f"  {name:9} {str(array.shape):18} {array.dtype}")

    print("\nEmbedding vocabularies")
    for col, size in data["vocab_sizes"].items():
        print(f"  {col:24} {size}")

    print(f"\n{pad.mean():.1%} of sample sequence slots are padding")


if __name__ == "__main__":
    main()
