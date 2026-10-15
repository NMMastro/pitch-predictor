"""Feature definition for the pitch-type model.

Single source of truth for *what the model sees*, implementing
``DESIGN_DECISIONS.md`` Section 6. The pipeline is:

    prepare (metrics)  ->  clean  ->  encode  ->  split  ->  build_{tabular,sequences}

* ``prepare`` (reused from ``evaluation.metrics``) drops spring training, drops
  null / non-pitch rows and attaches the ``pitch_class`` label.
* ``clean`` drops every column excluded by the contract — defence in depth
  against leakage.
* ``encode`` turns raw columns into model-ready numerics: booleans for
  handedness / baserunners / inning half, a capped ``inning``, contiguous
  integer codes for the embedding ids (with a rare-id ``UNK`` bucket), and
  integer codes for the lagged categoricals. It does **not** scale — scaling is
  stateful and must be fit on train only (see ``fit_scaler`` / ``apply_scaler``).
* ``split`` assigns train / val / test by season per the design doc.
* ``build_tabular`` flattens the previous-N-pitch lags into columns (baseline
  models). ``build_sequences`` emits ``(n, window, n_feat)`` arrays (transformer).

Both builders honour the leakage rule: a feature for pitch *t* uses only pitches
strictly before *t*, and history is continuous (it does not reset at at-bat,
game, season or split boundaries). Boundary flags mark where a "previous pitch"
came from a different plate appearance so the model can discount it.

The two Open Decisions this module takes a default on (override at call time):
window length ``WINDOW`` and the rare-batter threshold ``MIN_BATTER_COUNT``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from evaluation.metrics import prepare

# ==============================================================================
# 1. CANONICAL COLUMN LISTS (DESIGN_DECISIONS.md Section 6)
# ==============================================================================

# --- 6a. Current-pitch inputs: known before the pitch, fed directly ---
INPUT_CONTEXT = [
    "balls", "strikes", "outs_when_up", "inning", "inning_topbot",
    "stand", "p_throws", "on_1b", "on_2b", "on_3b",
    "bat_score", "fld_score", "home_score_diff", "bat_score_diff",
    "n_thruorder_pitcher", "n_priorpa_thisgame_player_at_bat",
    "pitcher_days_since_prev_game", "batter_days_since_prev_game",
    "age_pit", "age_bat", "sz_top", "sz_bot",
]
INPUT_EMBED = ["batter", "pitcher", "fielder_2"]  # fielder_2 = catcher

# --- 6b. Lagged previous-pitch features (described for PRIOR pitches only) ---
PREV_PITCH_LAGGED = [
    "pitch_type", "release_speed", "plate_x", "plate_z", "zone", "description",
]
# Which of the lagged columns are categorical (encoded to int codes, not scaled).
LAGGED_CATEGORICAL = ["pitch_type", "zone", "description"]

# --- 6c. Bookkeeping: build sequences / splits / demo, never fed ---
BOOKKEEPING = [
    "game_pk", "game_date", "game_year", "game_type",
    "at_bat_number", "pitch_number", "player_name",
    "pitch_type", "plate_x", "plate_z", "zone",
]
# Labels attached by prepare(); not model inputs, but expected in the frame.
LABELS = ["pitch_class", "zone_class"]

# Real-time ordering keys (used for the mandatory chronological sort).
SORT_KEYS = ["pitcher", "game_date", "game_pk", "at_bat_number", "pitch_number"]

# --- Excluded as current-pitch inputs (allowed only lagged, per 6b) ---
EXCL_FLIGHT_RELEASE = [
    "release_spin_rate", "spin_axis", "effective_speed",
    "release_extension", "release_pos_x", "release_pos_y", "release_pos_z",
    "pfx_x", "pfx_z", "vx0", "vy0", "vz0", "ax", "ay", "az",
    "api_break_z_with_gravity", "api_break_x_arm", "api_break_x_batter_in",
    "hyper_speed",
]
EXCL_OUTCOME = [
    "type", "des", "events", "bb_type", "hit_location",
    "launch_speed", "launch_angle", "launch_speed_angle", "hit_distance_sc",
    "hc_x", "hc_y", "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle", "estimated_slg_using_speedangle",
    "woba_value", "woba_denom", "babip_value", "iso_value",
    "delta_run_exp", "delta_pitcher_run_exp", "delta_home_win_exp",
]
EXCL_POST_STATE = [
    "post_away_score", "post_home_score", "post_bat_score", "post_fld_score",
    "home_win_exp", "bat_win_exp",
]
EXCL_V1 = [
    "pitcher_days_until_next_game", "batter_days_until_next_game",
    "fielder_3", "fielder_4", "fielder_5", "fielder_6",
    "fielder_7", "fielder_8", "fielder_9", "home_team", "away_team",
    "age_pit_legacy", "age_bat_legacy", "home_score", "away_score",
    "if_fielding_alignment", "of_fielding_alignment", "sv_id",
]
EXCL_NULL = [
    "arm_angle", "bat_speed", "swing_length", "attack_angle", "attack_direction",
    "swing_path_tilt", "miss_distance", "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
]
DEAD = [
    "spin_dir", "spin_rate_deprecated", "break_angle_deprecated",
    "break_length_deprecated", "tfs_deprecated", "tfs_zulu_deprecated", "umpire",
]

buckets = {
    "6a input: context": INPUT_CONTEXT,
    "6a input: embeddings": INPUT_EMBED,
    "6b lagged previous-pitch": PREV_PITCH_LAGGED,
    "6c bookkeeping": BOOKKEEPING,
    "labels": LABELS,
    "excluded: flight/release": EXCL_FLIGHT_RELEASE,
    "excluded: outcome": EXCL_OUTCOME,
    "excluded: post-state": EXCL_POST_STATE,
    "excluded: v1 decision": EXCL_V1,
    "excluded: null": EXCL_NULL,
    "dead": DEAD,
}


# TODO FIGURE THIS OUT
ALL_EXCLUDED = (
    EXCL_FLIGHT_RELEASE + EXCL_OUTCOME + EXCL_POST_STATE
    + EXCL_V1 + EXCL_NULL + DEAD
)

# Boolean / derived current-pitch inputs produced by encode().
DERIVED_FLAGS = ["is_LHB", "is_LHP", "is_inning_top",
                 "is_first_pitch_of_pa", "is_first_pitch_of_ab"]

# The current-pitch (6a) model-input columns after encoding. Single definition,
# imported by downstream code rather than recomputed.
MODEL_INPUTS = (
    [c for c in INPUT_CONTEXT if c not in ("stand", "p_throws", "inning_topbot")]
    + ["is_LHB", "is_LHP", "is_inning_top"]
    + INPUT_EMBED
)

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
INNING_CAP = 10       # extra innings are capped/bucketed at this value
UNK = -1              # reserved code for the UNK / padding bucket


# ==============================================================================
# 2. CLEAN
# ==============================================================================

def verify_and_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every excluded column; warn on anything unaccounted for.

    Defence in depth: even if a leakage column slipped past a later step, it is
    not in the frame to be used. The completeness warnings flag drift between
    the data and the Section 6 contract.
    """

    # TODO WHY IS THIS USEFUL
    assigned = {c for cols in buckets.values() for c in cols}
    unassigned = [c for c in df.columns if c not in assigned]
    if unassigned:
        print(f"[warn] columns not yet bucketed: {unassigned}")

    for name, cols in buckets.items():
        missing = [c for c in cols if c not in df.columns and c not in LABELS]
        if missing:
            print(f"[warn] listed but absent in data ({name}): {missing}")

    drop_targets = [c for c in ALL_EXCLUDED if c in df.columns]
    return df.drop(columns=drop_targets)


# ==============================================================================
# 3. ENCODE (stateless, no train/test leakage)
# ==============================================================================

def _encode_ids(df: pd.DataFrame, col: str, min_count: int) -> pd.DataFrame:
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


def encode(df: pd.DataFrame, min_batter_count: int = MIN_BATTER_COUNT) -> pd.DataFrame:
    """Turn raw columns into model-ready numerics. Stateless (no fitting).

    Produces the booleans, the capped inning, the boundary flags, the embedding
    id codes, and integer codes for the lagged categoricals. Scaling is NOT done
    here — it is stateful and must be fit on train only.
    """
    out = df.sort_values(SORT_KEYS).reset_index(drop=True).copy()

    # Handedness -> booleans.
    if "stand" in out.columns:
        out["is_LHB"] = (out["stand"] == "L").astype("int64")
    if "p_throws" in out.columns:
        out["is_LHP"] = (out["p_throws"] == "L").astype("int64")
    out = out.drop(columns=[c for c in ("stand", "p_throws") if c in out.columns])

    # Baserunners: id -> presence boolean.
    for base in ("on_1b", "on_2b", "on_3b"):
        if base in out.columns:
            out[base] = out[base].notna().astype("int64")

    # Inning half -> boolean; cap extra innings.
    if "inning_topbot" in out.columns:
        out["is_inning_top"] = (out["inning_topbot"] == "Top").astype("int64")
        out = out.drop(columns=["inning_topbot"])
    if "inning" in out.columns:
        out["inning"] = out["inning"].clip(upper=INNING_CAP).astype("int64")

    # Boundary flags (Open Decision #1): mark the first pitch of each PA / AB so
    # the model can discount a "previous pitch" from a different context.
    if {"game_pk", "at_bat_number", "pitch_number"} <= set(out.columns):
        out["is_first_pitch_of_ab"] = (out["pitch_number"] == 1).astype("int64")
        # A new plate appearance = a new (game_pk, at_bat_number) in time order.
        pa_key = out["game_pk"].astype(str) + "_" + out["at_bat_number"].astype(str)
        out["is_first_pitch_of_pa"] = (pa_key != pa_key.shift(1)).astype("int64")

    # Embedding ids -> contiguous codes with an UNK bucket (Open Decision #3).
    for col in INPUT_EMBED:
        if col in out.columns:
            out = _encode_ids(out, col, min_batter_count if col == "batter" else 1)

    return out


# ==============================================================================
# 4. SPLIT (by season, per DESIGN_DECISIONS.md Section 4)
# ==============================================================================

TRAIN_SEASONS = set(range(2016, 2024))  # 2016–2023
VAL_SEASONS = {2024}
TEST_SEASONS = {2025}


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Partition rows into train / val / test by ``game_year``."""
    year = df["game_year"]
    return {
        "train": df[year.isin(TRAIN_SEASONS)].copy(),
        "val": df[year.isin(VAL_SEASONS)].copy(),
        "test": df[year.isin(TEST_SEASONS)].copy(),
    }


def fit_scaler(train: pd.DataFrame, cols: list[str] = CONTINUOUS_INPUTS) -> StandardScaler:
    """Fit a scaler on TRAIN ONLY (the fix for scaler leakage)."""
    cols = [c for c in cols if c in train.columns]
    scaler = StandardScaler().fit(train[cols])
    scaler.feature_names_in_ = np.array(cols)  # remember the column order
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply a pre-fit scaler to a split. Does not fit."""
    cols = list(scaler.feature_names_in_)
    out = df.copy()
    out[cols] = scaler.transform(out[cols])
    return out


# ==============================================================================
# 5. LAG / WINDOW BUILDERS
# ==============================================================================

def _lag_columns() -> list[str]:
    """6b lag columns that are present after encoding (continuous ones)."""
    return [c for c in PREV_PITCH_LAGGED]


def build_tabular(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Flatten the previous-``window`` pitches into lag columns (baseline models).

    For each 6b column c and each lag i in 1..window, adds ``c_lag{i}`` holding
    that column's value i pitches earlier for the SAME pitcher, in real time
    order. History is continuous across at-bats/games/seasons; the boundary
    flags from encode() tell the model when a lag crosses a PA/AB.

    Assumes ``df`` is already encoded. Returns a wide frame; the first few
    pitches per pitcher have NaN/UNK lags (handle in the model or drop later).
    """
    out = df.sort_values(SORT_KEYS).reset_index(drop=True).copy()
    grouped = out.groupby("pitcher", sort=False)
    for col in _lag_columns():
        if col not in out.columns:
            continue
        for i in range(1, window + 1):
            lagged = grouped[col].shift(i)
            if col in LAGGED_CATEGORICAL:
                lagged = lagged.fillna(UNK)
            out[f"{col}_lag{i}"] = lagged
    return out


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
    out = df.sort_values(SORT_KEYS).reset_index(drop=True).copy()
    seq_cols = [c for c in _lag_columns() if c in out.columns]

    # Encode the categorical sequence columns to int codes so the tensor is
    # numeric. (Continuous ones are left as-is; scale upstream if desired.)
    seq_frame = out[seq_cols].copy()
    for col in LAGGED_CATEGORICAL:
        if col in seq_frame.columns:
            seq_frame[col] = seq_frame[col].astype("category").cat.codes  # -1 = NaN

    n_feat = len(seq_cols)
    pitcher = out["pitcher"].to_numpy()
    values = seq_frame.to_numpy(dtype="float64")
    n = len(out)

    X_seq = np.full((n, window, n_feat), float(UNK), dtype="float64")
    for t in range(n):
        # Walk backward while still the same pitcher.
        for k in range(1, window + 1):
            j = t - k
            if j < 0 or pitcher[j] != pitcher[t]:
                break
            X_seq[t, window - k, :] = values[j]  # oldest at index 0

    ctx_cols = [c for c in MODEL_INPUTS if c in out.columns]
    X_ctx = out[ctx_cols].to_numpy(dtype="float64")
    return X_seq, X_ctx, seq_cols


# ==============================================================================
# 6. ORCHESTRATION
# ==============================================================================

def build(input_path: str, window: int = WINDOW) -> dict:
    """Full pipeline: load -> prepare -> clean -> encode -> split -> scale.

    Returns a dict with the per-split encoded+scaled frames and the fitted
    scaler, ready for either builder. Builders are called by the caller so the
    tabular vs sequence choice stays explicit.
    """
    raw = pd.read_parquet(input_path)
    prepared = prepare(raw)
    cleaned = verify_and_clean_columns(prepared)
    encoded = encode(cleaned)

    parts = split(encoded)
    scaler = fit_scaler(parts["train"])
    scaled = {name: apply_scaler(part, scaler) for name, part in parts.items()}

    for name, part in scaled.items():
        print(f"  {name:5s}  {len(part):>7,} rows")
    return {"splits": scaled, "scaler": scaler, "window": window}


def main(input_path: str, output_path: str, window: int = WINDOW) -> None:
    """CLI entry: build the tabular baseline table and write it to Parquet."""
    built = build(input_path, window=window)
    tab = {name: build_tabular(part, window=window)
           for name, part in built["splits"].items()}
    combined = pd.concat(tab.values(), ignore_index=True)
    combined.to_parquet(output_path, index=False)
    print(f"\nPipeline complete. Wrote {len(combined):,} rows to {output_path}")
    print(f"6a model inputs: {len(MODEL_INPUTS)}  |  window: {window}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="data/pitchers/453286.parquet",
        help="Per-pitcher raw Parquet (default: Scherzer)",
    )
    parser.add_argument(
        "--output", default="data/features/453286_tabular.parquet",
        help="Where the tabular feature table is written",
    )
    parser.add_argument("--window", type=int, default=WINDOW)
    args = parser.parse_args()

    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    main(args.input, args.output, window=args.window)
