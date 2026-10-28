"""Tabular features: clean pitch table -> one wide feature table for XGBoost.

Input:  data/clean/pitches.parquet (from clean.py), encoded by encode.py.
Output: data/features/tabular.parquet, one row per pitch: META_COLUMNS, then
        every feature below. Join to the clean table on rid.

Added features (FEATURE_GROUPS; each uses only earlier pitches):
  game_state        the encoded pre-pitch columns
  pitcher_types     p_type_{slice}_{class}: the pitcher's pitch-type mix, over
                    slices career, last200, vs_hand, count, game
  pitcher_zones     p_zone_{slice}_{zone}: the same over zones, batter's view
  batter_mix        b_type_*, b_zone_*: how pitchers have attacked this batter
  previous_pitches  prev*: his last pitches this game (type, zone, result, speed, ...)
  at_bat            pa_*: this plate appearance so far
  workload          game_pitch_count, is_starter

Sparse mixes are shrunk toward a broader one (PSEUDO_PITCHES, league_mixes).

Usage:
    python src/features/tabular_features.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.encode import encode  # noqa: E402
from features.schema import (  # noqa: E402
    CLEAN_PATH,
    ENCODED_PRE_PITCH_COLUMNS,
    FAMILIES,
    PITCH_CLASSES,
    PITCH_RESULTS,
    SWINGS,
    ZONE_CLASSES,
    ZONE_LABELS,
)

OUT_PATH = Path("data/features/tabular.parquet")

PSEUDO_PITCHES = 20.0   # pitches of the broader mix added to every mix
RECENT_PITCHES = 200    # length of the "last 200" window
PREV_TYPE_LAGS = 5      # pitch types of the last 5 pitches
PREV_ZONE_LAGS = 3      # zones of the last 3 pitches


# --- feature columns ----------------------------------------------------------

SLICES = ["career", "last200", "vs_hand", "count", "game"]

PREV_TYPE_COLUMNS = [f"prev{k}_type" for k in range(1, PREV_TYPE_LAGS + 1)]
PREV_ZONE_COLUMNS = [f"prev{k}_zone" for k in range(1, PREV_ZONE_LAGS + 1)]

FEATURE_GROUPS: dict[str, list[str]] = {
    "game_state": ENCODED_PRE_PITCH_COLUMNS,
    "pitcher_types": [
        *(f"p_type_{s}_{c}" for s in SLICES for c in PITCH_CLASSES),
        "pitcher_prior_pitches",
    ],
    "pitcher_zones": [
        *(f"p_zone_{s}_{z}" for s in SLICES for z in ZONE_LABELS),
    ],
    "batter_mix": [
        *(f"b_type_{c}" for c in PITCH_CLASSES),
        *(f"b_zone_{z}" for z in ZONE_LABELS),
        "batter_prior_pitches",
    ],
    "previous_pitches": [
        *PREV_TYPE_COLUMNS,
        *PREV_ZONE_COLUMNS,
        "prev_result", "prev_swung", "prev_same_pa",
        "prev_speed", "prev_speed_vs_avg", "prev_spin", "prev_spin_axis", "prev_extension",
        "prev_pfx_x", "prev_pfx_z", "prev_release_x", "prev_release_z",
        "prev_plate_x", "prev_plate_z",
        "last_pa_result",
    ],
    "at_bat": [
        "pa_pitch_idx", "pa_n_fastball", "pa_n_breaking", "pa_n_offspeed", "streak_same_type",
    ],
    "workload": [
        "game_pitch_count", "is_starter",
    ],
}
FEATURE_COLUMNS = [col for cols in FEATURE_GROUPS.values() for col in cols]

CATEGORICAL_FEATURES = [
    "if_fielding_alignment", "of_fielding_alignment",
    *PREV_TYPE_COLUMNS,
    *PREV_ZONE_COLUMNS,
    "prev_result", "last_pa_result",
]

META_COLUMNS = ["rid", "split", "eligible", "game_year", "pitch_class", "zone_class"]


# --- pipeline -----------------------------------------------------------------

def build(table: pd.DataFrame, out: Path = OUT_PATH) -> int:
    """Clean table -> tabular feature table, written to ``out``. Returns the row count.

    Every step works on the whole table in rid (real-time) order.
    """
    table = encode(table)
    league_types, league_zones = league_mixes(table)

    features = pd.concat([
        game_state(table),
        pitcher_types(table, league_types),
        pitcher_zones(table, league_zones),
        batter_mix(table, league_types, league_zones),
        previous_pitches(table),
        at_bat(table),
        workload(table),
    ], axis=1)
    if list(features.columns) != FEATURE_COLUMNS:
        raise AssertionError("feature columns do not match FEATURE_GROUPS")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    pd.concat([table[META_COLUMNS], features], axis=1).to_parquet(tmp, index=False)
    tmp.replace(out)
    return len(table)


def league_mixes(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per pitch: the league's pitch-type and zone mix over every earlier date.

    Not a feature: the mix that career and batter mixes are shrunk toward when
    they have little history.
    """
    _, day = np.unique(table["game_date"].to_numpy(), return_inverse=True)
    types = _mix_before_each_day(day, table["type_code"].to_numpy(), len(PITCH_CLASSES))
    zones = _mix_before_each_day(day, table["zone_code"].to_numpy(), len(ZONE_CLASSES))
    return types[day], zones[day]


# --- feature groups, in FEATURE_GROUPS order ----------------------------------

def game_state(df: pd.DataFrame) -> pd.DataFrame:
    """The pre-pitch columns and encode()'s flags, with fielding alignments as categories."""
    out = df[FEATURE_GROUPS["game_state"]].copy()
    for col in ["if_fielding_alignment", "of_fielding_alignment"]:
        out[col] = _category(out[col].cat.codes)
    return out


def pitcher_types(df: pd.DataFrame, league_types: np.ndarray) -> pd.DataFrame:
    """The pitcher's pitch-type mix over his earlier pitches, five ways."""
    mixes = _slice_mixes(df, one_hot(df["type_code"], len(PITCH_CLASSES)), league_types)

    columns = {}
    for name, mix in mixes.items():
        columns |= _mix_columns(f"p_type_{name}", mix, PITCH_CLASSES)
    columns["pitcher_prior_pitches"] = df["pitcher_prior_pitches"].to_numpy()
    return pd.DataFrame(columns)


def pitcher_zones(df: pd.DataFrame, league_zones: np.ndarray) -> pd.DataFrame:
    """The pitcher's zone mix (batter's view) over his earlier pitches, five ways."""
    mixes = _slice_mixes(df, one_hot(df["zone_code"], len(ZONE_CLASSES)), league_zones)

    columns = {}
    for name, mix in mixes.items():
        columns |= _mix_columns(f"p_zone_{name}", mix, ZONE_LABELS)
    return pd.DataFrame(columns)


def batter_mix(df: pd.DataFrame, league_types: np.ndarray, league_zones: np.ndarray) -> pd.DataFrame:
    """How pitchers have attacked this batter on earlier pitches, shrunk toward the league."""
    batter = df["batter"].to_numpy()
    type_counts = prior_counts(one_hot(df["type_code"], len(PITCH_CLASSES)), batter)
    zone_counts = prior_counts(one_hot(df["zone_code"], len(ZONE_CLASSES)), batter)

    return pd.DataFrame({
        **_mix_columns("b_type", shrink(type_counts, league_types), PITCH_CLASSES),
        **_mix_columns("b_zone", shrink(zone_counts, league_zones), ZONE_LABELS),
        "batter_prior_pitches": type_counts.sum(axis=1),
    })


def previous_pitches(df: pd.DataFrame) -> pd.DataFrame:
    """This pitcher's earlier pitches in this game. Missing on his first pitch of each game."""
    game = group_ids(df, ["pitcher", "game_pk"])

    def prev(values, k: int = 1) -> pd.Series:
        return pd.Series(np.asarray(values)).groupby(game).shift(k)

    out = {}
    for k, col in enumerate(PREV_TYPE_COLUMNS, start=1):
        out[col] = _category(prev(df["type_code"], k))
    for k, col in enumerate(PREV_ZONE_COLUMNS, start=1):
        out[col] = _category(prev(df["zone_code"], k))

    swung = np.isin(df["result_code"], [list(PITCH_RESULTS).index(r) for r in SWINGS])
    out["prev_result"] = _category(prev(df["result_code"]))
    out["prev_swung"] = prev(np.where(df["result_code"] >= 0, swung, np.nan))
    out["prev_same_pa"] = (prev(df["at_bat_number"]) == df["at_bat_number"]).astype(np.int8)

    out["prev_speed"] = prev(df["release_speed"])
    out["prev_speed_vs_avg"] = prev(_speed_vs_type_average(df))
    out["prev_spin"] = prev(df["release_spin_rate"].astype("float32"))
    out["prev_spin_axis"] = prev(df["spin_axis"].astype("float32"))
    out["prev_extension"] = prev(df["release_extension"])
    out["prev_pfx_x"] = prev(df["pfx_x"])
    out["prev_pfx_z"] = prev(df["pfx_z"])
    out["prev_release_x"] = prev(df["release_pos_x"])
    out["prev_release_z"] = prev(df["release_pos_z"])
    out["prev_plate_x"] = prev(df["plate_x_rel"])
    out["prev_plate_z"] = prev(df["plate_z"])

    # events is filled only on a plate appearance's last pitch: carry the most
    # recent one forward to every later pitch of the game.
    pa_result = prev(df["pa_result_code"].where(df["pa_result_code"] >= 0))
    out["last_pa_result"] = _category(pa_result.groupby(game).ffill())

    out = pd.DataFrame(out)
    numeric = [c for c in out.columns if c not in CATEGORICAL_FEATURES and c != "prev_same_pa"]
    out[numeric] = out[numeric].astype("float32")
    return out


def at_bat(df: pd.DataFrame) -> pd.DataFrame:
    """Pitches so far this plate appearance, by family, and the same-type streak."""
    pa = group_ids(df, ["pitcher", "game_pk", "at_bat_number"])
    game = group_ids(df, ["pitcher", "game_pk"])
    by_family = prior_counts(one_hot(df["family_code"], len(FAMILIES)), pa)

    # Length of the run of one pitch type that ended with the previous pitch.
    type_code = pd.Series(df["type_code"].to_numpy())
    new_run = type_code != type_code.groupby(game).shift(1)
    run_id = new_run.groupby(game).cumsum()
    run_length = type_code.groupby([game, run_id]).cumcount() + 1

    return pd.DataFrame({
        "pa_pitch_idx": pd.Series(pa).groupby(pa).cumcount().to_numpy(np.int16),
        "pa_n_fastball": by_family[:, 0],
        "pa_n_breaking": by_family[:, 1],
        "pa_n_offspeed": by_family[:, 2],
        "streak_same_type": run_length.groupby(game).shift(1).astype("float32"),
    })


def workload(df: pd.DataFrame) -> pd.DataFrame:
    """Pitches already thrown this game, and whether the pitcher started it."""
    by_game = df.groupby(["pitcher", "game_pk"], sort=False)
    return pd.DataFrame({
        "game_pitch_count": by_game.cumcount().to_numpy(np.int16),
        "is_starter": (by_game["inning"].transform("first") == 1).to_numpy(np.int8),
    })


# --- history helpers: sums over each pitch's earlier pitches ------------------

def group_ids(df: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """One integer per distinct combination of ``keys``."""
    return df.groupby(keys, sort=False, observed=True).ngroup().to_numpy()


def one_hot(codes, k: int) -> np.ndarray:
    """``(n, k)`` indicators; negative codes (none) give all-zero rows."""
    codes = np.asarray(codes)
    out = np.zeros((len(codes), k), dtype=np.float32)
    has = codes >= 0
    out[np.flatnonzero(has), codes[has]] = 1.0
    return out


def prior_counts(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per row: sum of ``values`` over the earlier rows of its group."""
    inclusive = pd.DataFrame(values).groupby(groups, sort=False).cumsum().to_numpy(np.float32)
    return inclusive - values


def prior_window_counts(values: np.ndarray, groups: np.ndarray, window: int) -> np.ndarray:
    """Like :func:`prior_counts`, over only the ``window`` most recent earlier rows."""
    prior = prior_counts(values, groups)
    older = pd.DataFrame(prior).groupby(groups, sort=False).shift(window).fillna(0.0)
    return prior - older.to_numpy(np.float32)


def _speed_vs_type_average(df: pd.DataFrame) -> np.ndarray:
    """Speed minus the pitcher's average for that pitch type over his earlier pitches this season."""
    speed = df["release_speed"].to_numpy(np.float32)
    has = ~np.isnan(speed)
    groups = group_ids(df, ["pitcher", "type_code", "game_year"])
    sums = prior_counts(np.column_stack([np.where(has, speed, 0.0), has]).astype(np.float32), groups)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sums[:, 1] > 0, speed - sums[:, 0] / sums[:, 1], np.nan)


# --- mix helpers: shares of each class, shrunk toward a broader mix -----------

def shrink(counts: np.ndarray, broader: np.ndarray, pseudo_pitches: float = PSEUDO_PITCHES) -> np.ndarray:
    """Mix from ``counts`` plus ``pseudo_pitches`` of the ``broader`` mix. Rows sum to 1."""
    total = counts.sum(axis=1, keepdims=True)
    return ((counts + pseudo_pitches * broader) / (total + pseudo_pitches)).astype(np.float32)


def _slice_mixes(df: pd.DataFrame, onehot: np.ndarray, league: np.ndarray) -> dict[str, np.ndarray]:
    """The pitcher's mix of ``onehot`` for each slice in :data:`SLICES`."""
    pitcher = group_ids(df, ["pitcher"])
    career = shrink(prior_counts(onehot, pitcher), league)
    return {
        "career": career,
        "last200": shrink(prior_window_counts(onehot, pitcher, RECENT_PITCHES), career),
        "vs_hand": shrink(prior_counts(onehot, group_ids(df, ["pitcher", "is_LHB"])), career),
        "count": shrink(prior_counts(onehot, group_ids(df, ["pitcher", "balls", "strikes"])), career),
        "game": shrink(prior_counts(onehot, group_ids(df, ["pitcher", "game_pk"])), career),
    }


def _mix_before_each_day(day: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
    """For each day: class shares over all earlier days, plus one pseudo-pitch per class."""
    has = codes >= 0
    daily = np.bincount(day[has] * k + codes[has], minlength=(day.max() + 1) * k).reshape(-1, k)
    before = daily.cumsum(axis=0) - daily
    return ((before + 1.0) / (before.sum(axis=1, keepdims=True) + k)).astype(np.float32)


# --- column helpers -----------------------------------------------------------

def _mix_columns(prefix: str, mix: np.ndarray, labels: list) -> dict[str, np.ndarray]:
    """One ``{prefix}_{label}`` column per column of ``mix``."""
    return {f"{prefix}_{label}": mix[:, i] for i, label in enumerate(labels)}


def _category(codes) -> pd.Series:
    """Codes as a nullable column: negative or missing codes become missing."""
    codes = pd.Series(codes)
    return codes.where(codes >= 0).astype("Int8")


# --- command line -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean", type=Path, default=CLEAN_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--force", action="store_true", help="overwrite an existing feature table")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} already exists; pass --force to rebuild")

    start = time.time()
    rows = build(pd.read_parquet(args.clean), args.out)
    print(f"\nWrote {rows:,} rows x {len(FEATURE_COLUMNS)} features to {args.out} "
          f"({args.out.stat().st_size / 1e6:.0f} MB) in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
