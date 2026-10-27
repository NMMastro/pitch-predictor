"""Tabular features: clean pitch table -> one wide feature table for XGBoost.

Input:  data/clean/pitches.parquet (from clean.py).
Output: data/features/tabular.parquet, one row per pitch: rid, split, eligible,
        game_year, the labels pitch_class and zone_class, then every feature
        below. Rows are grouped by pitcher; join to the clean table on rid.

Every feature for a pitch uses only its pre-pitch state and earlier pitches.

Steps, in the order build() runs them:

    1. encode             every pitch as numbers (encode.py)

  Whole table: look across every pitcher, so computed once
    2. LeagueMix          league mix before each day; not a feature, the mix
                          that sparse mixes are shrunk toward
    3. batter_mix         how pitchers have attacked each batter

  Each pitcher (pitcher_features): need only the pitcher's own pitches, so run
  a batch of pitchers at a time to fit in memory
    4. game_state
    5. pitcher_types
    6. pitcher_zones
       batter_mix         looked up for these pitches by rid
    7. previous_pitches
    8. at_bat
    9. workload

   10. write              each batch into one parquet file as it is built

Each feature group below is a key of FEATURE_GROUPS and the function that builds it.

Features
--------
Categories are stored as integer codes (missing = none); see CATEGORICAL_FEATURES.

game_state: known before the pitch
  <pre-pitch columns>     schema.PRE_PITCH_COLUMNS as they are, except:
  inning                  capped at 10
  if/of_fielding_alignment  category
  is_LHB                  1 if the batter bats left
  is_LHP                  1 if the pitcher throws left
  same_hand               1 if batter and pitcher are the same hand
  is_inning_top           1 in the top of the inning

pitcher_types: share of each pitch type among the pitcher's earlier pitches
  p_type_{slice}_{class}  class is one of PITCH_CLASSES; slice is
                            career   all his earlier pitches
                            last200  his last 200 pitches
                            vs_hand  his earlier pitches to batters of this hand
                            count    his earlier pitches in this ball-strike count
                            game     his earlier pitches this game
  pitcher_prior_pitches   number of his earlier pitches, all seasons

pitcher_zones: the same, over zones as the batter sees them
  p_zone_{slice}_{zone}   zone is up_in, up_mid, up_away, mid_in, ... low_away,
                          out_up_in, out_up_away, out_low_in, out_low_away.
                          A left-handed batter's zones are mirrored, so "in" is
                          inside for everyone.

batter_mix: how pitchers have attacked this batter on earlier pitches
  b_type_{class}          share of each pitch type
  b_zone_{zone}           share of each zone
  batter_prior_pitches    number of earlier pitches he has seen

  Every mix above adds PSEUDO_PITCHES of a broader mix, so it is sensible
  with little history: career and batter mixes toward the league mix (all
  earlier days), every other slice toward the pitcher's career mix.

previous_pitches: this pitcher's earlier pitches this game (missing on his first)
  prev{k}_type            pitch type k pitches ago, k = 1..PREV_TYPE_LAGS (category)
  prev{k}_zone            zone k pitches ago, k = 1..PREV_ZONE_LAGS, batter's view (category)
  prev_result             last pitch's result (category): ball, called_strike,
                          swinging_strike, foul, in_play, hit_by_pitch
  prev_swung              1 if the batter swung at the last pitch
  prev_same_pa            1 if the last pitch was in this plate appearance
  prev_speed              last pitch's release speed (mph)
  prev_speed_vs_avg       that speed minus his average for that pitch type over
                          his earlier pitches this season
  prev_spin               spin rate (rpm)
  prev_spin_axis          spin axis (degrees)
  prev_extension          release extension (ft)
  prev_pfx_x, prev_pfx_z  horizontal and vertical movement (ft)
  prev_release_x, _z      release point (ft)
  prev_plate_x            horizontal location, batter's view (negative = inside)
  prev_plate_z            height at the plate (ft)
  last_pa_result          how his last finished plate appearance this game ended
                          (category): strikeout, walk, out, reached, single,
                          extra_base, home_run

at_bat: this plate appearance so far
  pa_pitch_idx            pitches already thrown in it
  pa_n_fastball           of those, fastballs (FF, SI, FC)
  pa_n_breaking           breaking balls (SL, CU, KC, SV)
  pa_n_offspeed           off-speed (CH, FS)
  streak_same_type        same pitch type in a row, ending with the last pitch

workload: this game so far
  game_pitch_count        pitches he has already thrown this game
  is_starter              1 if his first pitch of the game was in the 1st inning

Usage:
    python src/features/tabular_features.py
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.encode import encode  # noqa: E402
from features.schema import (  # noqa: E402
    CLEAN_PATH,
    ENCODED_PRE_PITCH_COLUMNS,
    FAMILIES,
    PITCH_CLASSES,
    PITCH_RESULTS,
    PITCHER_CHRONO_KEYS,
    SWINGS,
    ZONE_CLASSES,
    ZONE_LABELS,
)

OUT_PATH = Path("data/features/tabular.parquet")

PSEUDO_PITCHES = 20.0   # pitches of the broader mix added to every mix
RECENT_PITCHES = 200    # length of the "last 200" window
PREV_TYPE_LAGS = 5      # pitch types of the last 5 pitches
PREV_ZONE_LAGS = 3      # zones of the last 3 pitches
PITCHER_BATCHES = 8     # pitchers are processed in this many batches to fit in memory


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

def build(table: pd.DataFrame, out: Path = OUT_PATH, n_batches: int = PITCHER_BATCHES) -> int:
    """Clean table -> tabular feature table, written to ``out``. Returns the row count."""
    table = encode(table)

    # Whole table: these look across every pitcher, so they are computed once.
    league = LeagueMix.from_table(table)
    batter = batter_mix(table, league)

    # Each pitcher: these need only the pitcher's own pitches, so they run a
    # batch of pitchers at a time to fit in memory, and each batch is written
    # as soon as it is built.
    batches = pitcher_batches(table, n_batches)
    features = (pitcher_features(batch, batter, league) for batch in batches)
    return write(features, out)


# --- whole table: features that look across pitchers --------------------------

@dataclass
class LeagueMix:
    """League pitch-type and zone mix from every pitch on earlier dates.

    Not a feature: the mix that career and batter mixes are shrunk toward when
    they have little history. ``lookup`` gives each pitch its day's mix.
    """

    dates: np.ndarray
    types: np.ndarray  # (n_dates, n_types)
    zones: np.ndarray  # (n_dates, n_zones)

    @classmethod
    def from_table(cls, table: pd.DataFrame) -> LeagueMix:
        dates, day = np.unique(table["game_date"].to_numpy(), return_inverse=True)
        return cls(
            dates,
            _mix_before_each_day(day, table["type_code"].to_numpy(), len(PITCH_CLASSES)),
            _mix_before_each_day(day, table["zone_code"].to_numpy(), len(ZONE_CLASSES)),
        )

    def lookup(self, game_date: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        day = np.searchsorted(self.dates, game_date.to_numpy())
        return self.types[day], self.zones[day]


def batter_mix(table: pd.DataFrame, league: LeagueMix) -> pd.DataFrame:
    """How pitchers have attacked each batter so far, shrunk toward the league. Indexed by rid."""
    league_types, league_zones = league.lookup(table["game_date"])
    batter = table["batter"].to_numpy()
    type_counts = prior_counts(one_hot(table["type_code"], len(PITCH_CLASSES)), batter)
    zone_counts = prior_counts(one_hot(table["zone_code"], len(ZONE_CLASSES)), batter)

    columns = {
        **_mix_columns("b_type", shrink(type_counts, league_types), PITCH_CLASSES),
        **_mix_columns("b_zone", shrink(zone_counts, league_zones), ZONE_LABELS),
        "batter_prior_pitches": type_counts.sum(axis=1),
    }
    return pd.DataFrame(columns, index=table["rid"].to_numpy())


# --- each pitcher: features from the pitcher's own pitches --------------------

def pitcher_features(batch: pd.DataFrame, batter: pd.DataFrame, league: LeagueMix) -> pd.DataFrame:
    """Every feature group for one batch of pitchers, in pitcher-then-time order."""
    df = batch.sort_values(PITCHER_CHRONO_KEYS).reset_index(drop=True)

    features = pd.concat([
        game_state(df),
        pitcher_types(df, league),
        pitcher_zones(df, league),
        batter.loc[df["rid"]].reset_index(drop=True),  # batter_mix, already built

        previous_pitches(df),
        at_bat(df),
        workload(df),
    ], axis=1)

    if list(features.columns) != FEATURE_COLUMNS:
        raise AssertionError("feature columns do not match FEATURE_GROUPS")
    return pd.concat([df[META_COLUMNS], features], axis=1)


def game_state(df: pd.DataFrame) -> pd.DataFrame:
    """The pre-pitch columns and encode()'s flags, with fielding alignments as categories."""
    out = df[FEATURE_GROUPS["game_state"]].copy()
    for col in ["if_fielding_alignment", "of_fielding_alignment"]:
        out[col] = _category(out[col].cat.codes)
    return out


def pitcher_types(df: pd.DataFrame, league: LeagueMix) -> pd.DataFrame:
    """The pitcher's pitch-type mix over his earlier pitches, five ways."""
    league_types, _ = league.lookup(df["game_date"])
    mixes = _slice_mixes(df, one_hot(df["type_code"], len(PITCH_CLASSES)), league_types)

    columns = {}
    for name, mix in mixes.items():
        columns |= _mix_columns(f"p_type_{name}", mix, PITCH_CLASSES)
    columns["pitcher_prior_pitches"] = df["pitcher_prior_pitches"].to_numpy()
    return pd.DataFrame(columns)


def pitcher_zones(df: pd.DataFrame, league: LeagueMix) -> pd.DataFrame:
    """The pitcher's zone mix (batter's view) over his earlier pitches, five ways."""
    _, league_zones = league.lookup(df["game_date"])
    mixes = _slice_mixes(df, one_hot(df["zone_code"], len(ZONE_CLASSES)), league_zones)

    columns = {}
    for name, mix in mixes.items():
        columns |= _mix_columns(f"p_zone_{name}", mix, ZONE_LABELS)
    return pd.DataFrame(columns)


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
    type_code = df["type_code"].to_numpy()
    new_run = np.r_[True, (type_code[1:] != type_code[:-1]) | (game[1:] != game[:-1])]
    run_id = np.cumsum(new_run)
    run_length = pd.Series(run_id).groupby(run_id).cumcount() + 1

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


# --- batching and writing -----------------------------------------------------

def pitcher_batches(table: pd.DataFrame, n_batches: int) -> Iterator[pd.DataFrame]:
    """The table in ``n_batches`` batches, each holding whole pitchers."""
    for k in range(n_batches):
        yield table[table["pitcher"] % n_batches == k]


def write(parts: Iterable[pd.DataFrame], out: Path) -> int:
    """Write every part into one parquet file, via a temp file. Returns the row count."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    writer, rows = None, 0
    for k, part in enumerate(parts, start=1):
        arrow_part = pa.Table.from_pandas(part, preserve_index=False)
        writer = writer or pq.ParquetWriter(tmp, arrow_part.schema)
        writer.write_table(arrow_part)
        rows += len(part)
        print(f"  batch {k}: {len(part):,} rows")
    if writer is None:
        raise ValueError("no batches to write")
    writer.close()
    tmp.replace(out)
    return rows



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
    parser.add_argument("--batches", type=int, default=PITCHER_BATCHES, help="more = less memory")
    parser.add_argument("--force", action="store_true", help="overwrite an existing feature table")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} already exists; pass --force to rebuild")

    start = time.time()
    rows = build(pd.read_parquet(args.clean), args.out, args.batches)
    print(f"\nWrote {rows:,} rows x {len(FEATURE_COLUMNS)} features to {args.out} "
          f"({args.out.stat().st_size / 1e6:.0f} MB) in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
