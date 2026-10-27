"""Clean step: raw Statcast season files -> one clean pitch table.

Input:  data/raw/statcast_YYYY.parquet, one file per season (from fetch_seasons.py).
Output: data/clean/pitches.parquet, one row per real pitch in real-time order.

Steps, in the order clean() runs them:

  Each season
    1. read_season           keep schema.CLEAN_COLUMNS; fail if one is missing
    2. prepare               drop spring training and non-pitches;
                             pitch_type, zone -> labels pitch_class, zone_class
    3. runners_to_flags      runner ids of players on base -> 1 occupied / 0 empty
    4. shrink_dtypes         compact types

  All seasons
    5. join_seasons          one table
    6. check_unique_pitches  fail on duplicate pitches (doesn't occur in our data years but is a precaution)
    7. sort_by_time          real-time order; number the rows (rid)
    8. add_split             tag train / val / test; rows stay in one table
    9. add_eligibility       pitcher's prior pitches; eligible = 25+; nothing dropped
   10. write                 parquet

Usage:
    python src/features/clean.py                             # every season in data/raw
    python src/features/clean.py --seasons 2024 2025 --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.schema import (  # noqa: E402
    CHRONO_KEYS,
    CLEAN_COLUMNS,
    CLEAN_PATH,
    DROP_CODES,
    KEEP_GAME_TYPES,
    MERGE_CODES,
    MIN_PRIOR_PITCHES,
    OTHER_CODES,
    PITCH_CLASSES,
    PITCH_KEY,
    SPLITS,
)

RAW_DIR = Path("data/raw")

# Never-null columns get plain types, so a surprise null fails here.
DTYPES: dict[str, str] = {
    # ids
    "game_pk": "int32",
    "game_year": "int16",
    "game_type": "category",
    "at_bat_number": "int16",
    "pitch_number": "int16",
    "pitcher": "int32",
    "batter": "int32",
    "player_name": "category",
    "fielder_2": "int32",
    "fielder_3": "int32",
    "fielder_4": "int32",
    "fielder_5": "int32",
    "fielder_6": "int32",
    "fielder_7": "int32",
    "fielder_8": "int32",
    "fielder_9": "int32",
    "home_team": "category",
    "away_team": "category",
    # before the pitch
    "balls": "int8",
    "strikes": "int8",
    "outs_when_up": "int8",
    "inning": "int8",
    "inning_topbot": "category",
    "stand": "category",
    "p_throws": "category",
    "on_1b": "int8",
    "on_2b": "int8",
    "on_3b": "int8",
    "bat_score": "int16",
    "fld_score": "int16",
    "bat_score_diff": "int16",
    "home_score_diff": "int16",
    "sz_top": "float32",
    "sz_bot": "float32",
    "n_thruorder_pitcher": "int8",
    "n_priorpa_thisgame_player_at_bat": "int8",
    "pitcher_days_since_prev_game": "Int16",
    "batter_days_since_prev_game": "Int16",
    "age_pit": "int8",
    "age_bat": "int8",
    "if_fielding_alignment": "category",
    "of_fielding_alignment": "category",
    # the pitch
    "plate_x": "float32",
    "plate_z": "float32",
    "description": "category",
    "release_speed": "float32",
    "release_spin_rate": "Int16",
    "spin_axis": "Int16",
    "release_extension": "float32",
    "pfx_x": "float32",
    "pfx_z": "float32",
    "release_pos_x": "float32",
    "release_pos_z": "float32",
    # after the pitch
    "events": "category",
    "bb_type": "category",
    "launch_speed": "float32",
    "launch_angle": "Int8",
    "hit_distance_sc": "Int16",
    "hit_location": "Int8",
    "hc_x": "float32",
    "hc_y": "float32",
    # label
    "zone_class": "Int8",
}
CATEGORY_COLUMNS = [col for col, dtype in DTYPES.items() if dtype == "category"]


def clean(season_files: list[Path], out: Path = CLEAN_PATH) -> pd.DataFrame:
    """Raw season files -> one clean pitch table, written to ``out``."""
    seasons = [clean_season(path) for path in season_files]

    table = join_seasons(seasons)
    check_unique_pitches(table)
    table = sort_by_time(table)
    table = add_split(table)
    table = add_eligibility(table)
    write(table, out)
    return table


def clean_season(path: Path) -> pd.DataFrame:
    """One raw season file -> its clean rows. One season in memory at a time."""
    raw = read_season(path)

    df = prepare(raw)
    df = runners_to_flags(df)
    df = shrink_dtypes(df)

    print(f"  {path.name}: {len(raw):,} raw -> {len(df):,} kept")
    return df


# --- each season --------------------------------------------------------------

def read_season(path: Path) -> pd.DataFrame:
    """The :data:`CLEAN_COLUMNS` of one raw season file."""
    available = set(pq.ParquetFile(path).schema.names)
    missing = [col for col in CLEAN_COLUMNS if col not in available]
    if missing:
        raise KeyError(f"{path.name} is missing columns: {missing}")
    return pd.read_parquet(path, columns=CLEAN_COLUMNS)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Keep real regular-season and postseason pitches; replace the raw codes with the labels."""
    df = df[df["game_type"].isin(KEEP_GAME_TYPES)].copy()
    df["pitch_class"] = map_pitch_classes(df["pitch_type"])
    df = df.dropna(subset=["pitch_class"])  # non-pitches and missing pitch types
    if "zone" in df.columns:
        df["zone_class"] = df["zone"].astype("Int64")
    return df.drop(columns=["pitch_type", "zone"], errors="ignore")


def map_pitch_classes(pitch_type: pd.Series) -> pd.Series:
    """Raw ``pitch_type`` -> :data:`PITCH_CLASSES`. Non-pitches and nulls become NaN."""
    codes = pitch_type.astype("object").replace(MERGE_CODES)
    codes = codes.where(~codes.isin(OTHER_CODES), "OTHER")
    codes = codes.where(~codes.isin(DROP_CODES), np.nan)

    unknown = set(codes.dropna()) - set(PITCH_CLASSES)
    if unknown:
        raise ValueError(
            f"pitch_type codes not covered by the taxonomy: {sorted(unknown)}. "
            "Add them to PITCH_CLASSES, MERGE_CODES, OTHER_CODES or DROP_CODES."
        )
    return codes


def runners_to_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Runner ids -> 1 (occupied) / 0 (empty)."""
    for base in ("on_1b", "on_2b", "on_3b"):
        df[base] = df[base].notna()
    return df


def shrink_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :data:`DTYPES`; dates become day-precision timestamps."""
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df["pitch_class"] = pd.Categorical(df["pitch_class"], categories=PITCH_CLASSES)
    return df.astype(DTYPES).reset_index(drop=True)


# --- all seasons --------------------------------------------------------------

def join_seasons(seasons: list[pd.DataFrame]) -> pd.DataFrame:
    """One table. Categories are rebuilt because each season has its own set."""
    table = pd.concat(seasons, ignore_index=True)
    return table.astype({col: "category" for col in CATEGORY_COLUMNS})


def check_unique_pitches(table: pd.DataFrame) -> None:
    """Fail if two rows are the same pitch."""
    dupes = int(table.duplicated(PITCH_KEY).sum())
    if dupes:
        raise ValueError(f"{dupes} rows share {PITCH_KEY}; the input has duplicate pitches")


def sort_by_time(table: pd.DataFrame) -> pd.DataFrame:
    """Real-time order, with rows numbered 0..n-1 in ``rid``."""
    table = table.sort_values(CHRONO_KEYS, kind="stable").reset_index(drop=True)
    table.insert(0, "rid", np.arange(len(table), dtype=np.int64))
    return table


def add_split(table: pd.DataFrame) -> pd.DataFrame:
    """``split`` = train / val / test by season; ``other`` for seasons outside :data:`SPLITS`."""
    names = [*dict.fromkeys(SPLITS.values()), "other"]
    table["split"] = pd.Categorical(table["game_year"].map(SPLITS).fillna("other"), categories=names)
    return table


def add_eligibility(table: pd.DataFrame) -> pd.DataFrame:
    """``pitcher_prior_pitches`` across seasons; ``eligible`` once there are :data:`MIN_PRIOR_PITCHES`."""
    prior = table.groupby("pitcher", sort=False).cumcount()
    table["pitcher_prior_pitches"] = prior.astype(np.int32)
    table["eligible"] = prior >= MIN_PRIOR_PITCHES
    return table


def write(table: pd.DataFrame, out: Path) -> None:
    """Write the table via a temp file, so a crash never leaves half a table."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    table.to_parquet(tmp, index=False)
    tmp.replace(out)


# --- command line -------------------------------------------------------------

def find_season_files(raw_dir: Path, seasons: list[int] | None) -> list[Path]:
    """Season files in year order: the requested ``seasons``, or every one in ``raw_dir``."""
    if seasons:
        paths = [raw_dir / f"statcast_{year}.parquet" for year in sorted(seasons)]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            sys.exit(f"Missing season files: {', '.join(missing)}")
        return paths
    paths = sorted(raw_dir.glob("statcast_[0-9][0-9][0-9][0-9].parquet"))
    if not paths:
        sys.exit(f"No statcast_<year>.parquet files in {raw_dir}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", type=int, nargs="+", help="default: every season in --raw-dir")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=CLEAN_PATH)
    parser.add_argument("--force", action="store_true", help="overwrite an existing clean table")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        sys.exit(f"{args.out} already exists; pass --force to rebuild")
    season_files = find_season_files(args.raw_dir, args.seasons)

    start = time.time()
    print(f"Cleaning {len(season_files)} seasons")
    table = clean(season_files, args.out)

    print(f"\nWrote {len(table):,} rows x {len(table.columns)} columns to {args.out} "
          f"({args.out.stat().st_size / 1e6:.0f} MB) in {time.time() - start:.0f}s")
    print(table.groupby("split", observed=True).agg(rows=("rid", "size"), eligible=("eligible", "sum")).to_string())


if __name__ == "__main__":
    main()
