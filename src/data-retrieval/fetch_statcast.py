"""Pull pitch-level Statcast data from Baseball Savant via pybaseball.

Statcast is queried in chunks (Baseball Savant caps each request at ~30k rows /
a handful of days), concatenated, and written to a single Parquet file.

Examples
--------
Pull a single day (quick smoke test):
    python src/data/fetch_statcast.py --start 2024-07-01 --end 2024-07-01

Pull a full regular season:
    python src/data/fetch_statcast.py --start 2024-03-28 --end 2024-09-29 \
        --out data/raw/statcast_2024.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pybaseball import statcast


# A compact set of columns we care about for the pitch-prediction models.
# Statcast returns ~90 columns; this is just for quick previews, the full frame
# is what gets written to disk.
PREVIEW_COLS = [
    "game_date",
    "player_name",
    "pitch_type",
    "plate_x",
    "plate_z",
    "balls",
    "strikes",
    "outs_when_up",
    "stand",
    "p_throws",
    "release_speed",
]


def fetch(start: str, end: str) -> pd.DataFrame:
    """Fetch all pitches thrown between ``start`` and ``end`` (inclusive).

    Dates are ``YYYY-MM-DD`` strings. pybaseball handles internal chunking and
    caches responses under ~/.pybaseball so re-runs are fast.
    """
    df = statcast(start_dt=start, end_dt=end)
    # Statcast returns pitches in reverse-chronological order; sort so that
    # pitch history is easy to build downstream.
    if not df.empty:
        df = df.sort_values(
            ["game_pk", "at_bat_number", "pitch_number"]
        ).reset_index(drop=True)
    return df


def save(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output Parquet path (default: data/raw/statcast_<start>_<end>.parquet)",
    )
    args = parser.parse_args()

    out = args.out or Path(
        f"data/raw/statcast_{args.start}_{args.end}.parquet"
    )

    print(f"Fetching Statcast pitches {args.start} -> {args.end} ...")
    df = fetch(args.start, args.end)

    if df.empty:
        print("No pitches returned for that date range (off-season?).")
        return

    print(f"Retrieved {len(df):,} pitches x {len(df.columns)} columns.")
    preview = [c for c in PREVIEW_COLS if c in df.columns]
    print(df[preview].head(5).to_string(index=False))

    save(df, out)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB).")


if __name__ == "__main__":
    main()
