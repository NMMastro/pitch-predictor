"""Extract one pitcher's pitches from the per-season Statcast files.

This builds a small single-pitcher dataset for a first-pass model, instead of
training on the full multi-season dump. It reads the combined
``data/raw/statcast_<year>.parquet`` files produced by ``fetch_seasons.py``,
keeps only the rows thrown by the requested pitcher, concatenates them across
seasons, and writes one Parquet file.

The pitcher is identified by MLBAM id (Statcast's ``pitcher`` column), looked up
from a name with ``pybaseball.playerid_lookup``. Pass ``--mlbam`` to skip the
lookup entirely.

Nothing here applies the DESIGN_DECISIONS.md filtering (spring training, non-pitch
types, pitch-count thresholds). This is purely a row-subset by pitcher; all game
types and all columns are preserved so the filtering can happen downstream, the
same way it would on the full dataset.

Examples
--------
Max Scherzer, all seasons found in data/raw::

    python src/data-retrieval/extract_pitcher.py --first Max --last Scherzer

By MLBAM id, specific seasons::

    python src/data-retrieval/extract_pitcher.py --mlbam 453286 --seasons 2016 2017
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def resolve_mlbam(first: str, last: str) -> int:
    """Look up a player's MLBAM id from first/last name.

    Raises if the name is ambiguous or not found, so we never silently extract
    the wrong player.
    """
    from pybaseball import playerid_lookup

    hits = playerid_lookup(last, first)
    hits = hits[hits["key_mlbam"].notna()]
    if hits.empty:
        raise SystemExit(f"No MLBAM id found for {first} {last!r}.")
    if len(hits) > 1:
        # Prefer the row with the most recent appearance, but tell the user.
        hits = hits.sort_values("mlb_played_last", ascending=False)
        options = ", ".join(
            f"{r.name_first} {r.name_last} ({int(r.key_mlbam)}, "
            f"{r.mlb_played_first:.0f}-{r.mlb_played_last:.0f})"
            for r in hits.itertuples()
        )
        print(f"Multiple matches, using the most recent. Candidates: {options}")
    return int(hits.iloc[0]["key_mlbam"])


def find_season_files(raw_dir: Path, seasons: list[int] | None) -> list[Path]:
    """Season Parquet paths to scan, in year order."""
    if seasons:
        paths = [raw_dir / f"statcast_{y}.parquet" for y in seasons]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise SystemExit(
                "Missing season files: " + ", ".join(str(p) for p in missing)
            )
        return paths
    return sorted(raw_dir.glob("statcast_[0-9][0-9][0-9][0-9].parquet"))


def extract(mlbam: int, season_files: list[Path]) -> pd.DataFrame:
    """Concatenate one pitcher's rows across the given season files."""
    frames: list[pd.DataFrame] = []
    for path in season_files:
        season = pd.read_parquet(path)
        if "pitcher" not in season.columns:
            raise SystemExit(f"{path} has no 'pitcher' column; is it a Statcast file?")
        mine = season[season["pitcher"] == mlbam]
        year = path.stem.split("_")[-1]
        print(f"  {year}  {len(mine):>7,} / {len(season):>9,} pitches")
        if not mine.empty:
            frames.append(mine)

    if not frames:
        raise SystemExit(
            f"Pitcher {mlbam} not found in any season file. "
            "Wrong id, or he didn't pitch in these seasons?"
        )

    df = pd.concat(frames, ignore_index=True)
    keys = [k for k in ("game_pk", "at_bat_number", "pitch_number") if k in df.columns]
    if keys:
        df = df.sort_values(keys).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", help="Pitcher first name (for name lookup)")
    parser.add_argument("--last", help="Pitcher last name (for name lookup)")
    parser.add_argument(
        "--mlbam",
        type=int,
        default=None,
        help="MLBAM id, skips the name lookup",
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help="Seasons to include (default: every statcast_<year>.parquet in --raw-dir)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Where the combined season files live (default: data/raw)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: data/pitchers/<mlbam>.parquet)",
    )
    args = parser.parse_args()

    if args.mlbam is not None:
        mlbam = args.mlbam
    elif args.first and args.last:
        mlbam = resolve_mlbam(args.first, args.last)
        print(f"Resolved {args.first} {args.last} -> MLBAM {mlbam}")
    else:
        raise SystemExit("Give either --mlbam, or both --first and --last.")

    season_files = find_season_files(args.raw_dir, args.seasons)
    print(f"Scanning {len(season_files)} season file(s) for pitcher {mlbam}:")
    df = extract(mlbam, season_files)

    out = args.out or Path("data/pitchers") / f"{mlbam}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    name = df["player_name"].iloc[0] if "player_name" in df.columns else str(mlbam)
    print(f"\n=== {name} (MLBAM {mlbam}) ===")
    print(f"  rows        {len(df):,}")
    print(f"  columns     {len(df.columns)}")
    if "game_date" in df.columns:
        dates = pd.to_datetime(df["game_date"])
        print(f"  date range  {dates.min().date()} -> {dates.max().date()}")
    if "game_type" in df.columns:
        counts = df["game_type"].value_counts().sort_index()
        print("  game_type   " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))
    if "pitch_type" in df.columns:
        top = df["pitch_type"].value_counts().head(12)
        print("  pitch_type  " + ", ".join(f"{k}={v:,}" for k, v in top.items()))
    print(f"  file        {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
