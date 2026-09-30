"""Bulk-pull multiple full seasons of pitch-level Statcast data.

Why this exists instead of one big ``statcast(start, end)`` call
---------------------------------------------------------------
``pybaseball.statcast`` already splits a date range into small sub-requests to
stay under Baseball Savant's ~30k-row-per-query cap, and it caches raw
responses under ``.pybaseball/`` when the cache is enabled. What it does *not*
do is checkpoint the assembled result: a single call spanning four seasons
holds every row in RAM and returns nothing until the last sub-request lands, so
one unhandled network error late in the run throws away the concatenation work.

So this script owns the chunking at a coarser grain:

* one Savant pull per **calendar month**, each written straight to Parquet
* months already on disk are skipped, making the whole job resumable
* transient failures are retried with exponential backoff
* months are only concatenated into a season file once all of them exist

A month that legitimately has no pitches (January, December) gets a ``.empty``
marker file so resumed runs don't re-request it.

The pull covers the **full calendar year**. That deliberately includes spring
training, which ``DESIGN_DECISIONS.md`` excludes from training: filtering on
``game_type`` is cheap and reversible, re-downloading a season is not. Nothing
is dropped at this layer -- all 119 Statcast columns and all game types land on
disk as returned.

Known upstream limitation: spring training is only partially retrievable
-------------------------------------------------------------------------
``pybaseball.utils.statcast_date_range`` clamps every request to a season
window. For years present in its hardcoded ``STATCAST_VALID_DATES`` table it
uses the real dates, but that table ends at 2020; for 2021 onward it falls back
to ``Mar 15 - Nov 15``. Requests for January, February and Mar 1-14 therefore
return zero rows regardless of whether games were played, which is why the
February chunks below are always marked empty even though spring training
starts in late February.

This does not affect the data the project actually uses. Every regular season
from 2022 on opens on or after Mar 18 (Seoul 2024, Tokyo 2025), every
postseason ends by Nov 5, and both fall inside the clamp. The only loss is part
of spring training, which is discarded by ``game_type == 'S'`` anyway. If full
spring training is ever needed, add the real season bounds to that table.

Examples
--------
Pull the four seasons the project is scoped to::

    python src/data-retrieval/fetch_seasons.py --seasons 2022 2023 2024 2025

Resume an interrupted run (same command -- finished months are skipped)::

    python src/data-retrieval/fetch_seasons.py --seasons 2022 2023 2024 2025

Re-download one season from scratch::

    python src/data-retrieval/fetch_seasons.py --seasons 2024 --force
"""

from __future__ import annotations

import argparse
import calendar
import time
from pathlib import Path

import pandas as pd
import pybaseball as pyb
from pybaseball import statcast

# Order that makes a pitch sequence readable: game, then plate appearance,
# then pitch within the plate appearance. Statcast hands rows back in
# reverse-chronological order.
SORT_KEYS = ["game_pk", "at_bat_number", "pitch_number"]

DEFAULT_SEASONS = [2022, 2023, 2024, 2025]


def month_bounds(year: int, month: int) -> tuple[str, str]:
    """First and last day of ``month`` as ``YYYY-MM-DD`` strings."""
    last = calendar.monthrange(year, month)[1]
    return f"{year}-{month:02d}-01", f"{year}-{month:02d}-{last:02d}"


def chunk_paths(chunk_dir: Path, year: int, month: int) -> tuple[Path, Path]:
    """Parquet path and empty-month marker path for one month."""
    stem = f"statcast_{year}-{month:02d}"
    return chunk_dir / f"{stem}.parquet", chunk_dir / f"{stem}.empty"


def fetch_month(year: int, month: int, retries: int, backoff: float) -> pd.DataFrame:
    """Pull one calendar month of pitches, retrying transient failures.

    Returns an empty frame for months with no games. Raises if every attempt
    fails, so the caller can stop rather than silently write a hole into the
    dataset.
    """
    start, end = month_bounds(year, month)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            return statcast(start_dt=start, end_dt=end, verbose=False)
        except Exception as err:  # noqa: BLE001 - network/parse errors are all retryable
            last_error = err
            if attempt == retries:
                break
            wait = backoff * (2 ** (attempt - 1))
            print(
                f"    attempt {attempt}/{retries} failed ({type(err).__name__}: {err}); "
                f"retrying in {wait:.0f}s"
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Could not fetch {start} -> {end} after {retries} attempts"
    ) from last_error


def ensure_month(
    year: int,
    month: int,
    chunk_dir: Path,
    retries: int,
    backoff: float,
) -> None:
    """Make sure one month is on disk, downloading it only if it isn't."""
    parquet, marker = chunk_paths(chunk_dir, year, month)

    if parquet.exists():
        print(f"  {year}-{month:02d}  cached ({parquet.stat().st_size / 1e6:.1f} MB)")
        return
    if marker.exists():
        print(f"  {year}-{month:02d}  cached (no games)")
        return

    print(f"  {year}-{month:02d}  fetching ...")
    df = fetch_month(year, month, retries=retries, backoff=backoff)

    if df.empty:
        marker.touch()
        print(f"  {year}-{month:02d}  no games")
        return

    # Write to a temp file first: a crash mid-write would otherwise leave a
    # truncated Parquet that the resume logic would happily treat as complete.
    tmp = parquet.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(parquet)
    print(f"  {year}-{month:02d}  {len(df):>7,} pitches -> {parquet.name}")


def combine_season(year: int, chunk_dir: Path, out_dir: Path) -> Path | None:
    """Concatenate a season's monthly chunks into one sorted Parquet file."""
    months = sorted(chunk_dir.glob(f"statcast_{year}-*.parquet"))
    if not months:
        print(f"  {year}: no monthly chunks found, nothing to combine")
        return None

    frames = [pd.read_parquet(p) for p in months]
    season = pd.concat(frames, ignore_index=True)

    keys = [k for k in SORT_KEYS if k in season.columns]
    if keys:
        season = season.sort_values(keys).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"statcast_{year}.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    season.to_parquet(tmp, index=False)
    tmp.replace(out)

    summarise(year, season, out)
    return out


def summarise(year: int, season: pd.DataFrame, out: Path) -> None:
    """Print the sanity checks worth eyeballing before trusting a season file."""
    size_mb = out.stat().st_size / 1e6
    print(f"\n  === {year} ===")
    print(f"  rows            {len(season):,}")
    print(f"  columns         {len(season.columns)}")
    print(f"  file            {out}  ({size_mb:.1f} MB)")

    if "game_date" in season.columns:
        dates = pd.to_datetime(season["game_date"])
        print(f"  date range      {dates.min().date()} -> {dates.max().date()}")
    if "game_pk" in season.columns:
        print(f"  games           {season['game_pk'].nunique():,}")
    if "game_type" in season.columns:
        counts = season["game_type"].value_counts().sort_index()
        breakdown = ", ".join(f"{k}={v:,}" for k, v in counts.items())
        print(f"  game_type       {breakdown}")
    if "pitch_type" in season.columns:
        null_pct = season["pitch_type"].isna().mean() * 100
        print(f"  pitch_type null {null_pct:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=DEFAULT_SEASONS,
        help=f"Season years to pull (default: {' '.join(map(str, DEFAULT_SEASONS))})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/raw"),
        help="Where combined per-season Parquet files go (default: data/raw)",
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=Path("data/raw/chunks"),
        help="Where monthly chunks are checkpointed (default: data/raw/chunks)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Attempts per month before giving up (default: 4)",
    )
    parser.add_argument(
        "--backoff",
        type=float,
        default=15.0,
        help="Seconds before the first retry, doubling each time (default: 15)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing chunks for the requested seasons and re-download",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't enable pybaseball's on-disk request cache",
    )
    args = parser.parse_args()

    if not args.no_cache:
        # Caches raw Savant responses, so a retry or a re-run after a crash
        # doesn't re-hit the server for days already scraped.
        pyb.cache.enable()

    args.chunk_dir.mkdir(parents=True, exist_ok=True)

    if args.force:
        for year in args.seasons:
            for stale in list(args.chunk_dir.glob(f"statcast_{year}-*")):
                stale.unlink()
            print(f"Cleared existing chunks for {year}")

    written: list[Path] = []
    for year in args.seasons:
        print(f"\nSeason {year}: pulling {year}-01-01 -> {year}-12-31 by month")
        for month in range(1, 13):
            ensure_month(
                year,
                month,
                chunk_dir=args.chunk_dir,
                retries=args.retries,
                backoff=args.backoff,
            )
        out = combine_season(year, args.chunk_dir, args.out_dir)
        if out is not None:
            written.append(out)

    print("\nDone. Season files written:")
    for path in written:
        print(f"  {path}  ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
