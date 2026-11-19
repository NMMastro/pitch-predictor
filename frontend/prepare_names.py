"""Cache 2025 batter names so replay can run offline."""

import argparse
import json
from pathlib import Path

import pandas as pd
from pybaseball import playerid_reverse_lookup

from features.schema import CLEAN_PATH

from frontend.replay import NAME_CACHE, REPLAY_FILTERS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=CLEAN_PATH)
    parser.add_argument("--output", type=Path, default=NAME_CACHE)
    args = parser.parse_args()

    ids = pd.read_parquet(args.clean, columns=["batter"], filters=REPLAY_FILTERS)["batter"].unique()
    players = playerid_reverse_lookup(ids.tolist(), key_type="mlbam")
    valid = players.dropna(subset=["name_first", "name_last"])
    names = {
        int(player.key_mlbam): f"{player.name_first.title()} {player.name_last.title()}"
        for player in valid.itertuples(index=False)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(names, ensure_ascii=False, indent=2) + "\n")
    print(f"Cached {len(names):,} of {len(ids):,} batter names in {args.output}.")


if __name__ == "__main__":
    main()
