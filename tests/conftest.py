"""Shared test setup: the import path, and a fake batch for the model tests.

``src`` is not an installed package, so it goes on ``sys.path`` here once
instead of at the top of every test file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# --- a clean table that needs no download -------------------------------------
# data/clean/pitches.parquet is 291 MB and takes a minute to build, so the
# pipeline and dataset tests run against this instead. Same columns, same
# dtypes, 80 rows, and the expected values stay checkable by hand.

SYNTHETIC_PITCHERS = [100, 200]
SYNTHETIC_PITCHES_EACH = 40
SYNTHETIC_MIN_PRIOR = 5  # stands in for schema.MIN_PRIOR_PITCHES, which is 25


def _synthetic_table() -> pd.DataFrame:
    """A clean-table-shaped frame: every column build() reads, nothing else.

    Two pitchers, interleaved in time, so a window that ran past a pitcher
    boundary or into the future would pick up a visibly wrong value. Pitch types
    cycle with a different period per pitcher, so ``type_code`` acts as a
    fingerprint of which row a sequence slot came from.
    """
    from features.schema import PITCH_CLASSES, SPLITS

    n = SYNTHETIC_PITCHES_EACH
    rows = []
    for p_index, pitcher in enumerate(SYNTHETIC_PITCHERS):
        for i in range(n):
            rows.append({
                # interleaved: pitcher 100 takes the even rids, 200 the odd
                "rid": i * len(SYNTHETIC_PITCHERS) + p_index,
                "pitcher": pitcher,
                "batter": 500 + (i % 7),
                "fielder_2": 900 + p_index,
                "game_pk": 1 + i // 10,
                "at_bat_number": 1 + i // 3,
                "pitch_number": 1 + i % 3,
                "pitch_class": PITCH_CLASSES[(i * (p_index + 1)) % len(PITCH_CLASSES)],
                "zone_class": 1 + (i % 9),
                "description": ["ball", "called_strike", "foul", "hit_into_play"][i % 4],
                "events": None,
                "release_speed": 85.0 + i % 10,
                "plate_x": -1.0 + 0.05 * (i % 20),
                "plate_z": 1.5 + 0.05 * (i % 20),
                "stand": "L" if i % 2 else "R",
                "p_throws": "R",
                "inning": 1 + i % 12,  # above INNING_CAP, to exercise the clip
                "inning_topbot": "Top" if p_index else "Bot",
                "balls": i % 4,
                "strikes": i % 3,
                "outs_when_up": i % 3,
                "on_1b": i % 2,
                "on_2b": 0,
                "on_3b": i % 3 == 0,
                "bat_score": i % 5,
                "fld_score": i % 4,
                "bat_score_diff": i % 5 - i % 4,
                "home_score_diff": i % 3,
                "sz_top": 3.3 + 0.01 * (i % 5),
                "sz_bot": 1.5 + 0.01 * (i % 5),
                "n_thruorder_pitcher": 1 + i // 20,
                "n_priorpa_thisgame_player_at_bat": i % 3,
                # null on each pitcher's first pitch: the nullable path
                "pitcher_days_since_prev_game": None if i == 0 else 4 + i % 3,
                "batter_days_since_prev_game": None if i == 0 else 1 + i % 2,
                "age_pit": 30,
                "age_bat": 27,
                "if_fielding_alignment": ["Standard", "Infield shift", "Strategic"][i % 3],
                "of_fielding_alignment": ["Standard", "Strategic"][i % 2],
                # three seasons -> one per split, so the scaler has train rows
                "game_year": [2016, 2024, 2025][min(i // (n // 3 + 1), 2)],
            })

    table = pd.DataFrame(rows).sort_values("rid").reset_index(drop=True)
    table["pitch_class"] = pd.Categorical(table["pitch_class"], categories=PITCH_CLASSES)
    for col in ("description", "if_fielding_alignment", "of_fielding_alignment"):
        table[col] = table[col].astype("category")
    table["zone_class"] = table["zone_class"].astype("Int64")
    for col in ("pitcher_days_since_prev_game", "batter_days_since_prev_game"):
        table[col] = table[col].astype("Int16")
    table["split"] = pd.Categorical(
        table["game_year"].map(SPLITS).fillna("other"),
        categories=["train", "val", "test", "other"],
    )
    table["eligible"] = (
        table.groupby("pitcher", sort=False).cumcount() >= SYNTHETIC_MIN_PRIOR
    )
    return table


@pytest.fixture(scope="session")
def synthetic_clean_path(tmp_path_factory) -> str:
    """The synthetic table on disk, so the parquet path is exercised too."""
    path = tmp_path_factory.mktemp("clean") / "pitches.parquet"
    _synthetic_table().to_parquet(path, index=False)
    return str(path)


# --- a batch that needs no data -----------------------------------------------
# The model tests should fail because the model is wrong, not because
# data/clean/pitches.parquet has not been built. These shapes match what
# dataset.py produces, so a model that passes here will run on the real thing.

FAKE_VOCAB_SIZES: dict[str, int] = {
    "type_code": 12,
    "zone_code": 15,
    "result_code": 8,
    "batter": 50,
    "pitcher": 3,
    "fielder_2": 10,
    "if_align_code": 6,
    "of_align_code": 6,
    "inning_code": 12,
}

FAKE_BATCH_SIZE = 8
FAKE_WINDOW = 6
FAKE_N_CLASSES = 10


@pytest.fixture
def vocab_sizes() -> dict[str, int]:
    return dict(FAKE_VOCAB_SIZES)


@pytest.fixture
def fake_batch(vocab_sizes) -> dict:
    """One batch of random-but-valid tensors, with realistic left padding.

    Keys and shapes match ``src/model/dataset.py``:
        seq_cat  (B, W, 3) int64     seq_num  (B, W, 3) float32
        pad_mask (B, W)    bool      ctx_cat  (B, 6)    int64
        ctx_num  (B, C)    float32   y        (B,)      int64

    Row i is padded in its first i slots, so the batch covers "no history at
    all" through "a full window" -- including the all-padding row that trips up
    a model which pools over the sequence instead of using the context token.
    """
    import torch

    from model.dataset import CTX_CATEGORICAL, CTX_NUMERIC, SEQ_CATEGORICAL

    rng = np.random.default_rng(0)
    b, w = FAKE_BATCH_SIZE, FAKE_WINDOW

    pad_mask = np.arange(w)[None, :] < np.arange(b)[:, None].clip(max=w)
    pad_mask = pad_mask[:, ::-1].copy()  # padding on the left, as the builder emits it

    seq_cat = np.stack([
        rng.integers(2, vocab_sizes[col], size=(b, w)) for col in SEQ_CATEGORICAL
    ], axis=-1)
    seq_cat[pad_mask] = 0

    seq_num = rng.normal(size=(b, w, 3)).astype(np.float32)
    seq_num[pad_mask] = 0.0

    ctx_cat = np.stack([
        rng.integers(2, vocab_sizes[col], size=b) for col in CTX_CATEGORICAL
    ], axis=-1)

    return {
        "seq_cat": torch.from_numpy(seq_cat),
        "seq_num": torch.from_numpy(seq_num),
        "pad_mask": torch.from_numpy(pad_mask),
        "ctx_cat": torch.from_numpy(ctx_cat),
        "ctx_num": torch.from_numpy(rng.normal(size=(b, len(CTX_NUMERIC))).astype(np.float32)),
        "y": torch.from_numpy(rng.integers(0, FAKE_N_CLASSES, size=b)),
    }
