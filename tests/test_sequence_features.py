"""Tests for the transformer's input arrays.

The pipeline's job is to hand the model history and nothing else, so these
tests are mostly leakage tests: every sequence slot must hold a pitch that
really did come earlier, from the same pitcher. The rest guard the mistakes
that would surface much later as a NaN loss or an nn.Embedding crash.

A synthetic clean table is used rather than data/clean/pitches.parquet, so the
tests run without the 291 MB build and the expected values stay checkable by
hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import SYNTHETIC_PITCHERS as PITCHERS
from features.schema import CODE_SHIFT, SPLITS
from features.sequence_features import (
    CTX_CATEGORICAL,
    CTX_NUMERIC,
    SEQ_CATEGORICAL,
    SEQ_NUMERIC,
    _gather_index,
    build_dataset,
)

WINDOW = 5


@pytest.fixture(scope="module")
def clean_path(synthetic_clean_path) -> str:
    return synthetic_clean_path


@pytest.fixture(scope="module")
def data(clean_path) -> dict:
    return build_dataset(clean_path, pitcher=None, window=WINDOW, verbose=False)


# --- the leakage rule ---------------------------------------------------------

def test_gather_index_only_looks_backward_within_one_pitcher():
    """Section 4: a pitch may use earlier pitches, and nothing else."""
    table = pd.DataFrame({"pitcher": [0] * 4 + [1] * 3})
    rows, pad = _gather_index(table, window=3)

    # Pitcher 0's rows are 0..3 and pitcher 1's are 4..6.
    expected_pad = np.array([
        [True, True, True],      # row 0: no history at all
        [True, True, False],     # row 1: one earlier pitch
        [True, False, False],
        [False, False, False],   # row 3: a full window
        [True, True, True],      # row 4: pitcher 1's first pitch
        [True, True, False],
        [True, False, False],
    ])
    assert pad.tolist() == expected_pad.tolist()

    # Where a slot is real it must point strictly before its own row...
    own_row = np.arange(len(table))[:, None]
    assert (rows[~pad] < np.broadcast_to(own_row, rows.shape)[~pad]).all()
    # ...and at the same pitcher.
    pitcher = table["pitcher"].to_numpy()
    assert (pitcher[rows][~pad] == np.broadcast_to(pitcher[:, None], rows.shape)[~pad]).all()


def test_newest_slot_is_the_previous_pitch(data):
    """Slot W-1 holds pitch t-1, slot 0 the oldest."""
    table = data["table"]
    type_code = table["type_code"].to_numpy() + CODE_SHIFT
    position = table.groupby("pitcher", sort=False).cumcount().to_numpy()
    newest = data["seq_cat"][:, -1, SEQ_CATEGORICAL.index("type_code")]

    has_history = position > 0
    # The table is sorted by (pitcher, rid), so the previous row of a pitcher
    # with history is the previous table row.
    previous = np.roll(type_code, 1)
    assert (newest[has_history] == previous[has_history]).all()
    assert not data["pad_mask"][has_history, -1].any()


def test_window_holds_the_pitchers_own_history_in_order(data):
    """Every real slot of every row, checked against the table it came from."""
    table = data["table"]
    type_code = table["type_code"].to_numpy() + CODE_SHIFT
    pitcher = table["pitcher"].to_numpy()
    position = table.groupby("pitcher", sort=False).cumcount().to_numpy()
    col = SEQ_CATEGORICAL.index("type_code")

    for t in range(len(table)):
        for k in range(WINDOW):
            age = WINDOW - k   # 1 = previous pitch, WINDOW = oldest
            # The table is sorted by (pitcher, rid), so a pitcher's rows are
            # contiguous: their pitch `age` ago sits `age` rows back.
            source = t - age
            if position[t] - age < 0:
                assert data["pad_mask"][t, k], f"row {t} slot {k} should be padding"
                continue
            assert not data["pad_mask"][t, k]
            assert pitcher[source] == pitcher[t]
            assert data["seq_cat"][t, k, col] == type_code[source]


def test_padded_slots_hold_the_pad_index(data):
    """Padding must be distinguishable from a real value, not just ignored."""
    pad = data["pad_mask"]
    assert pad.any(), "the fixture should produce some padding"
    assert (data["seq_cat"][pad] == 0).all()
    assert (data["seq_num"][pad] == 0.0).all()
    # ...and a real slot never uses the pad index, or the mask would be a lie.
    assert (data["seq_cat"][~pad] >= CODE_SHIFT - 1).all()


def test_labels_line_up_with_the_table(data):
    table = data["table"]
    assert data["y"].tolist() == table["type_code"].tolist()
    assert (data["y"] == table["pitch_class"].cat.codes.to_numpy()).all()


# --- shapes, dtypes and the things that break training ------------------------

def test_shapes_and_dtypes(data):
    n = len(data["table"])
    assert data["seq_cat"].shape == (n, WINDOW, len(SEQ_CATEGORICAL))
    assert data["seq_num"].shape == (n, WINDOW, len(SEQ_NUMERIC))
    assert data["pad_mask"].shape == (n, WINDOW)
    assert data["ctx_cat"].shape == (n, len(CTX_CATEGORICAL))
    assert data["ctx_num"].shape == (n, len(CTX_NUMERIC))
    assert data["y"].shape == (n,)

    assert data["seq_cat"].dtype == np.int64
    assert data["ctx_cat"].dtype == np.int64
    assert data["y"].dtype == np.int64
    assert data["seq_num"].dtype == np.float32
    assert data["ctx_num"].dtype == np.float32
    assert data["pad_mask"].dtype == bool


def test_no_nans_anywhere(data):
    """One NaN anywhere makes the loss NaN, and it is invisible until it does."""
    for name in ("seq_num", "ctx_num"):
        assert np.isfinite(data[name]).all(), f"{name} has NaN or inf"


def test_codes_fit_their_embedding_tables(data):
    """Every index must be in range, or nn.Embedding raises mid-epoch."""
    sizes = data["vocab_sizes"]
    for group, columns in (("seq_cat", SEQ_CATEGORICAL), ("ctx_cat", CTX_CATEGORICAL)):
        array = data[group]
        for i, col in enumerate(columns):
            values = array[..., i]
            assert values.min() >= 0, f"{col} has a negative index"
            assert values.max() < sizes[col], f"{col} exceeds its vocabulary of {sizes[col]}"


def test_missing_values_are_flagged_not_silently_filled(data):
    """A null rest value becomes 0 (the train mean), so the flag is the signal."""
    table = data["table"]
    flag = data["ctx_num"][:, CTX_NUMERIC.index("pitcher_days_since_prev_game_missing")]
    assert flag.sum() == len(PITCHERS)  # one null per pitcher in the fixture
    filled = data["ctx_num"][:, CTX_NUMERIC.index("pitcher_days_since_prev_game")]
    assert (filled[flag == 1] == 0.0).all()
    assert table["pitcher_days_since_prev_game_missing"].sum() == len(PITCHERS)


def test_first_pitch_flags_reach_the_model(data):
    """Open Decision 1: the window never resets, so the model needs the flags."""
    for flag in ("is_first_pitch_of_ab", "is_first_pitch_of_pa"):
        assert flag in CTX_NUMERIC
        values = data["ctx_num"][:, CTX_NUMERIC.index(flag)]
        assert set(np.unique(values)) <= {0.0, 1.0}
        assert 0 < values.sum() < len(values)


# --- scaling ------------------------------------------------------------------

def test_scalers_are_fit_on_train_rows_only(data, clean_path):
    """Fitting on everything would leak val and test statistics into training."""
    assert (data["table"]["split"] == "train").sum() > 0
    assert (data["table"]["split"] != "train").sum() > 0

    # Compare against the unscaled table: data["table"] has been scaled already.
    raw = pd.read_parquet(clean_path)
    train = raw["game_year"].map(SPLITS) == "train"

    for key, columns in (("ctx", ["sz_top", "age_pit"]), ("seq", ["release_speed", "plate_z"])):
        scaler = data["scalers"][key]
        for col in columns:
            i = list(scaler.feature_names_in_).index(col)
            assert scaler.mean_[i] == pytest.approx(raw.loc[train, col].mean()), col
            # ...and not the mean over every row, which is what leakage looks like.
            if raw[col].mean() != pytest.approx(raw.loc[train, col].mean()):
                assert scaler.mean_[i] != pytest.approx(raw[col].mean()), col


def test_scaled_columns_are_standardised_on_train(data):
    table = data["table"]
    train = table["split"] == "train"
    for col in ("sz_top", "release_speed"):
        assert table.loc[train, col].mean() == pytest.approx(0.0, abs=1e-6), col


def test_sequence_numbers_are_scaled_not_raw(data):
    """The lagged values used to come through as raw mph; they must not now."""
    speed = data["seq_num"][~data["pad_mask"]][:, SEQ_NUMERIC.index("release_speed")]
    assert abs(speed).max() < 10, "release_speed looks unscaled"


# --- window length ------------------------------------------------------------

@pytest.mark.parametrize("window", [1, 3, 64])
def test_other_window_lengths_build(clean_path, window):
    built = build_dataset(clean_path, pitcher=None, window=window, verbose=False)
    assert built["seq_cat"].shape[1] == window
    assert np.isfinite(built["seq_num"]).all()
    # A window longer than the history pads the rest, never wraps around.
    assert built["pad_mask"][0].all()
