"""Tests for the split filtering and the batches handed to the model.

The dataset layer has one job that is easy to get wrong and invisible when it
is: deciding which rows are examples. Rows are split by season and filtered by
eligibility, while the *history* behind each row stays whole, so these tests
check the row index and leave the arrays alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import SYNTHETIC_MIN_PRIOR, SYNTHETIC_PITCHERS
from features.schema import PITCH_CLASSES
from model.config import TrainConfig
from model.dataset import (
    BATCH_KEYS,
    SPLIT_NAMES,
    PitchDataset,
    load_splits,
    make_loaders,
)

WINDOW = 5


@pytest.fixture(scope="module")
def splits(synthetic_clean_path):
    return load_splits(synthetic_clean_path, pitcher=None, window=WINDOW, verbose=False)


@pytest.fixture(scope="module")
def raw(synthetic_clean_path) -> pd.DataFrame:
    return pd.read_parquet(synthetic_clean_path)


# --- which rows are examples --------------------------------------------------

def test_each_split_holds_its_own_eligible_rows(splits, raw):
    """Section 4 and 5: split by season, then drop the warm-up pitches."""
    for name in SPLIT_NAMES:
        expected = int(((raw["split"] == name) & raw["eligible"]).sum())
        assert len(getattr(splits, name)) == expected, name
    assert len(splits.train) > 0 and len(splits.val) > 0 and len(splits.test) > 0


def test_ineligible_rows_are_excluded_but_still_feed_history(splits, raw):
    """The first pitches of a career are not examples, but are not deleted."""
    dropped = int((~raw["eligible"]).sum())
    assert dropped == SYNTHETIC_MIN_PRIOR * len(SYNTHETIC_PITCHERS)

    kept = sum(len(getattr(splits, name)) for name in SPLIT_NAMES)
    assert kept == len(raw) - dropped
    # The arrays keep every row, so a window can still reach back into them.
    assert len(splits.arrays["y"]) == len(raw)


def test_splits_never_share_a_row(splits):
    rows = [set(getattr(splits, name).rows.tolist()) for name in SPLIT_NAMES]
    assert not rows[0] & rows[1]
    assert not rows[0] & rows[2]
    assert not rows[1] & rows[2]


def test_split_rows_come_from_the_right_season(splits):
    table = splits.table
    for name in SPLIT_NAMES:
        dataset = getattr(splits, name)
        assert (table["split"].to_numpy()[dataset.rows] == name).all(), name
        assert table["eligible"].to_numpy()[dataset.rows].all(), name


def test_splits_share_one_copy_of_the_arrays(splits):
    """Three views, not three copies -- the full dataset is large."""
    for name in SPLIT_NAMES:
        assert getattr(splits, name).arrays is splits.arrays


# --- batches ------------------------------------------------------------------

def test_one_example_has_the_expected_keys_shapes_and_dtypes(splits):
    example = splits.train[0]
    assert set(example) == set(BATCH_KEYS)

    assert example["seq_cat"].shape == (WINDOW, 3)
    assert example["seq_num"].shape == (WINDOW, 3)
    assert example["pad_mask"].shape == (WINDOW,)
    assert example["ctx_cat"].shape == (splits.arrays["ctx_cat"].shape[1],)
    assert example["ctx_num"].shape == (splits.n_ctx_numeric,)
    assert example["y"].shape == ()

    assert example["seq_cat"].dtype == torch.int64
    assert example["ctx_cat"].dtype == torch.int64
    assert example["y"].dtype == torch.int64
    assert example["seq_num"].dtype == torch.float32
    assert example["ctx_num"].dtype == torch.float32
    assert example["pad_mask"].dtype == torch.bool


def test_default_collate_stacks_a_batch(splits):
    config = TrainConfig(batch_size=4)
    train_loader, _, _ = make_loaders(splits, config)
    batch = next(iter(train_loader))

    assert set(batch) == set(BATCH_KEYS)
    assert batch["seq_cat"].shape == (4, WINDOW, 3)
    assert batch["seq_num"].shape == (4, WINDOW, 3)
    assert batch["pad_mask"].shape == (4, WINDOW)
    assert batch["y"].shape == (4,)


def test_a_loader_covers_its_split_exactly_once(splits):
    """No row dropped by drop_last, none seen twice."""
    config = TrainConfig(batch_size=7)  # does not divide the split evenly
    _, val_loader, _ = make_loaders(splits, config)

    seen = sum(len(batch["y"]) for batch in val_loader)
    assert seen == len(splits.val)


def test_shuffling_is_off_for_val_and_test(splits):
    _, val_loader, _ = make_loaders(splits, TrainConfig(batch_size=4))
    first = torch.cat([batch["y"] for batch in val_loader])
    second = torch.cat([batch["y"] for batch in val_loader])
    assert torch.equal(first, second)
    # ...and the order matches the split's row order, so predictions line up.
    assert first.tolist() == splits.arrays["y"][splits.val.rows].tolist()


def test_batches_carry_no_nans(splits):
    train_loader, _, _ = make_loaders(splits, TrainConfig(batch_size=16))
    for batch in train_loader:
        assert torch.isfinite(batch["seq_num"]).all()
        assert torch.isfinite(batch["ctx_num"]).all()


# --- metadata the model is configured from ------------------------------------

def test_metadata_matches_the_arrays(splits):
    assert splits.window == WINDOW
    assert splits.n_seq_numeric == splits.arrays["seq_num"].shape[-1]
    assert splits.n_ctx_numeric == splits.arrays["ctx_num"].shape[-1]
    assert splits.n_classes == len(PITCH_CLASSES)
    assert set(splits.vocab_sizes) >= {"type_code", "batter", "inning_code"}


def test_train_classes_are_the_ones_actually_thrown(splits):
    """Open Decision 2: scoring is restricted to these, the head is not."""
    y_train = splits.arrays["y"][splits.train.rows]
    assert splits.train_classes == sorted(set(y_train.tolist()))
    assert splits.class_counts.sum() == len(y_train)
    assert all(splits.class_counts[i] > 0 for i in splits.train_classes)
    assert splits.class_names == [PITCH_CLASSES[i] for i in splits.train_classes]


def test_labels_are_valid_class_indices(splits):
    y = splits.arrays["y"]
    assert y.min() >= 0
    assert y.max() < len(PITCH_CLASSES)


def test_dataset_repr_and_summary_are_readable(splits):
    assert "train" in repr(splits.train)
    assert "window" in splits.summary()


def test_pitch_dataset_indexes_the_rows_it_was_given(splits):
    """A PitchDataset is a view: item i must be table row rows[i]."""
    rows = np.array([3, 1, 2])
    dataset = PitchDataset(splits.arrays, rows, "custom")
    assert len(dataset) == 3
    for i, row in enumerate(rows):
        assert dataset[i]["y"].item() == splits.arrays["y"][row]
