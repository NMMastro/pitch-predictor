"""Dataset and loaders: the pipeline's arrays -> batches of tensors.

The whole v1 dataset is a few hundred megabytes, so it is built once, held in
memory, and sliced per split. A ``PitchDataset`` is a view over those shared
arrays, not a copy.

Two filters decide which rows are examples, both from DESIGN_DECISIONS.md:

  split      Section 4: train 2016-2023, val 2024, test 2025. The *rows* are
             split by season, but the *history* behind each row is not -- a 2025
             pitch legitimately sees that pitcher's 2016-2024 record, so the
             arrays stay in one piece and only the row index is split.
  eligible   Section 5: a pitch counts as an example once the pitcher has 25
             earlier pitches. Those first 25 still feed everyone's history;
             they just are not predicted.

Usage:
    from model.dataset import load_splits

    splits = load_splits(pitcher=SCHERZER, window=64)
    train_loader, val_loader, test_loader = make_loaders(splits, TrainConfig())
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features.schema import CLEAN_PATH, PITCH_CLASSES  # noqa: E402
from features.sequence_features import (  # noqa: E402
    CTX_CATEGORICAL,
    CTX_NUMERIC,
    SEQ_CATEGORICAL,
    SEQ_NUMERIC,
    WINDOW,
    build_dataset,
)

__all__ = [
    "CTX_CATEGORICAL",
    "CTX_NUMERIC",
    "SEQ_CATEGORICAL",
    "SEQ_NUMERIC",
    "PitchDataset",
    "SplitData",
    "load_splits",
    "make_loaders",
]

# The keys of one example, in the order the model reads them.
BATCH_KEYS = ("seq_cat", "seq_num", "pad_mask", "ctx_cat", "ctx_num", "y")

SPLIT_NAMES = ("train", "val", "test")


class PitchDataset(Dataset):
    """The rows of one split, as tensors.

    Holds a reference to the shared arrays plus the row numbers belonging to
    this split, so three splits cost one copy of the data between them.
    ``__getitem__`` returns a dict with :data:`BATCH_KEYS`; the default
    collate function stacks those into a batch, no custom collate needed.
    """

    def __init__(self, arrays: dict[str, np.ndarray], rows: np.ndarray, name: str = "") -> None:
        self.arrays = arrays
        self.rows = np.asarray(rows, dtype=np.int64)
        self.name = name

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        row = self.rows[i]
        return {key: torch.as_tensor(self.arrays[key][row]) for key in BATCH_KEYS}

    def __repr__(self) -> str:
        return f"PitchDataset({self.name!r}, {len(self)} rows)"


class SplitData:
    """One built dataset, its three splits, and the metadata a model needs.

    Attributes
    ----------
    train, val, test:
        :class:`PitchDataset` views.
    vocab_sizes:
        Embedding rows per categorical column.
    window, n_seq_numeric, n_ctx_numeric, n_classes:
        The shapes a :class:`~model.config.ModelConfig` needs.
    train_classes:
        Indices into :data:`~features.schema.PITCH_CLASSES` that appear in the
        training split -- the restricted label set for scoring (Open Decision
        2). Sorted.
    class_counts:
        How often each of the ten classes appears in training, used for the
        frequency baseline.
    table:
        The feature table behind the arrays, for inspection.
    """

    def __init__(self, data: dict) -> None:
        table = data["table"]
        self.arrays = {key: data[key] for key in BATCH_KEYS}
        self.table = table
        self.vocab_sizes = data["vocab_sizes"]
        self.window = data["window"]
        self.scalers = data["scalers"]

        eligible = table["eligible"].to_numpy(dtype=bool)
        split = table["split"].to_numpy()
        for name in SPLIT_NAMES:
            rows = np.flatnonzero(eligible & (split == name))
            setattr(self, name, PitchDataset(self.arrays, rows, name))

        y_train = self.arrays["y"][self.train.rows]
        self.class_counts = np.bincount(y_train, minlength=len(PITCH_CLASSES))
        self.train_classes = np.flatnonzero(self.class_counts > 0).tolist()

    @property
    def n_seq_numeric(self) -> int:
        return self.arrays["seq_num"].shape[-1]

    @property
    def n_ctx_numeric(self) -> int:
        return self.arrays["ctx_num"].shape[-1]

    @property
    def n_classes(self) -> int:
        return len(PITCH_CLASSES)

    @property
    def class_names(self) -> list[str]:
        """The names of :attr:`train_classes`, in the same order."""
        return [PITCH_CLASSES[i] for i in self.train_classes]

    def summary(self) -> str:
        counts = ", ".join(f"{n} {len(getattr(self, n)):,}" for n in SPLIT_NAMES)
        mix = ", ".join(
            f"{PITCH_CLASSES[i]} {self.class_counts[i] / self.class_counts.sum():.0%}"
            for i in self.train_classes
        )
        return (
            f"window {self.window}, {counts}\n"
            f"classes thrown in train: {mix}"
        )


def load_splits(
    clean_path: str | Path = CLEAN_PATH,
    pitcher: int | None = None,
    window: int = WINDOW,
    verbose: bool = True,
) -> SplitData:
    """Build the arrays and wrap them as three splits."""
    return SplitData(build_dataset(clean_path, pitcher, window, verbose))


def make_loaders(
    splits: SplitData, config, shuffle_train: bool = True
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Train, val and test loaders. Only training is shuffled.

    ``config`` is a :class:`~model.config.TrainConfig`.
    """
    def loader(dataset: PitchDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=config.num_workers,
            drop_last=False,
        )

    return (
        loader(splits.train, shuffle_train),
        loader(splits.val, False),
        loader(splits.test, False),
    )


def main() -> None:
    """Print one real batch's shapes -- the quickest check that the data is sane."""
    import argparse

    from model.config import SCHERZER, TrainConfig

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--clean", type=Path, default=CLEAN_PATH)
    parser.add_argument("--pitcher", type=int, default=SCHERZER, help="0 = every pitcher")
    parser.add_argument("--window", type=int, default=WINDOW)
    args = parser.parse_args()

    splits = load_splits(args.clean, args.pitcher or None, args.window)
    print(f"\n{splits.summary()}\n")

    train_loader, _, _ = make_loaders(splits, TrainConfig())
    batch = next(iter(train_loader))
    print("One batch")
    for key in BATCH_KEYS:
        print(f"  {key:9} {str(tuple(batch[key].shape)):18} {batch[key].dtype}")

    print("\nEmbedding vocabularies")
    for col, size in splits.vocab_sizes.items():
        print(f"  {col:24} {size}")


if __name__ == "__main__":
    main()
