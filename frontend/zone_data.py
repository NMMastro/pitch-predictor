"""Shared feature preparation for zone training and replay."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

from features.encode import pitch_codes, pre_pitch_flags
from features.schema import INNING_CAP, PITCHER_CHRONO_KEYS, PRE_PITCH_COLUMNS, SPLITS, ZONE_CLASSES
from features.sequence_features import (
    CODE_SHIFT,
    CONTINUOUS_INPUTS,
    CTX_NUMERIC,
    INPUT_EMBED,
    MIN_BATTER_COUNT,
    SEQ_CATEGORICAL,
    SEQ_NUMERIC,
    TAXONOMY_SIZES,
    UNK_INDEX,
    fit_scaler,
    missing_flags,
)

# Statcast measures these during the pitch. The replay diagram is schematic.
MEASURED_BOUNDS = {"sz_top", "sz_bot"}
ZONE_CONTEXT = [column for column in CTX_NUMERIC if column not in MEASURED_BOUNDS]
ALIGNMENTS = {"if_align_code": "if_fielding_alignment", "of_align_code": "of_fielding_alignment"}
CATEGORICAL_SOURCES = {**{column: column for column in INPUT_EMBED}, **ALIGNMENTS}
SCALED_COLUMNS = SEQ_NUMERIC + [
    column for column in CONTINUOUS_INPUTS if column not in MEASURED_BOUNDS
]
FEATURE_COLUMNS = [
    "rid", "game_year", "split", "eligible", "game_pk", "at_bat_number", "pitch_number",
    *INPUT_EMBED, *(column for column in PRE_PITCH_COLUMNS if column not in MEASURED_BOUNDS),
    "pitch_class", "zone_class", "description", "events", "release_speed", "plate_x", "plate_z",
]
FEATURE_ORDER = {
    "seq_cat": SEQ_CATEGORICAL,
    "seq_num": SEQ_NUMERIC,
    "ctx_cat": [*CATEGORICAL_SOURCES, "inning_code"],
    "ctx_num": ZONE_CONTEXT,
}


def read_pitches(path: Path, pitcher: int | None = None) -> pd.DataFrame:
    """Read only the inputs needed for the model, ordered within each pitcher."""
    filters = [("game_year", "<=", max(SPLITS))]
    if pitcher is not None:
        filters.append(("pitcher", "==", pitcher))
    return pd.read_parquet(path, columns=FEATURE_COLUMNS, filters=filters).sort_values(
        PITCHER_CHRONO_KEYS, ignore_index=True
    )


def encode_pitches(pitches: pd.DataFrame) -> pd.DataFrame:
    # read_pitches already sorts within each pitcher and plate appearance.
    return missing_flags(pitch_codes(pre_pitch_flags(pitches)))


def fit_preprocessing(table: pd.DataFrame) -> dict:
    """Fit once on training seasons and save only primitive checkpoint values."""
    train = (table["split"] == "train").to_numpy()
    if not train.any():
        raise ValueError("No 2016–2023 pitches are available for training.")
    vocabularies = {}
    for feature, column in CATEGORICAL_SOURCES.items():
        counts = table.loc[train, column].value_counts()
        minimum = MIN_BATTER_COUNT if column == "batter" else 1
        vocabularies[feature] = counts[counts >= minimum].sort_index().index.tolist()
    scaler = fit_scaler(table.loc[train, SCALED_COLUMNS], SCALED_COLUMNS)
    return {
        "vocabularies": vocabularies,
        "scaling": {
            column: [float(mean), float(scale)]
            for column, mean, scale in zip(SCALED_COLUMNS, scaler.mean_, scaler.scale_)
        },
        "feature_order": FEATURE_ORDER,
    }


def vocabulary_sizes(preprocessing: dict) -> dict[str, int]:
    sizes = {key: len(values) for key, values in preprocessing["vocabularies"].items()}
    sizes.update(TAXONOMY_SIZES)
    return {key: size + CODE_SHIFT for key, size in sizes.items()}


def numeric_values(table: pd.DataFrame, columns: list[str], scaling: dict) -> np.ndarray:
    output = np.empty((len(table), len(columns)), dtype=np.float32)
    for index, column in enumerate(columns):
        values = table[column].to_numpy(dtype=np.float64, na_value=np.nan)
        if column in scaling:
            mean, scale = scaling[column]
            values = (values - mean) / scale
        output[:, index] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return output


@dataclass
class PitchRows:
    """One encoded row per pitch. History tensors are built one batch at a time."""

    seq_cat: np.ndarray
    seq_num: np.ndarray
    ctx_cat: np.ndarray
    ctx_num: np.ndarray
    labels: np.ndarray
    pitcher_start: np.ndarray

    def batch(self, rows: np.ndarray, window: int) -> dict[str, torch.Tensor]:
        previous = rows[:, None] + np.arange(-window, 0)
        padding = previous < self.pitcher_start[rows, None]
        previous[padding] = 0
        seq_cat = self.seq_cat[previous].astype(np.int64)
        seq_num = self.seq_num[previous]
        seq_cat[padding] = 0
        seq_num[padding] = 0.0
        return {
            "seq_cat": torch.from_numpy(seq_cat),
            "seq_num": torch.from_numpy(seq_num),
            "pad_mask": torch.from_numpy(padding),
            "ctx_cat": torch.from_numpy(self.ctx_cat[rows].astype(np.int64)),
            "ctx_num": torch.from_numpy(self.ctx_num[rows]),
            "y": torch.from_numpy(self.labels[rows].astype(np.int64)),
        }


def prepare_rows(table: pd.DataFrame, preprocessing: dict) -> PitchRows:
    mapping = preprocessing["vocabularies"]
    context = np.empty((len(table), len(FEATURE_ORDER["ctx_cat"])), dtype=np.int32)
    for index, (feature, column) in enumerate(CATEGORICAL_SOURCES.items()):
        lookup = {value: code for code, value in enumerate(mapping[feature], CODE_SHIFT)}
        context[:, index] = table[column].map(lookup).astype("float64").fillna(UNK_INDEX)
    context[:, -1] = table["inning"].clip(1, INNING_CAP).to_numpy() - 1 + CODE_SHIFT

    ids = table["pitcher"].to_numpy()
    pitcher_start = np.zeros(len(ids), dtype=np.int32)
    boundaries = np.flatnonzero(ids[1:] != ids[:-1]) + 1
    pitcher_start[boundaries] = boundaries
    np.maximum.accumulate(pitcher_start, out=pitcher_start)
    scaling = preprocessing["scaling"]
    return PitchRows(
        table[SEQ_CATEGORICAL].to_numpy(dtype=np.int16) + CODE_SHIFT,
        numeric_values(table, SEQ_NUMERIC, scaling),
        context,
        numeric_values(table, ZONE_CONTEXT, scaling),
        table["zone_code"].to_numpy(),
        pitcher_start,
    )


class PitchBatches:
    def __init__(self, features: PitchRows, rows: np.ndarray, window: int):
        self.features = features
        self.rows = rows
        self.window = window

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, indices):
        return self.features.batch(self.rows[indices], self.window)


def make_loader(features, rows, window, batch_size, *, shuffle=False):
    dataset = PitchBatches(features, rows, window)
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    return DataLoader(dataset, sampler=BatchSampler(sampler, batch_size, False), batch_size=None)


def split_rows(encoded: pd.DataFrame) -> dict[str, np.ndarray]:
    # A missing recorded location can still contribute history to later pitches.
    valid = encoded["eligible"] & encoded["zone_code"].between(0, len(ZONE_CLASSES) - 1)
    return {
        name: np.flatnonzero(valid & (encoded["split"] == name))
        for name in ("train", "val", "test")
    }
