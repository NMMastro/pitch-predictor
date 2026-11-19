"""Read the 2025 replay and run the model for a single selected pitch."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from features.schema import MIRROR_ZONE, PITCH_KEY, ZONE_CLASSES, ZONE_NAMES
from models.transformer.config import ModelConfig
from models.transformer.transformer import PitchTransformer

from frontend.train_zone import predict_probabilities
from frontend.zone_data import FEATURE_ORDER, PitchRows, encode_pitches, prepare_rows, read_pitches

REPLAY_YEAR = 2025
MIN_REPLAY_PRIOR_PITCHES = 250
NAME_CACHE = Path("frontend/artifacts/player_names.json")
REPLAY_FILTERS = [
    ("game_year", "==", REPLAY_YEAR),
    ("pitcher_prior_pitches", ">=", MIN_REPLAY_PRIOR_PITCHES),
]
GAME_COLUMNS = ["game_pk", "game_date", "away_team", "home_team"]
PitchKey = tuple[int, int, int]


def read_catalog(clean: Path) -> pd.DataFrame:
    columns = ["pitcher", "player_name", *GAME_COLUMNS]
    games = pd.read_parquet(clean, columns=columns, filters=REPLAY_FILTERS)
    return games.drop_duplicates(["pitcher", "game_pk"]).sort_values(["game_date", "game_pk"])


def pitcher_name(stored_name: str) -> str:
    last, separator, first = stored_name.partition(", ")
    return f"{first} {last}" if separator else stored_name


def read_names(path: Path) -> dict[int, str]:
    if not path.is_file():
        return {}
    return {int(player): name for player, name in json.loads(path.read_text()).items()}


def zone_for_catcher(zone: int, stand: str) -> int:
    return MIRROR_ZONE[zone] if stand == "L" else zone


def zone_label(relative_zone: int) -> str:
    return {
        "up_in": "High and inside", "up_mid": "High, over the middle", "up_away": "High and away",
        "mid_in": "Middle inside", "mid_mid": "Middle of the zone", "mid_away": "Middle away",
        "low_in": "Low and inside", "low_mid": "Low, over the middle", "low_away": "Low and away",
        "out_up_in": "High and inside, outside the zone",
        "out_up_away": "High and away, outside the zone",
        "out_low_in": "Low and inside, outside the zone",
        "out_low_away": "Low and away, outside the zone",
    }[ZONE_NAMES[relative_zone]]


@dataclass
class ZoneModel:
    model: PitchTransformer
    preprocessing: dict
    training_pitcher: int | None

    @classmethod
    def load(cls, path: Path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("target") != "zone" or checkpoint.get("zone_classes") != ZONE_CLASSES:
            raise ValueError("Choose a trained 13-zone checkpoint.")
        preprocessing = checkpoint["preprocessing"]
        if preprocessing["feature_order"] != FEATURE_ORDER:
            raise ValueError("Retrain the zone checkpoint for this version of the replay.")
        config = ModelConfig(**checkpoint["model_config"])
        model = PitchTransformer(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return cls(model, preprocessing, checkpoint["training_pitcher"])


@dataclass
class PitcherReplay:
    pitches: pd.DataFrame
    features: PitchRows
    rows_by_pitch: dict[PitchKey, int]

    def predict_zone(self, model: ZoneModel, key: PitchKey) -> tuple[int, float]:
        row = self.rows_by_pitch[key]
        batch = self.features.batch(np.array([row]), model.model.config.window)
        probabilities = predict_probabilities(model.model, [batch])[0]
        most_likely = int(probabilities.argmax())
        return ZONE_CLASSES[most_likely], float(probabilities[most_likely])


def load_pitcher(clean: Path, pitcher: int, preprocessing: dict) -> PitcherReplay:
    columns = [
        "rid", *PITCH_KEY, "inning", "inning_topbot", "balls", "strikes", "outs_when_up",
        "batter", "stand", "p_throws", "player_name", "home_team", "away_team", "zone_class",
    ]
    pitches = pd.read_parquet(
        clean, columns=columns, filters=[*REPLAY_FILTERS, ("pitcher", "==", pitcher)]
    ).sort_values("rid", ignore_index=True)
    raw = read_pitches(clean, pitcher)
    feature_indices = raw["rid"].searchsorted(pitches["rid"])
    if not np.array_equal(raw["rid"].to_numpy()[feature_indices], pitches["rid"].to_numpy()):
        raise ValueError("The replay and model feature rows are from different datasets.")
    rows_by_pitch = dict(zip(pitches[PITCH_KEY].itertuples(index=False, name=None), feature_indices))
    features = prepare_rows(encode_pitches(raw), preprocessing)
    return PitcherReplay(pitches, features, rows_by_pitch)
