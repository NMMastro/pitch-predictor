from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from features.schema import PITCH_CLASSES, ZONE_CLASSES
from models.transformer.config import ModelConfig
from models.transformer.transformer import PitchTransformer

from frontend.replay import PitcherReplay, ZoneModel, zone_for_catcher, zone_label
from frontend.strike_zone import ZONE_PATHS, strike_zone
from frontend.train_zone import predict_probabilities
from frontend.zone_data import (
    FEATURE_COLUMNS,
    FEATURE_ORDER,
    encode_pitches,
    fit_preprocessing,
    make_loader,
    prepare_rows,
    split_rows,
    vocabulary_sizes,
)


@pytest.fixture
def pitches():
    rows = []
    for pitcher in (1, 2):
        for index, split in enumerate(("train", "train", "val", "test", "test")):
            rows.append({
                **dict.fromkeys(FEATURE_COLUMNS, 0),
                "rid": len(rows),
                "pitcher": pitcher,
                "batter": pitcher if split == "train" else 99,
                "eligible": index > 0,
                "split": split,
                "game_pk": index,
                "at_bat_number": 1,
                "pitch_number": 1,
                "stand": "L" if index == 3 else "R",
                "inning": 1,
                "inning_topbot": "Top",
                "p_throws": "R",
                "pitch_class": "FF",
                "zone_class": (None if index == 3 and pitcher == 2 else 14),
                "description": "ball",
                "release_speed": 90 + index,
                "plate_x": 1.0,
                "plate_z": 1.5,
                "if_fielding_alignment": "Standard",
                "of_fielding_alignment": "Standard",
            })
    table = pd.DataFrame(rows)
    table["pitch_class"] = pd.Categorical(table["pitch_class"], categories=PITCH_CLASSES)
    return table


def test_training_and_replay_share_the_exact_pitch_inputs(pitches):
    table = encode_pitches(pitches)
    preprocessing = fit_preprocessing(table)
    features = prepare_rows(table, preprocessing)
    config = ModelConfig(
        n_classes=len(ZONE_CLASSES),
        vocab_sizes=vocabulary_sizes(preprocessing),
        n_ctx_numeric=len(FEATURE_ORDER["ctx_num"]),
        window=4,
    )
    model = PitchTransformer(config)
    trained = ZoneModel(model, preprocessing, None)
    expected = predict_probabilities(model, make_loader(features, split_rows(table)["test"], 4, 2))

    selected = prepare_rows(table.loc[table["pitcher"] == 2], preprocessing)
    replay_probability = predict_probabilities(model, [selected.batch(np.array([4]), 4)])[0]
    np.testing.assert_allclose(replay_probability, expected[-1], rtol=1e-5, atol=1e-6)
    replay = PitcherReplay(pd.DataFrame(), selected, {(4, 1, 1): 4})
    zone, probability = replay.predict_zone(trained, (4, 1, 1))
    assert zone == ZONE_CLASSES[expected[-1].argmax()]
    assert probability == pytest.approx(expected[-1].max(), abs=1e-6)
    assert selected.labels[3] < 0  # missing outcomes stay in the history
    assert selected.batch(np.array([4]), 4)["pad_mask"].sum() == 0
    assert features.batch(np.array([5]), 4)["pad_mask"].all()  # new pitcher starts empty


def test_current_outcome_and_measured_bounds_do_not_reach_model(pitches):
    baseline = encode_pitches(pitches)
    preprocessing = fit_preprocessing(baseline)

    changed = deepcopy(pitches)
    changed.loc[3, ["zone_class", "description", "plate_x", "plate_z", "release_speed"]] = [
        5, "called_strike", 0.0, 2.5, 105,
    ]
    changed["sz_top"] = 6.0
    changed["sz_bot"] = 3.0

    original = prepare_rows(baseline, preprocessing).batch(np.array([3, 4]), 3)
    updated = prepare_rows(encode_pitches(changed), preprocessing).batch(np.array([3, 4]), 3)
    assert original["y"][0] != updated["y"][0]
    for key in ("seq_num", "seq_cat", "ctx_num", "ctx_cat", "pad_mask"):
        torch.testing.assert_close(original[key][0], updated[key][0])
    assert not torch.equal(original["seq_cat"][1], updated["seq_cat"][1])


def test_catcher_view_maps_every_region_and_keeps_matching_overlays():
    assert set(ZONE_PATHS) == set(ZONE_CLASSES)
    for zone in ZONE_CLASSES:
        assert zone_for_catcher(zone, "R") == zone
        assert zone_for_catcher(zone_for_catcher(zone, "L"), "L") == zone

    inside = zone_for_catcher(14, "L")
    assert inside == 13
    assert zone_label(inside) == "Low and inside, outside the zone"
    matching = strike_zone("L", 14, 14)
    assert 'translate(392 144)' in matching
    assert f'class="zone-prediction" d="{ZONE_PATHS[14]}"' in matching
    assert f'class="zone-actual" d="{ZONE_PATHS[14]}"' in matching
    assert 'translate(40 144)' in strike_zone("R", None, None)
