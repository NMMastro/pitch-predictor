import sys
from pathlib import Path

import pytest

from features.schema import ZONE_CLASSES

from frontend.prediction import PredictionError, Predictor, parse_prediction
from frontend.replay import zone_for_catcher, zone_label
from frontend.strike_zone import ZONE_PATHS, strike_zone


def test_predictor_passes_exact_pitch_and_checkpoint_to_backend(tmp_path):
    checkpoint = tmp_path / "run with spaces.pt"
    script = tmp_path / "backend prediction.py"
    flags = {
        "--clean": "pitches.parquet",
        "--checkpoint": str(checkpoint),
        "--pitcher": "453286",
        "--game-pk": "778536",
        "--at-bat-number": "1",
        "--pitch-number": "2",
    }
    script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "assert Path.cwd().name == 'backend repo'\n"
        f"assert dict(zip(sys.argv[1::2], sys.argv[2::2])) == {flags!r}\n"
        "print(json.dumps({'zone': 14, 'zone_probability': 0.42}))\n"
    )
    backend = tmp_path / "backend repo"
    backend.mkdir()

    result = Predictor(script, sys.executable, backend).predict(
        clean=Path("pitches.parquet"),
        checkpoint=checkpoint,
        pitcher=453286,
        pitch_key=(778536, 1, 2),
    )

    assert result.zone == 14
    assert result.zone_probability == 0.42


def test_placeholder_exercises_the_same_cli_without_weights():
    placeholder = Path(__file__).resolve().parents[1] / "placeholder_predict.py"
    result = Predictor(placeholder, sys.executable).predict(
        clean=Path("pitches.parquet"),
        checkpoint=None,
        pitcher=453286,
        pitch_key=(778536, 1, 1),
    )
    assert result.zone in ZONE_CLASSES
    assert result.zone_probability is None


@pytest.mark.parametrize("stdout", [
    "debug log\n{\"zone\": 1}", "[]", "{}", '{"zone": 10}',
    '{"zone": true}', '{"zone": 1, "zone_probability": 1.1}', '{"pitch_type": ""}',
])
def test_rejects_invalid_prediction_without_showing_a_zone(stdout):
    with pytest.raises(PredictionError):
        parse_prediction(stdout)


def test_pitch_type_response_does_not_create_a_zone():
    result = parse_prediction('{"pitch_type": "FF", "pitch_type_probability": 0.54}')
    assert result.pitch_type == "FF"
    assert result.pitch_type_probability == 0.54
    assert result.zone is None


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
