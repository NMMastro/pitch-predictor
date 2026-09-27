"""Smoke tests for the evaluation harness."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluation import (
    PITCH_CLASSES,
    ZONE_CLASSES,
    align_proba,
    evaluate,
    load_scoreboard,
    log_run,
    map_pitch_classes,
    prepare,
)

K = len(PITCH_CLASSES)


def _onehot(labels, classes=PITCH_CLASSES):
    """Perfect, fully confident predictions for `labels`."""
    classes = list(classes)
    idx = [classes.index(c) for c in labels]
    p = np.full((len(labels), len(classes)), 1e-12)
    p[np.arange(len(labels)), idx] = 1.0
    return p / p.sum(axis=1, keepdims=True)


def _manual_log_loss(y, p, classes):
    idx = {c: i for i, c in enumerate(classes)}
    return -np.log([p[i, idx[v]] for i, v in enumerate(y)]).mean()


# --- core scoring ------------------------------------------------------------

def test_perfect_predictions_score_perfectly():
    y = ["FF", "SL", "OTHER", "CH"]
    m = evaluate(y, _onehot(y))
    assert m["accuracy"] == 1.0
    assert m["top2_accuracy"] == 1.0
    assert m["log_loss"] < 1e-6
    assert m["per_class_f1"]["FF"] == 1.0


def test_uniform_predictions_give_log_uniform_loss():
    y = ["FF"] * 50 + ["SL"] * 50
    p = np.full((100, K), 1 / K)
    m = evaluate(y, p)
    assert m["log_loss"] == pytest.approx(np.log(K), abs=1e-9)
    assert m["n"] == 100


def test_macro_f1_averages_over_all_classes_not_just_present_ones():
    """Micro-F1 would equal accuracy here; macro over present classes would be ~0.47."""
    y = ["FF"] * 90 + ["SL"] * 10
    m = evaluate(y, _onehot(["FF"] * 100))
    assert m["accuracy"] == pytest.approx(0.90)
    assert m["macro_f1"] == pytest.approx(0.0861, abs=1e-3)


def test_per_class_can_be_switched_off():
    m = evaluate(["FF", "SL"], _onehot(["FF", "SL"]), per_class=False)
    assert "per_class_f1" not in m and "support" not in m
    assert {"log_loss", "accuracy", "top2_accuracy", "macro_f1"} <= m.keys()


def test_zero_support_classes_are_surfaced():
    m = evaluate(["FF", "SL"], _onehot(["FF", "SL"]))
    assert "OTHER" in m["zero_support_classes"]
    assert m["support"]["FF"] == 1


# --- input validation --------------------------------------------------------

def test_shuffled_columns_change_the_score():
    """The failure this harness exists to make visible."""
    y = ["FF", "SL", "CH", "OTHER"] * 10
    good = _onehot(y)
    assert evaluate(y, good)["accuracy"] == 1.0
    assert evaluate(y, good[:, ::-1])["accuracy"] == 0.0


def test_rejects_wrong_column_count():
    with pytest.raises(ValueError, match="columns"):
        evaluate(["FF", "SL"], np.full((2, K - 1), 1 / (K - 1)))


def test_rejects_rows_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        evaluate(["FF", "SL"], np.full((2, K), 0.5))


def test_rejects_nan_probabilities():
    p = np.full((2, K), 1 / K)
    p[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        evaluate(["FF", "SL"], p)


def test_logits_are_rejected_with_a_useful_message():
    """Sequence models emit logits; they need a softmax first."""
    logits = np.array([[2.0, -1.0] + [0.0] * (K - 2)] * 2)
    with pytest.raises(ValueError, match="softmax"):
        evaluate(["FF", "SL"], logits)


# --- both targets: string and integer classes --------------------------------

@pytest.mark.parametrize("classes", [PITCH_CLASSES, ZONE_CLASSES])
def test_log_loss_matches_manual_for_str_and_int_classes(classes):
    """Guards the column permutation for both targets; ints must sort numerically."""
    rng = np.random.default_rng(0)
    y = rng.choice(np.array(classes, dtype=object), 300)
    p = rng.random((300, len(classes)))
    p /= p.sum(1, keepdims=True)
    got = evaluate(list(y), p, classes)["log_loss"]
    assert got == pytest.approx(_manual_log_loss(y, p, list(classes)), abs=1e-9)


def test_zone_classes_score_end_to_end():
    y = [1, 5, 14, 11, 9]
    m = evaluate(y, _onehot(y, ZONE_CLASSES), ZONE_CLASSES)
    assert m["accuracy"] == 1.0
    assert m["support"]["14"] == 1


def test_sequence_model_flattened_output_scores():
    """(batch, timesteps, classes) flattened with padded positions dropped."""
    rng = np.random.default_rng(1)
    seq = rng.random((32, 10, K))
    seq /= seq.sum(-1, keepdims=True)
    mask = rng.random((32, 10)) > 0.3
    flat = seq[mask]
    y = list(rng.choice(np.array(PITCH_CLASSES, dtype=object), mask.sum()))
    m = evaluate(y, flat)
    assert m["n"] == int(mask.sum()) == flat.shape[0]
    assert np.isfinite(m["log_loss"]) and m["log_loss"] > 0


# --- align_proba -------------------------------------------------------------

def test_align_proba_fixes_sklearn_class_order():
    """sklearn's classes_ is alphabetical; ours is by frequency, sharing no position."""
    y = ["FF", "SL", "CH", "OTHER"] * 5
    sk_classes = sorted(PITCH_CLASSES)
    p_sk = _onehot(y, sk_classes)
    assert evaluate(y, p_sk)["accuracy"] == 0.0
    assert evaluate(y, align_proba(p_sk, sk_classes))["accuracy"] == 1.0


def test_align_proba_rejects_mismatched_label_sets():
    wrong = [c for c in sorted(PITCH_CLASSES) if c != "OTHER"] + ["XX"]
    with pytest.raises(ValueError, match="different label sets"):
        align_proba(np.full((2, K), 1 / K), wrong)


# --- preprocessing helpers ---------------------------------------------------

def test_map_pitch_classes_pools_and_drops():
    s = pd.Series(["FF", "KN", "SC", "PO", "IN", "UN", "FA", None, "ST"])
    out = map_pitch_classes(s)
    assert out.tolist()[:3] == ["FF", "OTHER", "OTHER"]
    assert out.isna().sum() == 5          # PO, IN, UN, FA, None
    assert out.iloc[8] == "ST"


def test_map_pitch_classes_rejects_unknown_code():
    with pytest.raises(ValueError, match="not covered"):
        map_pitch_classes(pd.Series(["FF", "ZZ"]))


def test_prepare_drops_spring_training_and_nulls():
    df = pd.DataFrame({
        "game_type": ["R", "S", "W", "R", "R"],
        "pitch_type": ["FF", "FF", "SL", None, "PO"],
    })
    out = prepare(df)
    assert len(out) == 2                  # R/FF and W/SL survive
    assert set(out["game_type"]) == {"R", "W"}


def test_prepare_attaches_nullable_zone_class():
    df = pd.DataFrame({
        "game_type": ["R", "R"],
        "pitch_type": ["FF", "SL"],
        "zone": [5.0, None],
    })
    out = prepare(df)
    assert out["zone_class"].dtype == "Int64"
    assert out["zone_class"].tolist()[0] == 5
    assert out["zone_class"].isna().sum() == 1


# --- scoreboard --------------------------------------------------------------

def test_log_run_appends_and_reloads(tmp_path):
    p = tmp_path / "results.jsonl"
    y = ["FF", "SL"]
    log_run(p, evaluate(y, _onehot(y)), model="baseline", target="type", features_version="v1")
    log_run(p, evaluate(y, _onehot(y)), model="logreg", target="type", features_version="v1")
    board = load_scoreboard(p)
    assert len(board) == 2
    assert list(board["model"]) == ["baseline", "logreg"]
    assert "run_at" in board.columns


def test_log_run_serialises_integer_zone_labels(tmp_path):
    p = tmp_path / "zone.jsonl"
    y = [1, 5, 14]
    log_run(p, evaluate(y, _onehot(y, ZONE_CLASSES), ZONE_CLASSES), model="baseline", target="zone")
    assert load_scoreboard(p).loc[0, "accuracy"] == 1.0


def test_load_scoreboard_returns_empty_frame_when_missing(tmp_path):
    assert load_scoreboard(tmp_path / "nope.jsonl").empty
