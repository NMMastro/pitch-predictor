"""Checks for the optional pitch-mix inputs to the transformer."""

import numpy as np
import pandas as pd

from features.encode import encode
from features.schema import PITCH_CLASSES
from features.sequence_features import CTX_NUMERIC
from features.tabular_features import SLICES, league_mixes, pitcher_type_mix_columns
from models.transformer.dataset import load_splits


def test_transformer_uses_the_tabular_pitch_mixes(synthetic_clean_path):
    clean = pd.read_parquet(synthetic_clean_path)
    encoded = encode(clean)
    league_types, _ = league_mixes(encoded)
    tabular = pd.DataFrame(pitcher_type_mix_columns(encoded, league_types, SLICES))
    tabular.index = encoded["rid"]
    columns = [
        f"p_type_{slice_name}_{pitch_type}"
        for slice_name in ("last200", "game")
        for pitch_type in PITCH_CLASSES
    ]

    for pitcher in (None, 100):
        base = load_splits(synthetic_clean_path, pitcher=pitcher, verbose=False)
        last200 = load_splits(
            synthetic_clean_path, pitcher=pitcher, verbose=False, pitch_mix="last200",
        )
        both = load_splits(
            synthetic_clean_path, pitcher=pitcher, verbose=False, pitch_mix="last200-game",
        )
        width = len(CTX_NUMERIC)
        np.testing.assert_array_equal(both.arrays["ctx_num"][:, :width], base.arrays["ctx_num"])
        np.testing.assert_array_equal(both.arrays["ctx_num"][:, :width + 10], last200.arrays["ctx_num"])
        np.testing.assert_allclose(
            both.arrays["ctx_num"][:, width:], tabular.loc[both.table["rid"], columns].to_numpy(),
        )


def test_mix_excludes_current_pitch_and_resets_game_history():
    rows = pd.DataFrame({
        "pitcher": 100,
        "game_pk": [1] * 201 + [2],
        "type_code": [0] + [1] * 200 + [2],
    })
    league = np.full((len(rows), 10), 0.1, dtype=np.float32)
    mixes = pitcher_type_mix_columns(rows, league, ("last200", "game"))

    career_ff = (1 + 20 * 0.1) / (201 + 20)
    # The fastball is the 201st prior pitch and falls outside the window.
    np.testing.assert_allclose(mixes["p_type_last200_FF"][-1], 20 * career_ff / 220)
    # The first pitch of a new game uses the career mix, with zero game counts.
    np.testing.assert_allclose(mixes["p_type_game_FF"][-1], career_ff)

    rows.loc[rows.index[-1], "type_code"] = 0
    changed = pitcher_type_mix_columns(rows, league, ("last200", "game"))
    for key, values in mixes.items():
        np.testing.assert_array_equal(values, changed[key])
