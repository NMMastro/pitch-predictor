"""Shared constants for every model: the rules in ``docs/DESIGN_DECISIONS.md``.

Anything that is a fact about the data -- the label taxonomy, which rows are
dropped, the season split, real-time ordering, which raw columns are kept --
lives here, once. Modelling choices (encodings, lags, running averages) do not:
they belong to each pipeline (``features/tabular``, ``features/build_features``).

This module imports nothing from the project, so anything may import it.
"""

from __future__ import annotations

# --- Labels (DESIGN_DECISIONS.md Sections 1 and 2) ----------------------------

PITCH_CLASSES: list[str] = [
    "FF", "SI", "SL", "CH", "FC", "CU", "FS", "KC", "SV", "OTHER",
]
# Variants folded into their parent pitch. ST (sweeper) is only labelled
# consistently from 2023; FO (forkball) and CS (slow curve) are a splitter and a
# curveball by another name.
MERGE_CODES: dict[str, str] = {"ST": "SL", "FO": "FS", "CS": "CU"}
# Rare pitch types pooled into OTHER.
OTHER_CODES: frozenset[str] = frozenset({"EP", "KN", "SC"})
# Non-pitches, dropped before training. AB (automatic ball) appears once, in 2016.
DROP_CODES: frozenset[str] = frozenset({"PO", "IN", "UN", "FA", "AB"})
ZONE_CLASSES: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]

# --- Rows and splits (Sections 4 and 5) ---------------------------------------

# Regular season and postseason; spring training ("S") is dropped.
KEEP_GAME_TYPES: frozenset[str] = frozenset({"R", "F", "D", "L", "W"})

SPLITS: dict[int, str] = {**{y: "train" for y in range(2016, 2024)}, 2024: "val", 2025: "test"}

# A pitch is a training example only once its pitcher has this many earlier
# pitches. The earlier pitches still count toward history.
MIN_PRIOR_PITCHES = 25

# Real-time order. Raw files are sorted by game_pk, which is not chronological
# across games; same-day doubleheaders fall back on game_pk, the closest proxy.
CHRONO_KEYS: list[str] = ["game_date", "game_pk", "at_bat_number", "pitch_number"]

# --- Raw columns kept by the clean step (Section 6) ---------------------------
# An allowlist: the other ~70 Statcast columns never leave data/raw. Being kept
# here does NOT make a column a model input -- each pipeline has its own
# allowlist of inputs.

ID_COLUMNS: list[str] = [
    "game_date", "game_pk", "game_year", "game_type", "at_bat_number", "pitch_number",
    "pitcher", "batter", "player_name", "fielder_2",
]

# 6a: known before the pitch is thrown.
PRE_PITCH_COLUMNS: list[str] = [
    "balls", "strikes", "outs_when_up", "inning", "inning_topbot", "stand", "p_throws",
    "on_1b", "on_2b", "on_3b", "bat_score", "fld_score", "bat_score_diff", "home_score_diff",
    "sz_top", "sz_bot", "n_thruorder_pitcher", "n_priorpa_thisgame_player_at_bat",
    "pitcher_days_since_prev_game", "batter_days_since_prev_game", "age_pit", "age_bat",
]

# 6b/6c: measured from the pitch itself. Labels, or inputs only when lagged
# onto LATER pitches -- never a feature of their own pitch.
PITCH_COLUMNS: list[str] = [
    "pitch_type", "zone", "plate_x", "plate_z", "description",
    "release_speed", "release_spin_rate", "pfx_x", "pfx_z", "release_pos_x", "release_pos_z",
]

CLEAN_COLUMNS: list[str] = ID_COLUMNS + PRE_PITCH_COLUMNS + PITCH_COLUMNS
