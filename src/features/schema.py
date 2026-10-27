"""Shared constants for every model: the rules in ``docs/DESIGN_DECISIONS.md``.

Sections, in design-doc order:
    Files
    Pitch types           Section 1
    Zones                 Section 2
    Results               pitch and plate-appearance outcomes
    Rows and splits       Sections 4 and 5
    Pitch order           keys that identify and order pitches
    Clean table columns   Section 6
    Pre-pitch encoding    Section 6a
"""

from __future__ import annotations

from pathlib import Path

# --- Files --------------------------------------------------------------------

# Written by clean.py, read by every feature step.
CLEAN_PATH = Path("data/clean/pitches.parquet")

# --- Pitch types (Section 1) --------------------------------------------------

PITCH_CLASSES: list[str] = ["FF", "SI", "SL", "CH", "FC", "CU", "FS", "KC", "SV", "OTHER"]

# Variants merged into their parent pitch.
MERGE_CODES: dict[str, str] = {"ST": "SL", "FO": "FS", "CS": "CU"}

# Rare types pooled into OTHER.
OTHER_CODES: frozenset[str] = frozenset({"EP", "KN", "SC"})

# Non-pitches, dropped.
DROP_CODES: frozenset[str] = frozenset({"PO", "IN", "UN", "FA", "AB"})

# Pitch classes by kind.
FAMILIES: dict[str, list[str]] = {
    "fastball": ["FF", "SI", "FC"],
    "breaking": ["SL", "CU", "KC", "SV"],
    "offspeed": ["CH", "FS"],
}

# --- Zones (Section 2) --------------------------------------------------------

ZONE_CLASSES: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]

# A left-handed batter's zones are mirrored, so "in" is inside for everyone.
ZONE_NAMES: dict[int, str] = {
    1: "up_in", 2: "up_mid", 3: "up_away",
    4: "mid_in", 5: "mid_mid", 6: "mid_away",
    7: "low_in", 8: "low_mid", 9: "low_away",
    11: "out_up_in", 12: "out_up_away", 13: "out_low_in", 14: "out_low_away",
}

MIRROR_ZONE: dict[int, int] = {1: 3, 2: 2, 3: 1, 4: 6, 5: 5, 6: 4, 7: 9, 8: 8, 9: 7, 11: 12, 12: 11, 13: 14, 14: 13}

ZONE_LABELS: list[str] = [ZONE_NAMES[z] for z in ZONE_CLASSES]

# --- Results ------------------------------------------------------------------

# description -> pitch result
PITCH_RESULTS: dict[str, list[str]] = {
    "ball": ["ball", "blocked_ball", "intent_ball"],
    "called_strike": ["called_strike"],
    "swinging_strike": ["swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"],
    "foul": ["foul", "foul_bunt", "bunt_foul_tip"],
    "in_play": ["hit_into_play"],
    "hit_by_pitch": ["hit_by_pitch"],
}
SWINGS: list[str] = ["swinging_strike", "foul", "in_play"]

# events -> plate-appearance result
PA_RESULTS: dict[str, list[str]] = {
    "strikeout": ["strikeout", "strikeout_double_play"],
    "walk": ["walk", "intent_walk", "hit_by_pitch", "catcher_interf"],
    "out": [
        "field_out", "force_out", "grounded_into_double_play", "double_play", "triple_play",
        "fielders_choice_out", "sac_fly", "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play",
    ],
    "reached": ["field_error", "fielders_choice"],
    "single": ["single"],
    "extra_base": ["double", "triple"],
    "home_run": ["home_run"],
}

# --- Rows and splits (Sections 4 and 5) ---------------------------------------

# Regular season and postseason (spring training dropped).
KEEP_GAME_TYPES: frozenset[str] = frozenset({"R", "F", "D", "L", "W"})

SPLITS: dict[int, str] = {**{y: "train" for y in range(2016, 2024)}, 2024: "val", 2025: "test"}

# Earlier pitches a pitcher needs before a pitch counts as an example.
MIN_PRIOR_PITCHES = 25

# --- Pitch order --------------------------------------------------------------

# Identifies one pitch.
PITCH_KEY: list[str] = ["game_pk", "at_bat_number", "pitch_number"]

# Real-time order.
CHRONO_KEYS: list[str] = ["game_date", "game_pk", "at_bat_number", "pitch_number"]

# Each pitcher's pitches in real-time order (rid numbers the CHRONO_KEYS order).
PITCHER_CHRONO_KEYS: list[str] = ["pitcher", "rid"]

# --- Clean table columns (Section 6) ------------------------------------------
# Raw columns the clean step keeps. Dropped: future info, deprecated/empty,
# duplicates. Kept != model input.

ID_COLUMNS: list[str] = [
    "game_date", "game_pk", "game_year", "game_type", "at_bat_number", "pitch_number",
    "pitcher", "batter", "player_name",
    "fielder_2", "fielder_3", "fielder_4", "fielder_5", "fielder_6", "fielder_7", "fielder_8", "fielder_9",
    "home_team", "away_team",
]

# Known before the pitch.
PRE_PITCH_COLUMNS: list[str] = [
    "balls", "strikes", "outs_when_up", "inning", "inning_topbot", "stand", "p_throws",
    "on_1b", "on_2b", "on_3b", "bat_score", "fld_score", "bat_score_diff", "home_score_diff",
    "sz_top", "sz_bot", "n_thruorder_pitcher", "n_priorpa_thisgame_player_at_bat",
    "pitcher_days_since_prev_game", "batter_days_since_prev_game", "age_pit", "age_bat",
    "if_fielding_alignment", "of_fielding_alignment",
]

# Measured from the pitch: labels, or features of later pitches only.
PITCH_COLUMNS: list[str] = [
    "pitch_type", "zone", "plate_x", "plate_z", "description",
    "release_speed", "release_spin_rate", "spin_axis", "release_extension",
    "pfx_x", "pfx_z", "release_pos_x", "release_pos_z",
]

# Outcome after the pitch: features of later pitches only.
AFTER_PITCH_COLUMNS: list[str] = [
    "events", "bb_type", "launch_speed", "launch_angle", "hit_distance_sc",
    "hit_location", "hc_x", "hc_y",
]

CLEAN_COLUMNS: list[str] = ID_COLUMNS + PRE_PITCH_COLUMNS + PITCH_COLUMNS + AFTER_PITCH_COLUMNS

# --- Pre-pitch encoding (Section 6a) ------------------------------------------

# Text pre-pitch columns; encode.py replaces them with TEXT_FLAGS.
TEXT_COLUMNS: list[str] = ["stand", "p_throws", "inning_topbot"]

# The 0/1 columns encode.py makes from TEXT_COLUMNS.
TEXT_FLAGS: list[str] = ["is_LHB", "is_LHP", "same_hand", "is_inning_top"]

# The pre-pitch columns as encode.py leaves them.
ENCODED_PRE_PITCH_COLUMNS: list[str] = [c for c in PRE_PITCH_COLUMNS if c not in TEXT_COLUMNS] + TEXT_FLAGS

# Extra innings are capped at this value.
INNING_CAP = 10
