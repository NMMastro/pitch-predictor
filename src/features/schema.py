"""Shared constants for every model: the rules in ``docs/DESIGN_DECISIONS.md``."""

from __future__ import annotations

# --- Labels (DESIGN_DECISIONS.md Sections 1 and 2) ----------------------------

PITCH_CLASSES: list[str] = ["FF", "SI", "SL", "CH", "FC", "CU", "FS", "KC", "SV", "OTHER"]

# Variants merged into their parent pitch.
MERGE_CODES: dict[str, str] = {"ST": "SL", "FO": "FS", "CS": "CU"}

# Rare types pooled into OTHER.
OTHER_CODES: frozenset[str] = frozenset({"EP", "KN", "SC"})

# Non-pitches, dropped.
DROP_CODES: frozenset[str] = frozenset({"PO", "IN", "UN", "FA", "AB"})
ZONE_CLASSES: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]

# --- Rows and splits (Sections 4 and 5) ---------------------------------------

# Regular season and postseason (spring training dropped).
KEEP_GAME_TYPES: frozenset[str] = frozenset({"R", "F", "D", "L", "W"})

SPLITS: dict[int, str] = {**{y: "train" for y in range(2016, 2024)}, 2024: "val", 2025: "test"}

# Earlier pitches a pitcher needs before a pitch counts as an example.
MIN_PRIOR_PITCHES = 25

# Real-time order.
CHRONO_KEYS: list[str] = ["game_date", "game_pk", "at_bat_number", "pitch_number"]

# --- Raw columns kept by the clean step (Section 6) ---------------------------
# Dropped: future info, deprecated/empty, duplicates. Kept != model input.

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
