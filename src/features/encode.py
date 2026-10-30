"""Encode step: clean pitch table -> every pitch as numbers, shared by both models.

Input:  the clean table (from clean.py), as a DataFrame.
Output: the same rows in rid order, the text columns replaced by the columns
        below. Kept in memory: tabular_features.py and sequence_features.py
        both call encode() on the table they load.

Each column describes one pitch or the state before it. The only look at other
pitches is whether the pitcher's previous pitch was in the same plate appearance.

Steps, in the order encode() runs them:
    1. pre_pitch_flags   capped inning, handedness, inning half, first-pitch flags
    2. pitch_codes       the pitch as integer codes, in the batter's view
    3. drop schema.TEXT_COLUMNS, now replaced by flags

Columns
-------
pre_pitch_flags
  inning                  capped at INNING_CAP
  is_LHB                  1 if the batter bats left          (replaces stand)
  is_LHP                  1 if the pitcher throws left       (replaces p_throws)
  same_hand               1 if batter and pitcher are the same hand
  is_inning_top           1 in the top of the inning         (replaces inning_topbot)
  is_first_pitch_of_ab    1 on the at-bat's first pitch
  is_first_pitch_of_pa    1 if the pitcher's previous pitch was in another plate appearance

pitch_codes: position in the schema vocabulary, -1 = none
  type_code               PITCH_CLASSES
  family_code             FAMILIES
  zone_code               ZONE_CLASSES, batter's view: a left-handed batter's
                          zones are mirrored, so "in" is inside for everyone
  result_code             description grouped into PITCH_RESULTS
  pa_result_code          events grouped into PA_RESULTS (a plate appearance's last pitch only)
  plate_x_rel             plate_x, batter's view (negative = inside)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.schema import (
    FAMILIES,
    INNING_CAP,
    MIRROR_ZONE,
    PA_RESULTS,
    PITCH_RESULTS,
    TEXT_COLUMNS,
    ZONE_CLASSES,
)


def encode(table: pd.DataFrame) -> pd.DataFrame:
    """Clean table -> the same rows in rid order, every pitch as numbers. Fits nothing."""
    table = table.sort_values("rid").reset_index(drop=True)
    table = pre_pitch_flags(table)
    table = pitch_codes(table)
    return table.drop(columns=TEXT_COLUMNS)


def pre_pitch_flags(df: pd.DataFrame) -> pd.DataFrame:
    """The capped inning, and handedness, inning half and first pitches as 0/1."""
    lhb = df["stand"] == "L"
    lhp = df["p_throws"] == "L"
    pitcher_pa = df.groupby(["pitcher", "game_pk", "at_bat_number"], sort=False)
    return df.assign(
        inning=df["inning"].clip(upper=INNING_CAP),
        is_LHB=lhb.astype(np.int8),
        is_LHP=lhp.astype(np.int8),
        same_hand=(lhb == lhp).astype(np.int8),
        is_inning_top=(df["inning_topbot"] == "Top").astype(np.int8),
        is_first_pitch_of_ab=(df["pitch_number"] == 1).astype(np.int8),
        is_first_pitch_of_pa=(pitcher_pa.cumcount() == 0).astype(np.int8),
    )


def pitch_codes(df: pd.DataFrame) -> pd.DataFrame:
    """The pitch as integer codes (-1 = none), with zone and plate_x in the batter's view."""
    lhb = (df["stand"] == "L").to_numpy()
    zone = df["zone_class"].astype("Int64")
    zone_rel = zone.where(~lhb, zone.map(MIRROR_ZONE))

    family_of = {c: f for f, classes in FAMILIES.items() for c in classes}
    result_of = {d: r for r, descriptions in PITCH_RESULTS.items() for d in descriptions}
    pa_result_of = {e: r for r, events in PA_RESULTS.items() for e in events}

    return df.assign(
        type_code=df["pitch_class"].cat.codes.to_numpy(),
        family_code=_codes(df["pitch_class"].astype(object).map(family_of), list(FAMILIES)),
        zone_code=_codes(zone_rel, ZONE_CLASSES),
        result_code=_codes(df["description"].astype(object).map(result_of), list(PITCH_RESULTS)),
        pa_result_code=_codes(df["events"].astype(object).map(pa_result_of), list(PA_RESULTS)),
        plate_x_rel=np.where(lhb, -df["plate_x"], df["plate_x"]),
    )


def _codes(values: pd.Series, labels: list) -> np.ndarray:
    """Position of each value in ``labels``; -1 if missing or not listed."""
    index = {label: i for i, label in enumerate(labels)}
    return values.astype(object).map(index).fillna(-1).to_numpy().astype(np.int8)
