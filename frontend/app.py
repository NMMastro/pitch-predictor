"""The Streamlit pitch replay. Run from the repository root."""

import argparse
import sys
from collections import Counter
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from features.schema import CLEAN_PATH, PITCH_KEY, ZONE_CLASSES

from frontend.prediction import PredictionError, Predictor
from frontend.replay import (
    DEFAULT_PITCHER,
    NAME_CACHE,
    REPLAY_YEAR,
    load_pitcher,
    pitcher_name,
    read_catalog,
    read_names,
    zone_for_catcher,
    zone_label,
)
from frontend.strike_zone import chart_description, strike_zone


@st.cache_data(max_entries=2)
def cached_catalog(path: Path, modified_ns: int):
    return read_catalog(path)


@st.cache_data(max_entries=4)
def cached_pitcher(clean: Path, clean_version: int, pitcher: int):
    return load_pitcher(clean, pitcher)


@st.cache_data(max_entries=2)
def cached_names(path: Path, modified_ns: int):
    return read_names(path)


def reset_result():
    st.session_state.prediction = None
    st.session_state.prediction_error = None
    st.session_state.revealed = False


def move_pitch(change: int):
    st.session_state.pitch_index += change
    reset_result()


def run_prediction(predictor, clean, checkpoint, pitcher, pitch_key):
    st.session_state.prediction_error = None
    try:
        with st.spinner("Waiting for the prediction…"):
            st.session_state.prediction = predictor.predict(
                clean=clean, checkpoint=checkpoint, pitcher=pitcher, pitch_key=pitch_key,
            )
    except PredictionError as error:
        st.session_state.prediction_error = str(error)


def reveal_zone():
    st.session_state.revealed = True


def matchup(pitch, names):
    top = pitch.inning_topbot == "Top"
    batting_team, fielding_team = (
        (pitch.away_team, pitch.home_team) if top else (pitch.home_team, pitch.away_team)
    )
    pitcher = escape(pitcher_name(pitch.player_name))
    batter = escape(names.get(pitch.batter, f"Player {pitch.batter}"))
    throwing = {"R": "Throws right", "L": "Throws left"}.get(pitch.p_throws, "Hand unknown")
    batting = {"R": "Bats right", "L": "Bats left"}.get(pitch.stand, "Side unknown")
    return f"""
    <div class="matchup">
      <section>
        <div class="profile-role">Pitcher</div><h2>{pitcher}</h2>
        <div>{escape(fielding_team)} · {throwing}</div>
      </section>
      <section>
        <div class="profile-role">Batter</div><h2>{batter}</h2>
        <div>{escape(batting_team)} · {batting}</div>
      </section>
    </div>
    """


def format_prediction(label, probability):
    return f"{label} · {probability:.0%}" if probability is not None else label


def compare_zones(prediction, revealed, actual, stand, placeholder):
    predicted_heading = "Placeholder zone" if placeholder else "Predicted zone"
    predicted_summary = "Awaiting prediction"
    predicted_type = ""
    if prediction and prediction.zone is not None:
        relative = zone_for_catcher(prediction.zone, stand)
        predicted_summary = format_prediction(zone_label(relative), prediction.zone_probability)
    if prediction and prediction.pitch_type:
        pitch_type = escape(format_prediction(prediction.pitch_type, prediction.pitch_type_probability))
        if prediction.zone is None:
            predicted_heading, predicted_summary = "Predicted pitch type", pitch_type
        else:
            predicted_type = f"<p>Pitch type {pitch_type}</p>"
    actual_summary = "Hidden"
    if revealed:
        actual_summary = (
            zone_label(zone_for_catcher(actual, stand)) if actual is not None else "Zone unavailable"
        )
    return f"""
    <div class="zone-legend" role="status" aria-live="polite">
      <div><span class="legend-mark predicted-mark"></span><div>
        <strong>{predicted_heading}</strong><p>{predicted_summary}</p>{predicted_type}
      </div></div>
      <div><span class="legend-mark actual-mark"></span><div>
        <strong>Actual zone</strong><p>{actual_summary}</p>
      </div></div>
    </div>
    """


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, default=CLEAN_PATH)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--predict-script", type=Path)
    parser.add_argument("--backend-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--backend-dir", type=Path)
    parser.add_argument("--names", type=Path, default=NAME_CACHE)
    args = parser.parse_args()

    st.set_page_config(page_title="2025 pitch replay", layout="centered")
    st.html(Path(__file__).with_name("style.css"))
    st.title("2025 pitch replay")
    clean = args.clean.resolve()
    if not clean.is_file():
        st.error(f"Clean Statcast data is missing at {clean}. Pass its path with `--clean`.")
        st.stop()

    placeholder_script = Path(__file__).resolve().with_name("placeholder_predict.py")
    script = args.predict_script.expanduser().resolve() if args.predict_script else placeholder_script
    placeholder = script == placeholder_script
    predictor = Predictor(
        script=script,
        python=str(args.backend_python.expanduser().absolute()),
        directory=args.backend_dir.expanduser().resolve() if args.backend_dir else None,
    )
    checkpoints = list(dict.fromkeys(path.expanduser().resolve() for path in args.checkpoint))
    checkpoint = checkpoints[0] if checkpoints else None
    predictor_issue = None
    if placeholder:
        st.caption("Placeholder mode · Sample zones for UI testing")
    else:
        if len(checkpoints) > 1:
            checkpoint = st.selectbox("Checkpoint", checkpoints, format_func=str)
        elif checkpoint is not None:
            st.caption(f"Checkpoint · {checkpoint.parent.name}/{checkpoint.name}")

        if not script.is_file():
            predictor_issue = f"Backend script is missing at {script}."
        elif checkpoint is None:
            predictor_issue = "Add a backend checkpoint with `--checkpoint`."
        elif not checkpoint.is_file():
            predictor_issue = f"Checkpoint is missing at {checkpoint}."
    if predictor_issue:
        st.warning(predictor_issue)

    clean_version = clean.stat().st_mtime_ns
    games = cached_catalog(clean, clean_version)
    if games.empty:
        st.info(f"No eligible {REPLAY_YEAR} games were found in this dataset.")
        st.stop()

    pitchers = games.drop_duplicates("pitcher").set_index("pitcher")["player_name"]
    labels = {int(player): pitcher_name(str(name)) for player, name in pitchers.items()}
    name_counts = Counter(labels.values())
    labels = {
        player: f"{name} · {player}" if name_counts[name] > 1 else name
        for player, name in labels.items()
    }
    pitcher_ids = sorted(labels, key=lambda player: labels[player].casefold())
    defaults = pitcher_ids.index(DEFAULT_PITCHER) if DEFAULT_PITCHER in labels else 0
    picker, matchup_filter = st.columns((2, 3))
    with picker:
        pitcher = st.selectbox("Pitcher", pitcher_ids, index=defaults, format_func=labels.__getitem__)

    player_games = games.loc[games["pitcher"] == pitcher]
    game_labels = {
        row.game_pk: f"{row.game_date:%b %d} · {row.away_team} at {row.home_team}"
        for row in player_games.itertuples()
    }
    matchup_counts = Counter(game_labels.values())
    game_labels = {
        game: f"{label} · Game {game}" if matchup_counts[label] > 1 else label
        for game, label in game_labels.items()
    }
    with matchup_filter:
        game = st.selectbox("Game", list(game_labels), format_func=game_labels.__getitem__, key=f"game-{pitcher}")

    try:
        pitches = cached_pitcher(clean, clean_version, pitcher)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Could not read this pitcher’s 2025 pitches. {error}")
        st.stop()
    try:
        names = cached_names(args.names, args.names.stat().st_mtime_ns)
    except (OSError, ValueError):
        names = {}

    game_pitches = pitches.loc[pitches["game_pk"] == game]
    selection = (pitcher, game, clean_version)
    if st.session_state.get("selection") != selection:
        st.session_state.selection = selection
        st.session_state.pitch_index = 0
        reset_result()
    if "checkpoint" not in st.session_state or st.session_state.checkpoint != checkpoint:
        st.session_state.checkpoint = checkpoint
        reset_result()

    index = st.session_state.pitch_index
    pitch = game_pitches.iloc[index]
    pitch_key = tuple(int(pitch[column]) for column in PITCH_KEY)
    st.html(matchup(pitch, names))

    half = "Top" if pitch.inning_topbot == "Top" else "Bottom"
    st.html(f"""
    <div class="pitch-context">
      <span>{half} {pitch.inning}</span><span>Count {pitch.balls}–{pitch.strikes}</span>
      <span>{pitch.outs_when_up} out{'' if pitch.outs_when_up == 1 else 's'}</span>
      <span>Pitch {index + 1} of {len(game_pitches)}</span>
    </div>
    """)

    prediction = st.session_state.prediction
    revealed = st.session_state.revealed
    predicted = prediction.zone if prediction else None
    actual = None
    if revealed and pd.notna(pitch.zone_class) and pitch.zone_class in ZONE_CLASSES:
        actual = int(pitch.zone_class)
    st.html('<div class="chart-label">Catcher’s view</div>')
    st.image(
        strike_zone(pitch.stand, predicted, actual, placeholder=placeholder),
        width=448,
        alt=chart_description(pitch.stand, predicted, actual, placeholder=placeholder),
    )
    st.html(compare_zones(prediction, revealed, actual, pitch.stand, placeholder))
    if st.session_state.prediction_error:
        st.error(st.session_state.prediction_error)

    with st.container(key="replay-controls"):
        action, previous, following = st.columns((2, 1, 1))
        with action:
            if prediction is None and not predictor_issue:
                st.button(
                    "Show placeholder" if placeholder else "Run prediction",
                    type="primary", width="stretch", on_click=run_prediction,
                    args=(predictor, clean, checkpoint, pitcher, pitch_key),
                )
            else:
                st.button(
                    "Reveal actual zone", type="primary", width="stretch",
                    disabled=revealed, on_click=reveal_zone,
                )
        with previous:
            st.button(
                "Previous pitch", disabled=index == 0, width="stretch",
                on_click=move_pitch, args=(-1,),
            )
        with following:
            st.button(
                "Next pitch", key="next-pitch", type="primary" if revealed else "secondary",
                disabled=index + 1 == len(game_pitches), width="stretch",
                on_click=move_pitch, args=(1,),
            )
    st.caption("2025 MLB Statcast · Held-out season")


if __name__ == "__main__":
    main()
