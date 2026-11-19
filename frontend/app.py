"""The Streamlit pitch replay. Run from the repository root."""

import argparse
import pickle
import shlex
from collections import Counter
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from features.schema import CLEAN_PATH, PITCH_KEY, ZONE_CLASSES
from models.transformer.config import SCHERZER

from frontend.replay import (
    NAME_CACHE,
    REPLAY_YEAR,
    ZoneModel,
    load_pitcher,
    pitcher_name,
    read_catalog,
    read_names,
    zone_for_catcher,
    zone_label,
)
from frontend.strike_zone import chart_description, strike_zone
from frontend.train_zone import CHECKPOINT


@st.cache_resource(max_entries=2)
def cached_model(path: Path, modified_ns: int) -> ZoneModel:
    return ZoneModel.load(path)


@st.cache_data(max_entries=2)
def cached_catalog(path: Path, modified_ns: int):
    return read_catalog(path)


@st.cache_resource(max_entries=4)
def cached_pitcher(clean: Path, clean_version: int, pitcher: int, checkpoint: Path, model_version: int):
    model = cached_model(checkpoint, model_version)
    return load_pitcher(clean, pitcher, model.preprocessing)


@st.cache_data(max_entries=2)
def cached_names(path: Path, modified_ns: int):
    return read_names(path)


def reset_result():
    st.session_state.prediction = None
    st.session_state.revealed = False


def move_pitch(change: int):
    st.session_state.pitch_index += change
    reset_result()


def run_prediction(replay, model, pitch_key):
    st.session_state.prediction = replay.predict_zone(model, pitch_key)


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


def compare_zones(prediction, revealed, actual, stand):
    predicted_zone, probability = prediction if prediction else (None, None)
    predicted_summary = (
        f"{zone_label(predicted_zone)} · {probability:.0%}" if prediction else "Awaiting prediction"
    )
    actual_summary = "Hidden"
    if revealed:
        actual_summary = (
            zone_label(zone_for_catcher(actual, stand)) if actual is not None else "Zone unavailable"
        )
    return f"""
    <div class="zone-legend" role="status" aria-live="polite">
      <div><span class="legend-mark predicted-mark"></span><div>
        <strong>Predicted zone</strong><p>{predicted_summary}</p>
      </div></div>
      <div><span class="legend-mark actual-mark"></span><div>
        <strong>Actual zone</strong><p>{actual_summary}</p>
      </div></div>
    </div>
    """


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, default=CLEAN_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--names", type=Path, default=NAME_CACHE)
    args = parser.parse_args()

    st.set_page_config(page_title="2025 pitch replay", layout="centered")
    st.html(Path(__file__).with_name("style.css"))
    st.title("2025 pitch replay")
    clean = args.clean.resolve()
    checkpoint = args.checkpoint.resolve()
    if not clean.is_file():
        st.error(f"Clean Statcast data is missing at {clean}. Pass its path with `--clean`.")
        st.stop()
    try:
        model_version = checkpoint.stat().st_mtime_ns
        model = cached_model(checkpoint, model_version)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, EOFError, pickle.UnpicklingError):
        save_at = shlex.quote(str(checkpoint))
        data_at = shlex.quote(str(clean))
        st.error("Load a trained zone checkpoint to start the replay.")
        st.code(
            f"PYTHONPATH=src python -m frontend.train_zone --clean {data_at} --output {save_at}",
            language="bash",
        )
        st.stop()

    clean_version = clean.stat().st_mtime_ns
    games = cached_catalog(clean, clean_version)
    if model.training_pitcher is not None:
        games = games.loc[games["pitcher"] == model.training_pitcher]
        st.caption("Smoke checkpoint · One pitcher")
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
    defaults = pitcher_ids.index(SCHERZER) if SCHERZER in labels else 0
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
        replay = cached_pitcher(clean, clean_version, pitcher, checkpoint, model_version)
    except (OSError, ValueError, IndexError, KeyError) as error:
        st.error(f"Could not read this pitcher’s history. {error}")
        st.stop()
    try:
        names = cached_names(args.names, args.names.stat().st_mtime_ns)
    except (OSError, ValueError):
        names = {}

    game_pitches = replay.pitches.loc[replay.pitches["game_pk"] == game]
    selection = (pitcher, game, clean_version, model_version)
    if st.session_state.get("selection") != selection:
        st.session_state.selection = selection
        st.session_state.pitch_index = 0
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
    predicted = zone_for_catcher(prediction[0], pitch.stand) if prediction else None
    actual = None
    if revealed and pd.notna(pitch.zone_class) and pitch.zone_class in ZONE_CLASSES:
        actual = int(pitch.zone_class)
    st.html('<div class="chart-label">Catcher’s view</div>')
    st.image(
        strike_zone(pitch.stand, predicted, actual),
        width=448,
        alt=chart_description(pitch.stand, predicted, actual),
    )
    st.html(compare_zones(prediction, revealed, actual, pitch.stand))

    with st.container(key="replay-controls"):
        action, previous, following = st.columns((2, 1, 1))
        with action:
            if prediction is None:
                st.button(
                    "Predict zone", type="primary", width="stretch",
                    on_click=run_prediction, args=(replay, model, pitch_key),
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
