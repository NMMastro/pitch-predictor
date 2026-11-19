# How the pitch replay works

The replay is a Streamlit app for exploring recorded MLB pitches from 2025. It reads the clean Statcast Parquet for the matchup and recorded location. A separate prediction script supplies the forecast for the selected pitch.

## Follow a pitch through the app

1. **Choose a pitcher and game.** [replay.py](replay.py) reads the available matchups from the Parquet. The picker includes regular-season and postseason pitches for which the pitcher had already thrown at least 250 pitches. The page shows the pitcher, batter, handedness, inning, count, and outs from the selected record.
2. **Request a prediction.** [app.py](app.py) sends the pitcher ID, pitch key, and selected checkpoint path to [prediction.py](prediction.py). The pitch key is the game ID, at-bat number, and pitch number. The prediction client runs the configured Python script once and validates the JSON it writes to stdout.
3. **Reveal the actual zone.** [strike_zone.py](strike_zone.py) draws the 13 Statcast regions from the catcher’s view. A prediction with a zone fills that region in blue. The actual zone comes from the recorded pitch’s `zone_class` and gets a copper outline on reveal. Inside and away are labeled relative to where the batter stands. A pitch-type prediction appears in text.
4. **Move to another pitch.** Streamlit keeps the current selection and result in session state. A change of pitcher, game, or pitch clears both the prediction and reveal. Switching checkpoints also clears the result so you can compare weights on the same pitch.

## Where to make changes

| To change | Start here |
| --- | --- |
| Matchup layout, controls, and session state | [app.py](app.py) and [style.css](style.css) |
| Which records are available and how zone labels read | [replay.py](replay.py) |
| Strike-zone shapes and overlays | [strike_zone.py](strike_zone.py) |
| Command-line arguments and response validation | [prediction.py](prediction.py) |
| The sample zone shown during UI development | [placeholder_predict.py](placeholder_predict.py) |

## Connect a model

The app uses a labeled placeholder by default, which returns a sample zone derived from the pitch key. To use a model, configure `--predict-script` and one or more `--checkpoint` paths. The backend script handles features, model loading, and inference. It can support new weights or architectures through the same arguments and JSON response.

A backend response can include `pitch_type`, `zone`, or both, with optional probabilities. Zone IDs use the catcher’s view. The [backend agent prompt](BACKEND_AGENT_PROMPT.md) spells out the contract and error handling, and the [README](README.md) shows how to run the app. The Parquet, checkpoint files, and generated name cache are local inputs or artifacts.
