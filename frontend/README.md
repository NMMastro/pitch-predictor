# 2025 pitch replay

Choose a pitcher and game, request a prediction, then reveal the recorded Statcast zone. The replay includes regular and postseason pitches with at least 250 prior pitches for that pitcher. The diagram uses the catcher’s view, with inside and away labeled relative to the batter.

[How the frontend works](ARCHITECTURE.md) explains the replay flow, the file layout, and where the backend script fits.

## Run with the placeholder

Run from the repository root. Use the absolute Parquet path when the data lives in a separate worktree.

```bash
python -m venv frontend/.venv
source frontend/.venv/bin/activate
python -m pip install -r frontend/requirements.txt

PITCH_DATA=/absolute/path/to/data/clean/pitches.parquet
PYTHONPATH=src python -m frontend.prepare_names --clean "$PITCH_DATA"
PYTHONPATH=src python -m streamlit run frontend/app.py --theme.base light -- --clean "$PITCH_DATA"
```

The name setup caches player names for offline use. The app uses the MLB player ID as a fallback. Use `--names` after the Streamlit argument separator to select a different cache.

The default predictor is a labeled placeholder. It returns a sample zone based on the selected pitch key.

## Connect the team’s backend

Give the backend agent [this prompt](BACKEND_AGENT_PROMPT.md) to build the inference script. Once the script is ready, pass its path and the checkpoints you want to use.

```bash
PYTHONPATH=src python -m streamlit run frontend/app.py --theme.base light -- \
  --clean "$PITCH_DATA" \
  --predict-script /absolute/path/to/backend/src/models/transformer/predict.py \
  --backend-dir /absolute/path/to/backend \
  --backend-python /absolute/path/to/backend/.venv/bin/python \
  --checkpoint /absolute/path/to/run-a.pt \
  --checkpoint /absolute/path/to/run-b.pt
```

Repeat `--checkpoint` to get a model picker. The app passes the selected path and the exact pitch key to the backend on each click. The backend owns checkpoint loading, feature preparation, and inference. The frontend reads the prediction JSON and the recorded Statcast result. With a pitch-type checkpoint, the app displays the type alongside the recorded zone when revealed. A location overlay requires a zone in the backend response.

## Tests

```bash
python -m pip install pytest
PYTHONPATH=src python -m pytest -q frontend/tests
```
