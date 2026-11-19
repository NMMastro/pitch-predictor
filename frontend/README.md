# 2025 pitch replay

Choose a pitcher and game, run the zone transformer, then reveal the recorded zone. The diagram shows all 13 Statcast regions from the catcher’s view. Inside and away are relative to the batter. The replay includes regular season and postseason pitches with at least 250 prior pitches for that pitcher.

## Run the app

Run these commands from the repository root. Create a dedicated environment if another environment is in use for training.

```bash
python -m venv frontend/.venv
source frontend/.venv/bin/activate
python -m pip install -r requirements.txt -r frontend/requirements.txt

# Use an absolute path when running in a separate Git worktree.
PITCH_DATA=/absolute/path/to/data/clean/pitches.parquet
PYTHONPATH=src python -m frontend.prepare_names --clean "$PITCH_DATA"
PYTHONPATH=src python -m streamlit run frontend/app.py --theme.base light -- --clean "$PITCH_DATA"
```

The name setup downloads a player lookup table once. The app uses the resulting local cache. If the cache is missing a player, the app shows the player’s MLB ID. Use `--names` after the Streamlit argument separator to select a different cache.

## Train a zone checkpoint

The app looks for `frontend/artifacts/zone.pt`. To make one, use a machine with the prepared PyTorch environment and the clean 2016–2025 Parquet. If bringing over a checkpoint from another machine, pass the file with `--checkpoint` after the Streamlit argument separator.

```bash
PYTHONPATH=src python -m frontend.train_zone --clean "$PITCH_DATA" --device cuda
```

Use `--device cpu` to train on a CPU. For a quick end-to-end check with a small CPU model, use the Scherzer smoke run below. When launching the app, pass `--checkpoint frontend/artifacts/scherzer-smoke.pt`. The smoke checkpoint shows only Scherzer’s games.

```bash
PYTHONPATH=src python -m frontend.train_zone --clean "$PITCH_DATA" \
  --pitcher 453286 --epochs 1 --window 16 --d-model 32 --layers 1 \
  --output frontend/artifacts/scherzer-smoke.pt
```

The trainer shares the model architecture and Statcast encoding with the backend. It learns the 13 pitch zones from 2016–2023, saves the checkpoint with the lowest 2024 validation log loss, and prints log loss and accuracy on the 2025 holdout. Each model input uses strictly earlier pitch outcomes and the current pre-pitch context. `zone_data.py` prepares the same features for both training and replay. `replay.py` handles data loading and single-pitch inference. `app.py` handles the page, controls, and selection state.

## Local demo checkpoint

The initial league-wide demo checkpoint used one CPU epoch, a 16-pitch window, one encoder layer, and a model width of 32. On 718,873 labeled 2025 test pitches it scored 2.3766 log loss, 22.08% accuracy, and 34.94% top-2 accuracy. The UI applies its 250-prior-pitch demo cutoff separately.

```bash
PYTHONPATH=src python -m frontend.train_zone --clean "$PITCH_DATA" \
  --epochs 1 --window 16 --d-model 32 --layers 1 --batch-size 256
```

The checkpoint and batter name cache are local artifacts. Generate them on another checkout or copy them into `frontend/artifacts/`.

## Tests

```bash
PYTHONPATH=src python -m pytest -q tests frontend/tests
```
