# Prompt for the backend agent

Implement a single-pitch inference CLI at `src/models/transformer/predict.py` in `pitch-predictor`. The 2025 replay frontend already calls a script as a subprocess. Make the script load the checkpoint supplied with `--checkpoint` so we can switch between trained weights from the UI.

### Command

Support these named arguments. `game_pk`, `at_bat_number`, and `pitch_number` identify the exact pitch, and `--pitcher` is its MLB player ID.

```bash
python src/models/transformer/predict.py \
  --clean /absolute/path/to/pitches.parquet \
  --checkpoint /absolute/path/to/weights.pt \
  --pitcher 453286 --game-pk 778536 --at-bat-number 1 --pitch-number 1
```

Reuse the backend’s model and preprocessing with the selected checkpoint’s original training scope, history window, vocabulary, scalers, and class order. Use only pre-pitch context and earlier pitches as model inputs. Run in evaluation mode with gradients disabled and default to CPU. If an older checkpoint lacks enough metadata to reconstruct the correct input pipeline, report what is needed. This task is inference only.

### Response

Write exactly one JSON object to stdout. Return one or both prediction fields according to the checkpoint’s target. These are example values.

```json
{"pitch_type": "FF", "pitch_type_probability": 0.54}
```

```json
{"zone": 14, "zone_probability": 0.42}
```

Probabilities are optional numbers from 0 to 1. `zone` must be a Statcast integer from 1–9 or 11–14 in **catcher’s-view coordinates**. If the model encodes location relative to the batter, convert it back with the repo’s `MIRROR_ZONE` mapping for left-handed batters. A pitch-type-only checkpoint should return the type fields alone.

Write diagnostics to stderr. On a missing pitch, pitcher mismatch, or incompatible checkpoint, exit nonzero with a clear message. Keep training entry points compatible.

### Verify

Compare the CLI output for a held-out 2025 pitch with the existing backend inference path. Verify that the same command accepts two compatible checkpoint paths, that a type-only checkpoint emits the documented JSON, and that zone models map left-handed pitches into the catcher’s view. Return the tested command and example output for the frontend maintainer.
