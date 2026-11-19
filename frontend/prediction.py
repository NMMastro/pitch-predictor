"""Call the backend's command-line predictor and read its JSON response."""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from features.schema import ZONE_CLASSES

from frontend.replay import PitchKey


class PredictionError(Exception):
    """The predictor could not return a usable prediction."""


@dataclass(frozen=True)
class Prediction:
    zone: int | None = None
    zone_probability: float | None = None
    pitch_type: str | None = None
    pitch_type_probability: float | None = None


def probability(response: dict, field: str) -> float | None:
    value = response.get(field)
    if value is None:
        return None
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise PredictionError(f"The backend must return {field} as a number from 0 to 1.")
    return float(value)


def parse_prediction(stdout: str) -> Prediction:
    try:
        response = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise PredictionError("The backend must write one JSON object to stdout.") from error
    if not isinstance(response, dict):
        raise PredictionError("The backend must write one JSON object to stdout.")

    zone = response.get("zone")
    pitch_type = response.get("pitch_type")
    if zone is not None and (type(zone) is not int or zone not in ZONE_CLASSES):
        raise PredictionError("The backend returned an invalid Statcast zone.")
    if pitch_type is not None and (not isinstance(pitch_type, str) or not pitch_type.strip()):
        raise PredictionError("The backend must return pitch_type as text.")
    if zone is None and pitch_type is None:
        raise PredictionError("The backend response must include a zone or a pitch type.")

    return Prediction(
        zone=zone,
        zone_probability=probability(response, "zone_probability"),
        pitch_type=pitch_type.strip() if pitch_type is not None else None,
        pitch_type_probability=probability(response, "pitch_type_probability"),
    )


@dataclass(frozen=True)
class Predictor:
    script: Path
    python: str
    directory: Path | None = None

    def predict(
        self, *, clean: Path, checkpoint: Path | None, pitcher: int, pitch_key: PitchKey,
    ) -> Prediction:
        game, at_bat, pitch = pitch_key
        command = [
            self.python, str(self.script),
            "--clean", str(clean),
            "--pitcher", str(pitcher),
            "--game-pk", str(game),
            "--at-bat-number", str(at_bat),
            "--pitch-number", str(pitch),
        ]
        if checkpoint is not None:
            command.extend(["--checkpoint", str(checkpoint)])

        try:
            result = subprocess.run(
                command, cwd=self.directory, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=90, check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise PredictionError("The backend took longer than 90 seconds. Try again.") from error
        except OSError as error:
            raise PredictionError(f"Could not start the backend. {error}") from error

        if result.returncode != 0:
            messages = result.stderr.strip().splitlines()
            detail = messages[-1] if messages else f"Exit code {result.returncode}."
            raise PredictionError(f"The backend could not predict this pitch. {detail}")
        return parse_prediction(result.stdout)
