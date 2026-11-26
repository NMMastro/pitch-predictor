"""Load the best checkpoint and show ten held-out pitch predictions.

Run from the project root with:

    .venv/bin/python demo_prediction.py 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from features.schema import CLEAN_PATH, PITCH_CLASSES  # noqa: E402
from model.config import SCHERZER, ModelConfig, TrainConfig  # noqa: E402
from model.dataset import load_splits, make_loaders  # noqa: E402
from model.transformer import build_model  # noqa: E402


CHECKPOINT = Path(
    "results/overnight/20261110_224536/checkpoints/"
    "pitch_type_w256_seed1.pt"
)


def parse_args() -> argparse.Namespace:
    """Read how many held-out pitches to display."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "count",
        type=int,
        nargs="?",
        default=10,
        help="number of 2025 test pitches to predict (default: 10)",
    )
    args = parser.parse_args()

    if args.count < 1:
        parser.error("count must be at least 1")

    return args


def main(count: int) -> None:
    """Print top-three predictions for unseen pitches from 2025."""
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])

    # Rebuild the same feature vocabularies and 256-pitch history windows.
    splits = load_splits(
        CLEAN_PATH,
        pitcher=SCHERZER,
        window=checkpoint["window"],
    )

    # Recreate the model structure, then restore its learned weights.
    model = build_model(splits, config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Load held-out test examples that were never used to train the model.
    loader_config = TrainConfig(batch_size=count)
    _, _, test_loader = make_loaders(
        splits,
        loader_config,
        shuffle_train=False,
    )
    batch = next(iter(test_loader))

    # Convert raw model scores into probabilities for each pitch class.
    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=-1)

    correct = 0
    rows_to_show = min(count, len(batch["y"]))

    for row in range(rows_to_show):
        top_probabilities, top_indices = probabilities[row].topk(3)
        actual_pitch = PITCH_CLASSES[batch["y"][row].item()]
        predicted_pitch = PITCH_CLASSES[top_indices[0].item()]
        is_correct = predicted_pitch == actual_pitch
        correct += int(is_correct)

        marker = "correct" if is_correct else "incorrect"
        print(f"Pitch {row + 1}: actual {actual_pitch} ({marker})")

        for probability, index in zip(top_probabilities, top_indices):
            pitch = PITCH_CLASSES[index.item()]
            print(f"  {pitch}: {probability.item():.1%}")

        print()

    print(f"Top-1 accuracy for these {rows_to_show} pitches: {correct}/{rows_to_show}")


if __name__ == "__main__":
    main(parse_args().count)
