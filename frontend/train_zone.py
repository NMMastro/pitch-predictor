"""Train the location transformer with the repo's chronological season splits."""

import argparse
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import log_loss

from evaluation import evaluate
from features.schema import CLEAN_PATH, ZONE_CLASSES
from models.transformer.config import ModelConfig, TrainConfig
from models.transformer.train import set_seed, to_device, train_one_epoch
from models.transformer.transformer import PitchTransformer

from frontend.zone_data import (
    FEATURE_ORDER,
    encode_pitches,
    fit_preprocessing,
    make_loader,
    prepare_rows,
    read_pitches,
    split_rows,
    vocabulary_sizes,
)

CHECKPOINT = Path("frontend/artifacts/zone.pt")


@torch.inference_mode()
def predict_probabilities(model: PitchTransformer, batches) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    for batch in batches:
        logits = model(to_device(batch, device)).cpu().double()
        predictions.append(torch.softmax(logits, dim=-1).numpy())
    return np.concatenate(predictions)


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    print("Preparing shared pitch features", flush=True)
    encoded = encode_pitches(read_pitches(args.clean, args.pitcher))
    preprocessing = fit_preprocessing(encoded)
    rows = split_rows(encoded)
    features = prepare_rows(encoded, preprocessing)
    del encoded
    print("Eligible labeled pitches " + ", ".join(f"{key} {len(value):,}" for key, value in rows.items()), flush=True)
    if any(not len(value) for value in rows.values()):
        raise ValueError("The selected data needs labeled pitches in train, validation and test.")
    config = ModelConfig(
        vocab_sizes=vocabulary_sizes(preprocessing),
        n_classes=len(ZONE_CLASSES),
        n_ctx_numeric=len(FEATURE_ORDER["ctx_num"]),
        window=args.window,
        d_model=args.d_model,
        n_layers=args.layers,
        d_ff=2 * args.d_model,
    )
    fit = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, device=args.device)
    model = PitchTransformer(config).to(fit.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=fit.lr, weight_decay=fit.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    loaders = {
        name: make_loader(features, indices, config.window, fit.batch_size, shuffle=name == "train")
        for name, indices in rows.items()
    }
    class_ids = list(range(len(ZONE_CLASSES)))
    best_loss = float("inf")
    for epoch in range(1, fit.epochs + 1):
        started = time.monotonic()
        loss = train_one_epoch(model, loaders["train"], optimizer, criterion, fit)
        probabilities = predict_probabilities(model, loaders["val"])
        validation_labels = features.labels[rows["val"]]
        validation_loss = log_loss(validation_labels, probabilities, labels=class_ids)
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_checkpoint(args.output, {
                "target": "zone",
                "zone_classes": ZONE_CLASSES,
                "model_config": asdict(config),
                "state_dict": {name: value.cpu() for name, value in model.state_dict().items()},
                "preprocessing": preprocessing,
                "training_pitcher": args.pitcher,
                "epoch": epoch,
                "validation_log_loss": validation_loss,
            })
        print(
            f"Epoch {epoch}/{fit.epochs}  train loss {loss:.4f}  "
            f"2024 log loss {validation_loss:.4f}  {time.monotonic() - started:.1f}s",
            flush=True,
        )

    checkpoint = torch.load(args.output, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_zones = [ZONE_CLASSES[index] for index in features.labels[rows["test"]]]
    metrics = evaluate(test_zones, predict_probabilities(model, loaders["test"]), ZONE_CLASSES, per_class=False)
    print("2025 holdout " + json.dumps(metrics), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, default=CLEAN_PATH)
    parser.add_argument("--output", type=Path, default=CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--window", type=int, default=ModelConfig.window)
    parser.add_argument("--d-model", type=int, default=ModelConfig.d_model)
    parser.add_argument("--layers", type=int, default=ModelConfig.n_layers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--pitcher", type=int, help="Limit a smoke run to this MLB player ID.")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.window, args.layers, args.d_model) < 1:
        parser.error("epochs, batch-size, window, layers and d-model must be positive")
    train(args)


if __name__ == "__main__":
    main()
