"""Train the pitch-type transformer and score it against a baseline.

One run: build the arrays, fit the model on 2016-2023, pick the epoch with the
best validation log loss, and score that checkpoint. Every run appends a row to
``results/scoreboard.jsonl`` through ``evaluation.log_run``, so runs stay
comparable.

The baseline matters more than it sounds. Predicting the training pitch mix for
every pitch, ignoring all features, is already a decent model of a pitcher --
Scherzer throws a fastball about half the time. A transformer that does not
clearly beat that has learned nothing, and it is easy to build one that looks
fine by accuracy alone while doing exactly as well as the baseline.

Scoring follows DESIGN_DECISIONS.md Open Decision 2: the head stays ten-wide,
but metrics are computed over only the classes the pitcher actually throws, by
slicing those columns out and renormalising.

Usage:
    python src/model/train.py                        # Scherzer, window 64
    python src/model/train.py --epochs 20 --window 400
    python src/model/train.py --pitcher 0            # every pitcher (slow)
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import evaluate, log_run  # noqa: E402
from features.schema import CLEAN_PATH, PITCH_CLASSES  # noqa: E402
from model.config import SCHERZER, ModelConfig, TrainConfig  # noqa: E402
from model.dataset import load_splits, make_loaders  # noqa: E402
from model.transformer import build_model, count_parameters  # noqa: E402

SCOREBOARD = Path("results/scoreboard.jsonl")
CHECKPOINT_DIR = Path("results/checkpoints")


# ==============================================================================
# THE TWO PIECES YOU WRITE
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: TrainConfig,
) -> float:
    """One pass over the training data. Returns the mean loss per example.

    The standard PyTorch training step, in order: zero the gradients, run the
    forward pass, compute the loss, backpropagate, optionally clip, then step.
    Getting the order wrong usually still runs -- it just does not learn, which
    is why ``tests/model/test_train.py`` checks that the loss actually falls.

    Parameters
    ----------
    model:
        The :class:`~model.transformer.PitchTransformer`. Its forward takes the
        whole batch dict and returns ``(B, n_classes)`` logits.
    loader:
        Yields batch dicts with ``y`` holding the labels.
    criterion:
        ``nn.CrossEntropyLoss``, which expects logits and integer labels.
    config:
        ``config.grad_clip`` is the maximum gradient norm, or None.

    Returns
    -------
    float
        Mean loss per example over the epoch. Weight each batch by its size,
        because the last batch is usually smaller than the rest.
    """
    # TODO 1: put the model in training mode, so dropout is active.
    # TODO 2: loop over the loader. For each batch:
    #           a. clear the gradients left over from the previous step
    #           b. forward pass: logits = model(batch)
    #           c. loss = criterion(logits, batch["y"])
    #           d. backward pass
    #           e. if config.grad_clip is not None, clip the gradient norm with
    #              nn.utils.clip_grad_norm_
    #           f. take an optimiser step
    # TODO 3: accumulate loss.item() * len(batch["y"]) and the row count, then
    #         return the weighted mean. Use .item(): keeping the tensor alive
    #         keeps its whole computation graph alive with it.
    raise NotImplementedError("Task 8: write the training step")


@torch.no_grad()
def predict_proba(model: nn.Module, loader) -> np.ndarray:
    """Class probabilities for every row of ``loader``, in loader order.

    Inference, not training: the model goes into eval mode so dropout is off,
    and no gradients are needed. The head returns logits, so this is where the
    softmax happens -- exactly once, because ``evaluation.evaluate`` rejects
    anything whose rows do not sum to 1.

    Returns
    -------
    np.ndarray
        ``(n_rows, n_classes)`` float64, each row summing to 1. Row order must
        match the loader's, or the predictions will be scored against the wrong
        labels; the val and test loaders do not shuffle, for this reason.
    """
    # The @torch.no_grad() decorator already covers the gradient half.
    # TODO 1: put the model in eval mode.
    # TODO 2: for each batch, run the model and softmax the logits over the
    #         class dimension (dim=-1). Cast to float64 before the softmax --
    #         in float32 the rows sum to 1 only to about 1e-7, and
    #         evaluate() checks.
    # TODO 3: concatenate the batches in order and return one NumPy array of
    #         shape (n_rows, n_classes).
    raise NotImplementedError("Task 9: write the prediction step")


# ==============================================================================
# EVERYTHING ELSE IS WRITTEN
# ==============================================================================

def set_seed(seed: int) -> None:
    """Seed every generator a run touches, so two runs of a config match."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def labels_of(loader) -> np.ndarray:
    """The labels of a loader's rows, in loader order."""
    return torch.cat([batch["y"] for batch in loader]).numpy()


def restrict(proba: np.ndarray, classes: list[int]) -> np.ndarray:
    """Keep only ``classes``' columns and renormalise the rows.

    Open Decision 2: the head predicts all ten classes, but a single pitcher
    throws about five, so macro metrics over ten would be dragged down by
    classes that cannot occur. Whatever probability the model put on those is
    redistributed proportionally rather than dropped.
    """
    kept = proba[:, classes]
    totals = kept.sum(axis=1, keepdims=True)
    return kept / np.where(totals > 0, totals, 1.0)


def score(y: np.ndarray, proba: np.ndarray, classes: list[int]) -> dict:
    """Score predictions over the restricted label set.

    ``y`` holds indices into :data:`~features.schema.PITCH_CLASSES`; ``evaluate``
    wants the class values themselves, so they are mapped to names here.
    """
    names = [PITCH_CLASSES[i] for i in classes]
    keep = np.isin(y, classes)
    if not keep.all():
        unexpected = sorted({PITCH_CLASSES[i] for i in y[~keep]})
        print(f"  dropping {int((~keep).sum())} rows of classes unseen in train: {unexpected}")
    return evaluate([PITCH_CLASSES[i] for i in y[keep]], restrict(proba, classes)[keep], names)


def baseline_proba(class_counts: np.ndarray, n_rows: int) -> np.ndarray:
    """The training pitch mix, predicted for every row. The bar to beat."""
    mix = class_counts / class_counts.sum()
    return np.tile(mix, (n_rows, 1))


def run(splits, model_config: ModelConfig, train_config: TrainConfig) -> dict:
    """Train, keep the best epoch by validation log loss, and score it."""
    set_seed(train_config.seed)
    train_loader, val_loader, test_loader = make_loaders(splits, train_config)

    # The baseline first: it needs no model, so it still prints while the
    # model is a stub, and it is the number every epoch below is judged against.
    y_val = labels_of(val_loader)
    classes = splits.train_classes
    base = score(y_val, baseline_proba(splits.class_counts, len(y_val)), classes)
    print(f"baseline (train pitch mix)  val log loss {base['log_loss']:.4f}  "
          f"accuracy {base['accuracy']:.3f}\n")

    model = build_model(splits, model_config).to(train_config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    print(f"{count_parameters(model):,} trainable parameters\n")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT_DIR / f"pitch_type_w{splits.window}.pt"
    best = {"log_loss": float("inf"), "epoch": -1}

    for epoch in range(1, train_config.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, train_config)
        val = score(y_val, predict_proba(model, val_loader), classes)

        flag = ""
        if val["log_loss"] < best["log_loss"]:
            best = {**val, "epoch": epoch}
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": asdict(model.config),
                    "train_classes": classes,
                    "window": splits.window,
                    "epoch": epoch,
                },
                checkpoint,
            )
            flag = "  <- best"

        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val['log_loss']:.4f}  "
              f"acc {val['accuracy']:.3f}  top2 {val['top2_accuracy']:.3f}  "
              f"{time.time() - start:.0f}s{flag}")

    if best["epoch"] < 0:
        raise RuntimeError("no epochs ran; --epochs must be at least 1")

    print(f"\nbest epoch {best['epoch']}, val log loss {best['log_loss']:.4f} "
          f"vs baseline {base['log_loss']:.4f} -> saved {checkpoint}")

    model.load_state_dict(torch.load(checkpoint, weights_only=False)["state_dict"])
    y_test = labels_of(test_loader)
    test = score(y_test, predict_proba(model, test_loader), classes)
    print(f"test log loss {test['log_loss']:.4f}  accuracy {test['accuracy']:.3f}  "
          f"top2 {test['top2_accuracy']:.3f}")

    return {
        "model": model,
        "baseline": base,
        "val": best,
        "test": test,
        "checkpoint": checkpoint,
        "classes": [PITCH_CLASSES[i] for i in classes],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    data = parser.add_argument_group("data")
    data.add_argument("--clean", type=Path, default=CLEAN_PATH)
    data.add_argument("--pitcher", type=int, default=SCHERZER, help="MLBAM id; 0 = every pitcher")
    data.add_argument("--window", type=int, default=ModelConfig.window)

    net = parser.add_argument_group("model")
    net.add_argument("--d-model", type=int, default=ModelConfig.d_model)
    net.add_argument("--n-heads", type=int, default=ModelConfig.n_heads)
    net.add_argument("--n-layers", type=int, default=ModelConfig.n_layers)
    net.add_argument("--d-ff", type=int, default=ModelConfig.d_ff)
    net.add_argument("--dropout", type=float, default=ModelConfig.dropout)

    fit = parser.add_argument_group("training")
    fit.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    fit.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    fit.add_argument("--lr", type=float, default=TrainConfig.lr)
    fit.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    fit.add_argument("--seed", type=int, default=TrainConfig.seed)

    parser.add_argument("--scoreboard", type=Path, default=SCOREBOARD)
    parser.add_argument("--no-log", action="store_true", help="skip the scoreboard row")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    model_config = ModelConfig(
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        d_ff=args.d_ff, dropout=args.dropout, window=args.window,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay,
        epochs=args.epochs, seed=args.seed,
    )

    splits = load_splits(args.clean, args.pitcher or None, args.window)
    print(f"\n{splits.summary()}\n")

    result = run(splits, model_config, train_config)

    if not args.no_log:
        row = log_run(
            args.scoreboard,
            result["val"],
            model="transformer",
            target="pitch_type",
            features_version="sequence_features@shifted-codes+pad-mask",
            split="val",
            pitcher=args.pitcher,
            classes=result["classes"],
            baseline_log_loss=result["baseline"]["log_loss"],
            test_log_loss=result["test"]["log_loss"],
            **{f"cfg_{k}": v for k, v in asdict(train_config).items()},
        )
        print(f"\nlogged run to {args.scoreboard} at {row['run_at']}")


if __name__ == "__main__":
    main()
