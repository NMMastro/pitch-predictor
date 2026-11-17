"""Tests for the training step and the prediction step.

These are the two pieces that look right and silently do nothing. A training
loop missing ``optimizer.step()`` runs at full speed and never learns; a
``predict_proba`` that forgets the softmax returns numbers that
``evaluation.evaluate`` rejects outright.

The overfit test is the one that matters. A model given one batch over and over
should be able to memorise it, and if it cannot, something in the loop is
broken. It is the standard first check on any new training code.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from conftest import FAKE_N_CLASSES, FAKE_WINDOW as W
from model.config import ModelConfig, TrainConfig
from model.dataset import CTX_NUMERIC, SEQ_CATEGORICAL
from model.train import predict_proba, train_one_epoch
from model.transformer import PitchTransformer


@pytest.fixture
def config(vocab_sizes) -> ModelConfig:
    return ModelConfig(
        vocab_sizes=vocab_sizes,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        dropout=0.0,  # off, so the overfit test is not fighting noise
        window=W,
        n_classes=FAKE_N_CLASSES,
        n_seq_numeric=len(SEQ_CATEGORICAL),
        n_ctx_numeric=len(CTX_NUMERIC),
    )


@pytest.fixture
def model(config) -> PitchTransformer:
    torch.manual_seed(0)
    return PitchTransformer(config)


class OneBatchLoader:
    """A loader that yields the same batch every time, `repeats` times."""

    def __init__(self, batch: dict, repeats: int = 1) -> None:
        self.batch = batch
        self.repeats = repeats

    def __iter__(self):
        for _ in range(self.repeats):
            yield self.batch

    def __len__(self) -> int:
        return self.repeats


# --- Task 8: train_one_epoch --------------------------------------------------

def test_train_one_epoch_returns_a_plausible_mean_loss(model, fake_batch):
    loader = OneBatchLoader(fake_batch, repeats=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, loader, optimizer, nn.CrossEntropyLoss(), TrainConfig())

    assert isinstance(loss, float)
    assert np.isfinite(loss)
    # An untrained 10-class model should start near ln(10) = 2.30.
    assert 0.0 < loss < 10.0


def test_train_one_epoch_actually_updates_the_weights(model, fake_batch):
    """Without optimizer.step() the loop runs happily and learns nothing."""
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_one_epoch(model, OneBatchLoader(fake_batch), optimizer,
                    nn.CrossEntropyLoss(), TrainConfig())

    changed = [
        name for name, p in model.named_parameters()
        if not torch.equal(before[name], p.detach())
    ]
    assert changed, "no parameter changed; is optimizer.step() being called?"


def test_train_one_epoch_leaves_the_model_in_training_mode(model, fake_batch):
    """Dropout has to be on while training, so the loop must call model.train()."""
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_one_epoch(model, OneBatchLoader(fake_batch), optimizer,
                    nn.CrossEntropyLoss(), TrainConfig())
    assert model.training, "model.train() was never called"


def test_train_one_epoch_does_not_accumulate_gradients_across_batches(model, fake_batch):
    """Forgetting optimizer.zero_grad() makes every step use stale gradients.

    Ten repeats of one batch should not produce a gradient ten times the size
    of one batch's.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)  # no weight change
    train_one_epoch(model, OneBatchLoader(fake_batch, repeats=10), optimizer,
                    nn.CrossEntropyLoss(), TrainConfig(grad_clip=None))
    after_many = model.head.weight.grad.norm().item()

    model.zero_grad(set_to_none=True)
    loss = nn.functional.cross_entropy(model(fake_batch), fake_batch["y"])
    loss.backward()
    after_one = model.head.weight.grad.norm().item()

    assert after_many == pytest.approx(after_one, rel=0.2), (
        "gradients are piling up across batches; call optimizer.zero_grad()"
    )


def test_the_model_can_overfit_one_batch(model, fake_batch):
    """The standard sanity check: memorising 8 examples must be possible.

    If this fails, the loop or the model is broken -- not under-tuned. Look at
    the order of zero_grad / backward / step first.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()

    first = train_one_epoch(model, OneBatchLoader(fake_batch), optimizer, criterion, TrainConfig())
    for _ in range(150):
        last = train_one_epoch(model, OneBatchLoader(fake_batch), optimizer, criterion, TrainConfig())

    assert last < first, f"loss did not fall at all ({first:.3f} -> {last:.3f})"
    assert last < 0.1, f"could not memorise one batch; loss stalled at {last:.3f}"


# --- Task 9: predict_proba ----------------------------------------------------

def test_predict_proba_returns_one_row_of_probabilities_per_example(model, fake_batch):
    proba = predict_proba(model, OneBatchLoader(fake_batch))

    assert isinstance(proba, np.ndarray)
    assert proba.shape == (len(fake_batch["y"]), FAKE_N_CLASSES)
    assert np.isfinite(proba).all()


def test_predict_proba_rows_sum_to_one(model, fake_batch):
    """evaluation.evaluate rejects anything that does not; it wants a softmax."""
    proba = predict_proba(model, OneBatchLoader(fake_batch))
    assert (proba >= 0).all(), "negative values -- these are logits, not probabilities"
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_is_accepted_by_the_evaluation_harness(model, fake_batch):
    """The real contract: the numbers have to survive evaluate()."""
    from evaluation import evaluate

    classes = [f"C{i}" for i in range(FAKE_N_CLASSES)]
    proba = predict_proba(model, OneBatchLoader(fake_batch))
    metrics = evaluate([classes[i] for i in fake_batch["y"].tolist()], proba, classes)

    assert metrics["n"] == len(fake_batch["y"])
    assert np.isfinite(metrics["log_loss"])


def test_predict_proba_keeps_the_loader_order(model, fake_batch):
    """Predictions are scored against labels by position, so order is the contract."""
    halves = [
        {key: value[:4] for key, value in fake_batch.items()},
        {key: value[4:] for key, value in fake_batch.items()},
    ]

    class TwoBatches:
        def __iter__(self):
            return iter(halves)

    both = predict_proba(model, TwoBatches())
    first = predict_proba(model, OneBatchLoader(halves[0]))
    np.testing.assert_allclose(both[:4], first, atol=1e-6)


def test_predict_proba_does_not_train_the_model(model, fake_batch):
    """Inference must not leave gradients or dropout behind."""
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    predict_proba(model, OneBatchLoader(fake_batch))

    assert not model.training, "model.eval() was never called"
    for name, p in model.named_parameters():
        assert torch.equal(before[name], p.detach()), f"{name} changed during inference"
        assert p.grad is None or not p.grad.any(), f"{name} accumulated a gradient"
