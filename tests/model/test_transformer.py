"""Tests for the model you are writing. Work through them top to bottom.

These fail with NotImplementedError until the stubs in
``src/model/transformer.py`` are filled in. Each test names the property it is
checking; read the name before reading the assertion, because the name is the
requirement and the assertion is only one way to check it.

Nothing here touches the parquet files -- the ``fake_batch`` fixture in
``tests/conftest.py`` makes tensors of the right shape, so the tests are fast
and fail for model reasons only.

Run one group at a time:
    pytest tests/model/test_transformer.py -k sequence_embedding
    pytest tests/model/test_transformer.py -k context_token
    pytest tests/model/test_transformer.py -k transformer
"""
from __future__ import annotations

import pytest
import torch

from conftest import FAKE_BATCH_SIZE as B
from conftest import FAKE_N_CLASSES, FAKE_WINDOW as W
from model.config import ModelConfig
from model.dataset import CTX_CATEGORICAL, CTX_NUMERIC, SEQ_CATEGORICAL
from model.transformer import ContextToken, PitchTransformer, SequenceEmbedding

D_MODEL = 32


@pytest.fixture
def config(vocab_sizes) -> ModelConfig:
    return ModelConfig(
        vocab_sizes=vocab_sizes,
        d_model=D_MODEL,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        dropout=0.1,
        window=W,
        n_classes=FAKE_N_CLASSES,
        n_seq_numeric=len(SEQ_CATEGORICAL),
        n_ctx_numeric=len(CTX_NUMERIC),
    )


@pytest.fixture
def model(config) -> PitchTransformer:
    torch.manual_seed(0)
    return PitchTransformer(config)


# --- Task 5: SequenceEmbedding ------------------------------------------------

def test_sequence_embedding_maps_a_window_to_one_token_per_pitch(config, fake_batch):
    """Each of the W previous pitches becomes one d_model-wide vector."""
    layer = SequenceEmbedding(config)
    out = layer(fake_batch["seq_cat"], fake_batch["seq_num"])

    assert out.shape == (B, W, config.d_model)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_sequence_embedding_uses_every_categorical_column(config, fake_batch):
    """All three per-pitch codes must be read, not just the pitch type."""
    layer = SequenceEmbedding(config).eval()
    with torch.no_grad():
        base = layer(fake_batch["seq_cat"], fake_batch["seq_num"])

    for i, name in enumerate(SEQ_CATEGORICAL):
        changed = fake_batch["seq_cat"].clone()
        # Move every real code in this column to a different non-padding index.
        # 1 and 2 are the only two that exist in every vocabulary, however
        # small -- `pitcher` has just three rows for a single-pitcher run.
        real = ~fake_batch["pad_mask"]
        column = changed[..., i]
        column[real] = torch.where(column[real] == 1, 2, 1)
        with torch.no_grad():
            out = layer(changed, fake_batch["seq_num"])
        assert not torch.allclose(out, base), f"{name} is being ignored"


def test_sequence_embedding_uses_the_numeric_features(config, fake_batch):
    layer = SequenceEmbedding(config).eval()
    with torch.no_grad():
        base = layer(fake_batch["seq_cat"], fake_batch["seq_num"])
        out = layer(fake_batch["seq_cat"], fake_batch["seq_num"] + 1.0)
    assert not torch.allclose(out, base), "seq_num is being ignored"


def test_sequence_embedding_knows_the_order_of_the_window(config):
    """Attention is order-blind, so position has to be added here.

    The same pitch in two different slots must not get the same vector, or the
    model cannot tell "fastball five pitches ago" from "fastball just now".
    """
    layer = SequenceEmbedding(config).eval()
    seq_cat = torch.full((1, W, len(SEQ_CATEGORICAL)), 2, dtype=torch.int64)
    seq_num = torch.zeros(1, W, config.n_seq_numeric)

    with torch.no_grad():
        out = layer(seq_cat, seq_num)[0]
    assert not torch.allclose(out[0], out[-1]), "no position information"


def test_sequence_embedding_pads_with_zeros_through_padding_idx(config):
    """Index 0 must be a frozen zero row: nn.Embedding(..., padding_idx=0)."""
    layer = SequenceEmbedding(config)
    for name, table in layer.categorical.items():
        assert table.padding_idx == 0, f"{name} needs padding_idx=0"
        assert torch.equal(table.weight[0], torch.zeros(config.d_model)), name


# --- Task 6: ContextToken -----------------------------------------------------

def test_context_token_is_a_single_token(config, fake_batch):
    """One token, so it can be concatenated onto the sequence: (B, 1, D)."""
    layer = ContextToken(config)
    out = layer(fake_batch["ctx_cat"], fake_batch["ctx_num"])

    assert out.shape == (B, 1, config.d_model)
    assert torch.isfinite(out).all()


def test_context_token_uses_every_categorical_column(config, fake_batch):
    layer = ContextToken(config).eval()
    with torch.no_grad():
        base = layer(fake_batch["ctx_cat"], fake_batch["ctx_num"])

    for i, name in enumerate(CTX_CATEGORICAL):
        changed = fake_batch["ctx_cat"].clone()
        # 1 and 2 are valid in every vocabulary; see the sequence version above.
        changed[:, i] = torch.where(changed[:, i] == 1, 2, 1)
        with torch.no_grad():
            out = layer(changed, fake_batch["ctx_num"])
        assert not torch.allclose(out, base), f"{name} is being ignored"


def test_context_token_uses_the_numeric_features(config, fake_batch):
    layer = ContextToken(config).eval()
    with torch.no_grad():
        base = layer(fake_batch["ctx_cat"], fake_batch["ctx_num"])
        out = layer(fake_batch["ctx_cat"], fake_batch["ctx_num"] + 1.0)
    assert not torch.allclose(out, base), "ctx_num is being ignored"


# --- Task 7: PitchTransformer.forward ----------------------------------------

def test_transformer_returns_one_logit_per_class(model, fake_batch):
    out = model(fake_batch)
    assert out.shape == (B, FAKE_N_CLASSES)
    assert torch.isfinite(out).all()


def test_transformer_returns_logits_not_probabilities(model, fake_batch):
    """CrossEntropyLoss wants logits; the softmax belongs in predict_proba."""
    with torch.no_grad():
        out = model(fake_batch)
    sums = out.sum(dim=-1)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3), (
        "rows sum to 1 -- a softmax has been applied inside forward"
    )


def test_transformer_ignores_whatever_is_in_the_padded_slots(model, fake_batch):
    """The point of src_key_padding_mask.

    Padded slots hold no pitch. Writing junk into them must not move the
    output at all. If this fails, the mask is missing, inverted, or not being
    extended to cover the prepended context token.
    """
    model.eval()
    with torch.no_grad():
        base = model(fake_batch)

    junk = {key: value.clone() for key, value in fake_batch.items()}
    pad = junk["pad_mask"]
    assert pad.any(), "the fixture should include padded slots"
    junk["seq_cat"][pad] = 5           # a valid but wrong code
    junk["seq_num"][pad] = 99.0

    with torch.no_grad():
        out = model(junk)
    assert torch.allclose(base, out, atol=1e-5), "padded positions are leaking in"


def test_transformer_handles_a_row_with_no_history(model, fake_batch):
    """A pitcher's first pitch has an entirely padded window.

    Masking every position of a row makes attention average over nothing, which
    is NaN. Prepending the context token is what prevents it, because that
    position is always real.
    """
    model.eval()
    batch = {key: value.clone() for key, value in fake_batch.items()}
    batch["pad_mask"][:] = True
    batch["seq_cat"][:] = 0
    batch["seq_num"][:] = 0.0

    with torch.no_grad():
        out = model(batch)
    assert torch.isfinite(out).all(), "an all-padding row produced NaN or inf"
    assert out.shape == (B, FAKE_N_CLASSES)


def test_transformer_reads_the_context(model, fake_batch):
    """The pre-pitch state must reach the output, not just the history."""
    model.eval()
    with torch.no_grad():
        base = model(fake_batch)
        changed = {**fake_batch, "ctx_num": fake_batch["ctx_num"] + 1.0}
        out = model(changed)
    assert not torch.allclose(base, out), "the context token is not being used"


def test_transformer_reads_the_history(model, fake_batch):
    model.eval()
    with torch.no_grad():
        base = model(fake_batch)
        changed = {**fake_batch, "seq_num": fake_batch["seq_num"] + 1.0}
        out = model(changed)
    assert not torch.allclose(base, out), "the sequence is not being used"


def test_each_row_is_predicted_from_its_own_data(model, fake_batch):
    """No leakage across the batch: row 0 must not depend on row 1.

    Attention must run across *positions*, never across the batch. Transposing
    a tensor by mistake is the usual way this breaks, and it looks like a very
    good model until the batch size changes.
    """
    model.eval()
    with torch.no_grad():
        base = model(fake_batch)

    changed = {key: value.clone() for key, value in fake_batch.items()}
    for key in ("seq_cat", "seq_num", "ctx_cat", "ctx_num", "pad_mask"):
        changed[key][1:] = changed[key][:1]  # overwrite every row but the first

    with torch.no_grad():
        out = model(changed)
    assert torch.allclose(base[0], out[0], atol=1e-5), "row 0 depends on other rows"


def test_gradients_reach_every_parameter(model, fake_batch):
    """A parameter with no gradient is a part of the model doing nothing."""
    loss = torch.nn.functional.cross_entropy(model(fake_batch), fake_batch["y"])
    loss.backward()

    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"

    # Some rows of an embedding table are legitimately untouched -- the
    # padding row always is -- but a table with no gradient anywhere is wired
    # up wrong.
    dead = [name for name, p in model.named_parameters() if not p.grad.any()]
    assert not dead, f"gradient was all zeros for: {dead}"


def test_the_model_is_small_enough_for_one_pitcher(model):
    """~23k training pitches cannot support a large model."""
    from model.transformer import count_parameters

    assert count_parameters(model) < 2_000_000
