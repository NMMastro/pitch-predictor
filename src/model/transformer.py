"""The pitch-type transformer: previous pitches + current state -> 10 logits.

WRITE THIS FILE. The three ``NotImplementedError`` stubs below are Tasks 5-7 of
``docs/TRANSFORMER_SESSION.md``, which has progressive hints for each. Run
``pytest tests/model/test_transformer.py`` as you go.

Shape of the thing, in words. A pitch is described two ways at once. Its
*history* is the pitcher's previous ``window`` pitches, one token each, which is
what attention runs over. Its *context* is the state before this pitch -- count,
base/out, batter, inning -- which is a single token prepended to the sequence,
the way BERT prepends ``[CLS]``. The encoder's output at that first position is
the model's summary of "what is about to be thrown", and a linear head turns it
into one logit per pitch class.

Prepending the context rather than concatenating it at the head means attention
can use it: "two strikes, left-handed batter" changes which previous pitches
matter. It also guarantees every row has at least one unmasked position, so a
pitch with no history at all cannot produce NaN.

    seq_cat  (B, W, 3) ---> embeddings, summed ---.
    seq_num  (B, W, 3) ---> Linear --------------->+--> + position --> (B, W, D)
                                                                          |
    ctx_cat  (B, 6) ------> embeddings, summed ---.                       |
    ctx_num  (B, C) ------> Linear --------------->+--> (B, 1, D) ---.    |
                                                                     v    v
                                      concatenate --> (B, 1 + W, D) -------'
                                                            |
                                      nn.TransformerEncoder, src_key_padding_mask
                                                            |
                                           position 0 --> Linear --> (B, 10)

Conventions worth remembering, because they are the usual source of silent bugs:

- Index 0 of every categorical is padding and index 1 is unknown. Build the
  embeddings with ``padding_idx=0``, which pins that row to zeros and keeps it
  there (no gradient).
- ``pad_mask`` is True where a position should be **ignored**. That is
  PyTorch's ``src_key_padding_mask`` convention, and it is the opposite of the
  "True means valid" convention used elsewhere.
- Keep everything ``batch_first``: tensors are (batch, position, features).
- The head returns raw logits, never probabilities. ``CrossEntropyLoss`` wants
  logits and ``evaluation.evaluate`` wants probabilities, so the softmax lives
  in ``train.predict_proba``, once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.config import ModelConfig  # noqa: E402
from model.dataset import CTX_CATEGORICAL, SEQ_CATEGORICAL  # noqa: E402


def sum_embeddings(
    tables: nn.ModuleDict, columns: list[str], codes: torch.Tensor
) -> torch.Tensor:
    """Look each column's codes up in its own table and add the results.

    Given to you -- it is index bookkeeping, not a design decision. ``codes`` is
    ``(..., len(columns))`` and holds column i's codes at ``codes[..., i]``;
    the result is ``(..., d_model)``.

    Summing rather than concatenating keeps the width at ``d_model`` however
    many categoricals there are, and lets each one learn its own offset. It
    works only because every table is ``d_model`` wide, so build them that way.
    """
    return sum(tables[name](codes[..., i]) for i, name in enumerate(columns))


class SequenceEmbedding(nn.Module):
    """The previous ``window`` pitches as one token each.

    Three things are added together per pitch: its categorical features (type,
    zone, result), a projection of its continuous features (speed, location),
    and a position embedding. Position has to be added explicitly -- attention
    has no idea what order its inputs came in, and the order is the whole point
    of a sequence model.

    forward(seq_cat, seq_num) -> (B, W, d_model)
        seq_cat  (B, W, 3) int64    codes, 0 = padding
        seq_num  (B, W, 3) float32  standardised, 0 where padded

    The config fields you need: ``vocab_sizes``, ``d_model``,
    ``n_seq_numeric``, ``window``.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # TODO 1: self.categorical -- an nn.ModuleDict with one nn.Embedding per
        #         column in SEQ_CATEGORICAL, each config.vocab_sizes[name] rows
        #         by config.d_model wide, with padding_idx=0. The keys must be
        #         the column names, because sum_embeddings looks them up by name.
        # TODO 2: self.numeric -- an nn.Linear projecting config.n_seq_numeric
        #         continuous features up to config.d_model.
        # TODO 3: self.position -- an nn.Embedding holding one learned vector
        #         per slot in the window: config.window rows by config.d_model.
        raise NotImplementedError("Task 5: build the sequence embedding layers")

    def forward(self, seq_cat: torch.Tensor, seq_num: torch.Tensor) -> torch.Tensor:
        # TODO 4: embed the categorical codes with sum_embeddings(...).
        # TODO 5: add the projected numeric features.
        # TODO 6: add the position embedding. Positions are 0..W-1; build them
        #         with torch.arange and keep them on seq_cat.device. Mind the
        #         shapes: positions give (W, d_model) and the tokens are
        #         (B, W, d_model), which broadcasts correctly as-is.
        # TODO 7: return (B, W, d_model).
        raise NotImplementedError("Task 5: combine codes, numbers and position")


class ContextToken(nn.Module):
    """The state before the pitch, as a single token.

    Same recipe as one sequence step, minus the position: the categoricals
    (batter, pitcher, catcher, the two fielding alignments, inning) are
    embedded and summed, and the continuous features are projected into the
    same width.

    forward(ctx_cat, ctx_num) -> (B, 1, d_model)
        ctx_cat  (B, 6) int64
        ctx_num  (B, C) float32

    Note the ``1``: the result is one token, ready to be concatenated onto the
    front of the sequence. ``unsqueeze`` is how you get it.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # TODO 1: self.categorical -- as in SequenceEmbedding, but over
        #         CTX_CATEGORICAL.
        # TODO 2: self.numeric -- nn.Linear from config.n_ctx_numeric to
        #         config.d_model.
        raise NotImplementedError("Task 6: build the context layers")

    def forward(self, ctx_cat: torch.Tensor, ctx_num: torch.Tensor) -> torch.Tensor:
        # TODO 3: sum the embedded codes and the projected numbers, then return
        #         (B, 1, d_model).
        raise NotImplementedError("Task 6: build the context token")


class PitchTransformer(nn.Module):
    """Context token + previous pitches -> one logit per pitch class.

    forward(batch) -> (B, n_classes)
        batch is the dict ``dataset.py`` produces; ``y`` is ignored here.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.sequence = SequenceEmbedding(config)
        self.context = ContextToken(config)
        # TODO 1: self.encoder -- an nn.TransformerEncoder of config.n_layers
        #         nn.TransformerEncoderLayer(s). The layer takes d_model,
        #         nhead, dim_feedforward, dropout, and batch_first=True.
        #         Two options worth setting: norm_first=True (pre-norm, which
        #         trains more steadily at this depth) and, on the encoder,
        #         enable_nested_tensor=False (its padding fast path is not
        #         worth the warning at this size).
        # TODO 2: self.head -- an nn.Linear from config.d_model to
        #         config.n_classes. Logits, so no activation after it.
        raise NotImplementedError("Task 7: build the encoder and the head")

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # TODO 3: build the token sequence: the context token, then the window.
        #         torch.cat along dim=1 gives (B, 1 + W, d_model). Context
        #         first, because the head reads position 0.
        # TODO 4: extend the mask to match. batch["pad_mask"] is (B, W) and the
        #         sequence is now 1 + W long. The context token is always real,
        #         so prepend a column of False. Getting this wrong by one
        #         column is the single most common bug here, and
        #         test_transformer_ignores_whatever_is_in_the_padded_slots is
        #         what catches it.
        # TODO 5: run self.encoder(tokens, src_key_padding_mask=mask).
        # TODO 6: take position 0 of the output and send it through self.head.
        #         Return (B, n_classes).
        raise NotImplementedError("Task 7: run the encoder over the tokens")


# --- given --------------------------------------------------------------------

def build_model(splits, config: ModelConfig | None = None) -> PitchTransformer:
    """A model whose shapes match the data in ``splits``.

    The vocabulary sizes and feature counts have to come from the arrays, not
    from a default, or the embeddings will be the wrong size for the codes.
    """
    config = config or ModelConfig()
    config = ModelConfig(**{
        **config.__dict__,
        "vocab_sizes": splits.vocab_sizes,
        "window": splits.window,
        "n_seq_numeric": splits.n_seq_numeric,
        "n_ctx_numeric": splits.n_ctx_numeric,
        "n_classes": splits.n_classes,
    })
    return PitchTransformer(config)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
