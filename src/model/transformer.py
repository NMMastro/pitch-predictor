"""The pitch-type transformer: previous pitches + current state -> 10 logits.

This file implements Tasks 5-7 from ``docs/TRANSFORMER_SESSION.md``: embedding
the pitch history, building the context token, and producing pitch logits with
a transformer encoder.

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
        # Register one embedding table per categorical column. Each table has
        # one row per code and produces a d_model-wide vector. The column names
        # are the keys because sum_embeddings looks up each table by name.
        self.categorical = nn.ModuleDict(
            {
                name: nn.Embedding(
                    num_embeddings=config.vocab_sizes[name],  # One row per code.
                    embedding_dim=config.d_model,  # Same vector width for every table.
                    padding_idx=0,  # Padding codes return a zero vector.
                )
                for name in SEQ_CATEGORICAL
            }
        )

        # Project the continuous pitch features to the same vector width.
        self.numeric = nn.Linear(
            in_features=config.n_seq_numeric,
            out_features=config.d_model,
        )

        # Learn one d_model-wide vector for each position in the history window.
        self.position = nn.Embedding(
            num_embeddings=config.window,
            embedding_dim=config.d_model,
        )

    def forward(self, seq_cat: torch.Tensor, seq_num: torch.Tensor) -> torch.Tensor:
        # Look up and sum the categorical vectors for each pitch.
        tokens = sum_embeddings(self.categorical, SEQ_CATEGORICAL, seq_cat)

        # Transform the numeric features to d_model values and add them.
        tokens = tokens + self.numeric(seq_num)

        # Create slot indices from 0 to W-1 on the same device as the input.
        positions = torch.arange(seq_cat.shape[1], device=seq_cat.device)

        # Add each slot's learned vector to that slot in every batch row.
        tokens = tokens + self.position(positions)

        # Return one vector per pitch, with shape (B, W, d_model).
        return tokens


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
        # Build one embedding table for each categorical context feature.
        self.categorical = nn.ModuleDict(
            {
                name: nn.Embedding(
                    num_embeddings=config.vocab_sizes[name],  # One row per code.
                    embedding_dim=config.d_model,  # Same vector width for every table.
                    padding_idx=0,  # Padding codes return a zero vector.
                )
                for name in CTX_CATEGORICAL
            }
        )

        # Project the numeric context features to d_model values.
        self.numeric = nn.Linear(
            in_features=config.n_ctx_numeric,
            out_features=config.d_model,
        )

    def forward(self, ctx_cat: torch.Tensor, ctx_num: torch.Tensor) -> torch.Tensor:
        # Combine the categorical and numeric context into one vector.
        tokens = sum_embeddings(self.categorical, CTX_CATEGORICAL, ctx_cat)
        tokens = tokens + self.numeric(ctx_num)

        # Add a token axis so the result has shape (B, 1, d_model).
        return tokens.unsqueeze(1)


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
        # Define one layer that combines attention and a feedforward network.
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True,  # Inputs use (batch, position, features).
            norm_first=True,  # Normalize before each attention/network block.
        )

        # Stack independently trained copies of the layer.
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )

        # Convert the encoder's summary into one raw score per pitch class.
        self.head = nn.Linear(
            in_features=config.d_model,
            out_features=config.n_classes,
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # Build the current situation's single token: (B, 1, d_model).
        context = self.context(batch["ctx_cat"], batch["ctx_num"])

        # Build one token per previous pitch: (B, W, d_model).
        sequence = self.sequence(batch["seq_cat"], batch["seq_num"])

        # Join along the token axis, with context first: (B, 1 + W, d_model).
        tokens = torch.cat([context, sequence], dim=1)

        # Read the history mask: True marks a padded pitch slot.
        pad_mask = batch["pad_mask"]

        # Create one False per example; the context token is always valid.
        context_mask = pad_mask.new_zeros((pad_mask.shape[0], 1))

        # Put context first so the mask matches the token sequence.
        mask = torch.cat([context_mask, pad_mask], dim=1)

        # Run attention while preventing it from reading padded pitch slots.
        encoded = self.encoder(tokens, src_key_padding_mask=mask)

        # Select the updated context token from every example: (B, d_model).
        summary = encoded[:, 0]

        # Convert each summary into pitch-class scores: (B, n_classes).
        return self.head(summary)


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
