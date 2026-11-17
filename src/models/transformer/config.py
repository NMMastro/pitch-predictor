"""Every knob in one place, so a run is described by two small objects.

``ModelConfig`` is the shape of the network and ``TrainConfig`` is how it is
trained. Both are plain dataclasses: ``asdict`` goes straight into the
scoreboard row, so a logged run can be reproduced from what was recorded.

The defaults are sized for the v1 scope -- one pitcher, roughly 23k training
pitches, on a laptop CPU. A model much bigger than this would memorise Max
Scherzer rather than learn anything about him.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Scherzer. DESIGN_DECISIONS.md "v1 scope: single pitcher".
SCHERZER = 453286


@dataclass
class ModelConfig:
    """The network's shape.

    Attributes
    ----------
    vocab_sizes:
        Embedding rows per categorical column, from
        ``sequence_features.vocab_sizes``. Index 0 is padding and 1 is unknown,
        so these are already larger than the number of real values. Left empty
        by default and filled in from the data.
    d_model:
        Width of the model. Every embedding and projection maps into this, and
        it must divide evenly by ``n_heads``.
    n_heads:
        Attention heads per layer. Each head gets ``d_model // n_heads``
        dimensions to work with.
    n_layers:
        Stacked encoder layers.
    d_ff:
        Width of the feed-forward block inside each layer. Conventionally 2-4x
        ``d_model``.
    dropout:
        Applied inside the encoder layers and to the pooled output.
    window:
        Previous pitches per sequence. Must match the arrays the pipeline built,
        because the position embedding is sized from it.
    n_classes:
        Output classes: all ten of ``schema.PITCH_CLASSES``. The head stays
        ten-wide even for one pitcher (Open Decision 2); the scoring step
        restricts the label set instead.
    n_seq_numeric, n_ctx_numeric:
        Continuous features per sequence step and on the current pitch. Filled
        in from the data.
    """

    vocab_sizes: dict[str, int] = field(default_factory=dict)
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    window: int = 64
    n_classes: int = 10
    n_seq_numeric: int = 3
    n_ctx_numeric: int = 26

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model {self.d_model} must divide evenly by n_heads {self.n_heads}"
            )

    @property
    def d_head(self) -> int:
        """Dimensions each attention head works in."""
        return self.d_model // self.n_heads


@dataclass
class TrainConfig:
    """How the model is trained.

    Attributes
    ----------
    batch_size:
        Rows per optimiser step.
    lr:
        Adam's learning rate.
    weight_decay:
        L2 penalty, via AdamW.
    epochs:
        Passes over the training split.
    seed:
        Seeds Python, NumPy and torch, so two runs of the same config match.
    grad_clip:
        Maximum gradient norm. ``None`` disables clipping.
    num_workers:
        DataLoader worker processes. Keep at 0: the arrays are already in
        memory, so workers would only add Windows process-spawn overhead.
    device:
        ``"cpu"``. The v1 model is small enough that a GPU buys little.
    """

    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 10
    seed: int = 0
    grad_clip: float | None = 1.0
    num_workers: int = 0
    device: str = "cpu"
