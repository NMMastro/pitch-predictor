# Transformer session

Everything around the model is written. You write the model, the training step
and the prediction step: five stubs across two files, each with tests that fail
until it is right.

Work in order. Each task's tests pass on their own, so you always know whether
you have broken something new or are still looking at an old failure.

```powershell
# the bar to beat, and the data is real
.venv\Scripts\python.exe -m pytest tests -q --ignore=tests/model   # should be green
.venv\Scripts\python.exe src\model\train.py --epochs 1 --no-log    # prints the baseline, then stops at Task 5
```

| Task | Where | Tests |
| --- | --- | --- |
| 5. `SequenceEmbedding` | `src/model/transformer.py` | `pytest tests/model -k sequence_embedding` |
| 6. `ContextToken` | `src/model/transformer.py` | `pytest tests/model -k context_token` |
| 7. `PitchTransformer` | `src/model/transformer.py` | `pytest tests/model/test_transformer.py` |
| 8. `train_one_epoch` | `src/model/train.py` | `pytest tests/model/test_train.py -k train_one_epoch` |
| 9. `predict_proba` | `src/model/train.py` | `pytest tests/model/test_train.py` |

At the end, `python src/model/train.py --epochs 1` runs end to end, writes a
checkpoint and appends a row to `results/scoreboard.jsonl`.

---

## What the model does

A pitch is described two ways at once.

Its **history** is the pitcher's previous `window` pitches (64 by default), one
token each. This is what attention runs over: which earlier pitches matter for
guessing the next one.

Its **context** is the state before the pitch is thrown — the count, the
base/out state, the batter, the inning. That becomes a single token prepended
to the sequence, the way BERT prepends `[CLS]`. The encoder's output at that
first position is the model's summary of "what is about to be thrown", and a
linear head turns it into one logit per pitch class.

```
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
```

Prepending the context rather than bolting it onto the head means attention can
use it: "two strikes, left-handed batter" changes which previous pitches
matter. It also guarantees every row has at least one position attention is
allowed to look at, which is what stops a pitch with no history from producing
NaN.

### What a batch looks like

`src/model/dataset.py` hands the model a dict. `B` is the batch size (128), `W`
the window (64), `D` is `d_model` (64).

| Key | Shape | dtype | What it is |
| --- | --- | --- | --- |
| `seq_cat` | `(B, W, 3)` | int64 | previous pitches: type, zone, result |
| `seq_num` | `(B, W, 3)` | float32 | previous pitches: speed, horizontal and vertical location |
| `pad_mask` | `(B, W)` | bool | `True` where the slot holds no pitch |
| `ctx_cat` | `(B, 6)` | int64 | batter, pitcher, catcher, two fielding alignments, inning |
| `ctx_num` | `(B, 26)` | float32 | count, base/out, score, zone size, rest, ages, flags |
| `y` | `(B,)` | int64 | the label: an index into `schema.PITCH_CLASSES` |

Three conventions the pipeline guarantees, which the model relies on:

- **Index 0 means padding, index 1 means unknown.** Every categorical is stored
  as its real code plus 2. Build embeddings with `padding_idx=0`.
- **`pad_mask` is `True` for "ignore this".** That matches PyTorch's
  `src_key_padding_mask` and is the opposite of the "True means valid"
  convention you will see elsewhere. Only the start of a pitcher's career is
  padded, because the window never resets at games or seasons.
- **Nothing is NaN** and no categorical is negative. If the loss goes to NaN,
  the bug is in the model, not the data.

---

## A PyTorch primer

Just the parts these five stubs need.

### Tensors and shapes

A tensor is an n-dimensional array that remembers how it was computed, so
gradients can flow back through it. Almost every bug in new PyTorch code is a
shape bug, so print shapes freely:

```python
print(seq_cat.shape)   # torch.Size([128, 64, 3])
```

Shape operations you will use:

```python
x.unsqueeze(1)                 # (B, D)    -> (B, 1, D)   insert an axis
torch.cat([a, b], dim=1)       # (B,1,D) + (B,W,D) -> (B, 1+W, D)
x[:, 0]                        # (B, 1+W, D) -> (B, D)    take one position
x[..., i]                      # (B, W, 3) -> (B, W)      take one feature
torch.arange(W)                # (W,)      0, 1, ... W-1
```

Broadcasting fills in missing leading axes. Adding a `(W, D)` position tensor to
a `(B, W, D)` token tensor works as written and adds the same position vector to
every row of the batch, which is exactly what you want.

### `nn.Module`

A layer or a model. Two things to write: `__init__`, which creates the
sub-layers, and `forward`, which uses them. Anything assigned to `self` in
`__init__` is registered automatically, so its parameters get trained and saved.

```python
class Thing(nn.Module):
    def __init__(self, config):
        super().__init__()           # never skip this
        self.project = nn.Linear(10, config.d_model)

    def forward(self, x):
        return self.project(x)

thing = Thing(config)
out = thing(x)        # calls forward; do not call thing.forward(x) directly
```

To hold several layers, use a container, not a plain list or dict — a plain one
hides them from PyTorch and they will never be trained:

```python
self.tables = nn.ModuleDict({name: nn.Embedding(...) for name in columns})
layer = nn.TransformerEncoderLayer(...)
self.encoder = nn.TransformerEncoder(layer, num_layers=2)   # clones the layer
```

### `nn.Embedding`

A lookup table: one learned vector per integer index. It turns "slider" into
something the model can do arithmetic on.

```python
table = nn.Embedding(num_embeddings=12, embedding_dim=64, padding_idx=0)
table(torch.tensor([[2, 5], [3, 0]]))   # (2, 2) -> (2, 2, 64)
```

`num_embeddings` is how many distinct indices exist, which is what
`config.vocab_sizes[name]` gives you. An index at or above it raises
`IndexError: index out of range in self` — usually a sign the vocabulary came
from somewhere other than the data.

`padding_idx=0` pins row 0 to zeros and leaves it there: no gradient ever
updates it. So a padded slot contributes nothing, which is what you want on top
of masking it.

### `nn.Linear`

`y = xW + b`, applied to the last axis only. A `(B, W, 3)` input through
`nn.Linear(3, 64)` gives `(B, W, 64)` — the batch and position axes are left
alone. That is why projecting the per-pitch numbers needs no reshaping.

### `nn.TransformerEncoder` and the mask

```python
layer = nn.TransformerEncoderLayer(
    d_model=64, nhead=4, dim_feedforward=128, dropout=0.1,
    batch_first=True,   # (batch, position, features) -- without this it is (position, batch, features)
    norm_first=True,    # pre-norm; trains more steadily at this depth
)
encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
out = encoder(tokens, src_key_padding_mask=mask)   # (B, L, D) -> (B, L, D)
```

Set `batch_first=True`. Without it PyTorch expects the position axis first, and
your batch axis gets treated as positions: attention then mixes different
pitches together, the model looks unreasonably good, and
`test_each_row_is_predicted_from_its_own_data` is what catches it.

`src_key_padding_mask` is `(B, L)` and `True` means "ignore". It must be exactly
as long as the token sequence — so after you prepend the context token, the
mask needs a column prepended too.

Attention is **order-blind**. It sees a set of tokens, not a sequence. The order
is the entire signal in pitch sequencing, so you have to add position
information yourself; that is what `self.position` is for.

### Loss, and train vs eval

`nn.CrossEntropyLoss` takes **raw logits** and integer labels, and applies the
log-softmax internally. Applying a softmax yourself before it squashes the
gradients and the model barely learns. Keep the head's output raw; the softmax
belongs in `predict_proba`, once.

```python
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, batch["y"])     # (B, 10) and (B,)
```

`model.train()` and `model.eval()` switch dropout on and off. They do not train
anything — the names are misleading. `@torch.no_grad()` stops PyTorch from
building the graph, which makes inference faster and uses far less memory.

The training step, in the order that matters:

```python
optimizer.zero_grad()    # gradients accumulate by default; clear last step's
loss.backward()          # compute this step's
optimizer.step()         # apply them
```

Get the order wrong and the code still runs. It just learns nothing, or learns
from stale gradients. That is why there is a test for each of the three.

---

## Hints

Read a task's tests first, then come here. The hints go from a nudge to nearly
the answer, so stop as soon as you have enough.

### Task 5 — `SequenceEmbedding`

> Turn `(B, W, 3)` codes and `(B, W, 3)` numbers into `(B, W, D)` tokens.

1. Three sub-layers: a table per categorical column, one `nn.Linear` for the
   numbers, one `nn.Embedding` for position. The forward pass adds all three.
2. The categorical tables go in an `nn.ModuleDict` keyed by column name,
   because the provided `sum_embeddings` helper looks them up by name. Iterate
   `SEQ_CATEGORICAL` and size each table `config.vocab_sizes[name]` by
   `config.d_model`.
3. Every table must be `d_model` wide. That is what makes summing them legal,
   and summing is what keeps the width fixed however many categoricals there
   are.
4. The position table has one row per slot: `nn.Embedding(config.window,
   config.d_model)`. It is learned, not the sinusoid from the original paper —
   learned is simpler and fine for a fixed-length window.
5. In `forward`, build the indices with `torch.arange(seq_cat.shape[1],
   device=seq_cat.device)`. Use the tensor's own window length rather than
   `config.window` and the layer keeps working if you change the window.
   Passing `device` matters the moment anything runs on a GPU.
6. `self.position(positions)` is `(W, D)` and your tokens are `(B, W, D)`.
   Adding them broadcasts correctly as-is; no reshaping needed.
7. Optional, and worth it: a final `nn.LayerNorm(config.d_model)` over the sum.
   Three summed signals of different scales otherwise hand the encoder
   something lopsided.

### Task 6 — `ContextToken`

> Turn `(B, 6)` codes and `(B, 26)` numbers into one `(B, 1, D)` token.

1. The same recipe as Task 5 with the position dropped: there is only one
   token, so it has no position to encode.
2. `CTX_CATEGORICAL` instead of `SEQ_CATEGORICAL`, and
   `config.n_ctx_numeric` instead of `config.n_seq_numeric`.
3. `sum_embeddings` works unchanged here. It indexes the last axis, which is
   the column axis in both `(B, W, 3)` and `(B, 6)`.
4. The sum is `(B, D)` and the test wants `(B, 1, D)`. `unsqueeze(1)` is the
   last step — the `1` is what lets `torch.cat` put it in front of the window.

### Task 7 — `PitchTransformer`

> Concatenate, mask, encode, and read position 0.

1. `__init__` needs two more things: the encoder and the head. The embedding
   layers are already wired up for you.
2. `nn.TransformerEncoderLayer` first, then `nn.TransformerEncoder(layer,
   num_layers=config.n_layers)` to stack copies of it. Pass
   `batch_first=True`, and `enable_nested_tensor=False` on the encoder to
   silence a warning about its padding fast path.
3. The head is one `nn.Linear(config.d_model, config.n_classes)`. Ten classes
   even for one pitcher — the scoring step in `train.py` narrows the label set
   instead, so the model stays usable for other pitchers.
4. In `forward`, context token first: `torch.cat([context, sequence], dim=1)`
   gives `(B, 1 + W, D)`. The order matters, because step 6 reads position 0.
5. The mask has to grow to match. `batch["pad_mask"]` is `(B, W)` and the
   tokens are now `1 + W` long. The context token is always real, so prepend a
   column of `False`:
   `torch.cat([torch.zeros_like(pad_mask[:, :1]), pad_mask], dim=1)`.
   Using `zeros_like` keeps the dtype and device right for free.
6. If `test_transformer_ignores_whatever_is_in_the_padded_slots` fails, the
   mask is the suspect: missing, inverted, or one column short.
7. Read the summary with `encoded[:, 0]`, which is `(B, D)`, then apply the
   head. A `nn.Dropout` in between is a reasonable touch.
8. Return logits. No softmax, no activation.

### Task 8 — `train_one_epoch`

> Loop, step, and report the mean loss.

1. `model.train()` once, before the loop, not inside it.
2. Five calls per batch, in this order: `optimizer.zero_grad()`, `model(batch)`,
   `criterion(logits, batch["y"])`, `loss.backward()`, `optimizer.step()`.
3. `nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)` goes
   between `backward` and `step` — it rescales the gradients that `backward`
   just computed, so it has to run after one and before the other. Skip it when
   `config.grad_clip is None`.
4. Accumulate `loss.item() * len(batch["y"])` and the row count, then divide.
   Weighting by batch size matters because the last batch is short.
5. Use `.item()`. Accumulating the loss *tensor* keeps its whole computation
   graph alive, and memory grows every batch until the epoch dies.
6. If `test_the_model_can_overfit_one_batch` fails, the order in hint 2 is the
   first thing to re-read. The loss falling slowly means a tuning problem; the
   loss not falling at all means a wiring problem.

### Task 9 — `predict_proba`

> Logits to probabilities, in loader order, as NumPy.

1. `model.eval()` first, so dropout is off. The `@torch.no_grad()` decorator is
   already on the function and handles the gradient side.
2. `torch.softmax(logits, dim=-1)`. `dim=-1` is the class axis; softmaxing the
   wrong axis normalises across the batch instead and the rows will not sum to
   1.
3. Cast to float64 before the softmax (`model(batch).double()`). In float32 the
   rows sum to 1 only to about 1e-7, and `evaluate()` checks.
4. Collect the batches in a list and `torch.cat` them, then `.numpy()`. Order is
   the contract: predictions are scored against labels by position, which is
   why the val and test loaders do not shuffle.

---

## Finishing up

```powershell
.venv\Scripts\python.exe -m pytest tests -q          # all green
.venv\Scripts\python.exe src\model\train.py --epochs 1
```

The last command prints the baseline, trains an epoch, saves the best
checkpoint to `results/checkpoints/`, scores the test split, and appends a row
to `results/scoreboard.jsonl`.

### Reading the result

The number to watch is **validation log loss against the baseline**. The
baseline predicts Scherzer's training pitch mix for every pitch and ignores
every feature, which already scores about 1.43. Accuracy on its own is
misleading: he throws a fastball roughly half the time, so always guessing
fastball gets about 44% without knowing anything.

Two things to know before you read too much into a number:

**The validation split is small.** Scherzer pitched only about nine starts in
2024, so validation is roughly 670 pitches. Differences of a few hundredths of
a log loss are noise. Confirm anything that looks like an improvement by
re-running with a different `--seed`.

**One epoch is about 45 seconds** at window 64 on CPU, so a 20-epoch run is
around 15 minutes.

### Things worth trying next

- `--epochs 20` and watch where validation stops improving.
- `--window 400` against `--window 64`. Longer history is the experiment the
  window was made configurable for; attention cost grows with the square of the
  window, so expect it to be slower.
- `--d-model 128 --n-layers 4`, to see whether 23k pitches support a bigger
  model. They may well not.
- `--dropout 0.3`, if train loss keeps falling while validation loss rises.

`evaluation.load_scoreboard("results/scoreboard.jsonl")` reads every run back
as a DataFrame for comparison.
