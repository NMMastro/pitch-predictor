# Transformer progress and initial findings

**Project:** Next-pitch type prediction  
**Report date:** November 11, 2026  
**Scope:** First single-pitcher experiments using Max Scherzer

## Overview

We have built an end-to-end transformer pipeline that predicts the type of the upcoming pitch using pre-pitch game context and a sequence of earlier pitches. The pipeline trains models, selects checkpoints using validation log loss, evaluates them, and saves results. A command-line demo loads a preserved checkpoint and displays predictions for held-out pitches.

The first experiments show a modest improvement over a baseline that always predicts the training pitch-mix probabilities. Results vary considerably between random seeds, and longer training shows clear signs of overfitting. These experiments establish a working prototype; they do not yet establish a reliable advantage over simpler models.

## 1. What we have implemented

### Data and feature pipeline

- Cleaned Statcast pitch data, with agreed pitch-class definitions and filtering rules.
- Chronological training, validation, and test splits.
- Per-pitcher history windows containing only earlier pitches.
- Encoded pitch types, zones, results, player identities, and other categorical features.
- Batter-relative horizontal locations and mirrored zone categories.
- Continuous-feature standardization fitted on training rows.
- Padding masks, unknown-category handling, missing-value filling, and missing-rest indicators.
- A minimum of 25 prior pitches before a row becomes eligible for training or evaluation. Those earlier pitches still contribute to history.

History continues across at-bats, games, seasons, and split boundaries. A validation or test pitch may use earlier observed pitches as history, including earlier pitches from the same evaluation season. This evaluates sequential prediction as pitches become available.

### Transformer model

Each historical pitch becomes a token. A separate context token represents the state before the upcoming pitch. The context token is placed before the history sequence, and a transformer encoder processes them together. The encoded context token feeds a classifier that produces pitch-type scores.

| Component | Initial configuration |
| --- | --- |
| Model width | 64 |
| Attention heads | 4 |
| Transformer encoder layers | 2 |
| Feed-forward width | 128 |
| Dropout in the overnight grid | 0.1 |
| Output head | 10 global pitch classes |
| History lengths in the overnight grid | 64, 128, 256 pitches |
| Position information | Learned positional embeddings |

The 10-class head preserves the global taxonomy. For these single-pitcher experiments, evaluation keeps the five classes present in training and renormalizes their probabilities: **FF, SL, CH, FC, CU**.

The overnight models contain 129,226 parameters with a 64-pitch history, 133,322 with 128 pitches, and 141,514 with 256 pitches. Longer windows increase the size of the learned position embedding table as well as computation.

### Training and prediction

- Cross-entropy training loss.
- AdamW optimizer, with gradient clipping at a maximum norm of 1.0.
- Checkpoint selection by the lowest validation log loss within each run.
- Reloading of the selected checkpoint before test evaluation.
- JSONL experiment scoreboard.
- Overnight experiment script that preserves separate logs and checkpoints for each window/seed combination.
- Demo script displaying the top three predicted pitch types and their probabilities.

The training loop runs the requested number of epochs and saves the best checkpoint. It does **not** currently stop automatically when validation performance stops improving.

Tests exist for feature construction, sequence ordering, scaling, padding, model inputs and outputs, gradient flow, weight updates, and prediction probabilities. This report summarizes saved experiment evidence; it does not claim a new test-suite run.

## 2. Inputs used by the model

### Current context

Six categorical fields describe the batter, pitcher, catcher, infield alignment, outfield alignment, and inning. Another 26 numeric or binary fields describe the count, outs, occupied bases, scores and score differences, handedness, strike-zone bounds, times through the order, prior plate appearances, rest, ages, and first-pitch/missing-rest flags.

### Historical pitches

Every earlier pitch contributes:

| Field | Meaning |
| --- | --- |
| `type_code` | Pitch type |
| `zone_code` | Zone adjusted for batter handedness |
| `result_code` | Grouped pitch result |
| `release_speed` | Pitch speed |
| `plate_x_rel` | Horizontal location adjusted for batter handedness |
| `plate_z` | Vertical location |

Measurements from the pitch being predicted are excluded from its inputs. The task is to forecast pitch selection before the pitch is thrown.

## 3. Dataset and evaluation setup

The first experiments use **Max Scherzer (MLBAM ID 453286)**.

| Split | Seasons | Eligible prediction rows |
| --- | --- | ---: |
| Training | 2016–2023 | 22,893 |
| Validation | 2024 | 669 |
| Test | 2025 | 1,610 |

There are 22,918 training-period rows before the minimum-history filter. Regular-season and postseason pitches are included; spring training is excluded.

Approximate training pitch mix: FF 49%, SL 20%, CH 14%, FC 8%, CU 9%.

**Primary metric: multiclass log loss.** Lower is better. This evaluates the probabilities assigned to actual outcomes and penalizes confident mistakes. We also report accuracy, top-2 accuracy, macro precision/recall/F1, and per-class F1.

The baseline assigns the same probability vector to every pitch, using pitch frequencies from the training split. Its highest-probability prediction is always FF.

| Validation baseline | Result |
| --- | ---: |
| Training pitch-mix log loss | 1.4301 |
| Accuracy from always choosing FF | 43.95% |

The validation baseline log loss must not be used as a test-set baseline: the test-set baseline must be calculated separately on test outcomes.

## 4. Experiments completed

The scoreboard contains **10 recorded runs**: three preliminary runs, six overnight grid runs, and one later training-configuration experiment.

### Six-run overnight grid

All six runs used 20 epochs, batch size 128, learning rate 0.001, weight decay 0.01, and dropout 0.1. Seeds change initialization and other seeded training randomness; comparing them helps reveal sensitivity to the training run.

All metrics below refer to the checkpoint selected by validation log loss, rather than the last training epoch.

| History | Seed | Selected epoch | Validation log loss | Validation accuracy | Validation top-2 | Test log loss | Test accuracy* | Test top-2* |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 7 | 1.4077 | 44.84% | 73.24% | 1.4311 | 41.4% | 70.5% |
| 64 | 2 | 5 | 1.4594 | 42.60% | 64.13% | 1.4356 | 44.0% | 70.7% |
| 128 | 1 | 1 | 1.4503 | 40.96% | 65.77% | 1.4386 | 41.6% | 68.2% |
| 128 | 2 | 1 | 1.4408 | 42.30% | 65.47% | 1.4086 | 46.6% | 68.4% |
| 256 | 1 | 6 | **1.3881** | 44.69% | 72.05% | 1.4288 | 43.6% | 71.6% |
| 256 | 2 | 1 | 1.4231 | 40.36% | 67.12% | 1.4501 | 42.0% | 66.0% |

*Test accuracy and top-2 accuracy are reported at the precision preserved in the text logs. Log loss and validation metrics come from the scoreboard.*

### Preliminary and later runs

| Run | Seed | Epoch budget | Selected epoch | Validation log loss | Validation accuracy | Test log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Preliminary 1 | 0 | 1 | 1 | 1.4685 | 43.20% | 1.4837 |
| Preliminary 2 | 0 | 20 | 5 | 1.4338 | 45.59% | 1.4789 |
| Preliminary 3 | 0 | 20 | 2 | 1.4238 | 42.15% | 1.6739 |
| Later configuration experiment | 3 | 30 | 3 | 1.4414 | 39.31% | 1.4223 |

The later experiment used a learning rate of 0.0003 and weight decay of 0.03. The current general 256-pitch checkpoint has dropout 0.3 and selected epoch 3, consistent with that experiment, but the scoreboard does not directly link the run to its checkpoint. The preliminary scoreboard rows also omit history length, so this report does not assign them a window based on inference. A 400-pitch checkpoint exists, but its full run configuration is not recorded in a uniquely linked scoreboard entry.

The later configuration did not improve validation log loss. Because several settings and the seed changed, it does not isolate the effect of any individual setting.

## 5. Findings from the first sessions

### A. The best validation result improves on the pitch-mix baseline, modestly

The best recorded validation log loss is **1.3881**, from the 256-pitch, seed-1 run at epoch 6. Compared with 1.4301 for the baseline, this is a reduction of about **0.042**, or **2.9%**.

Its validation accuracy is **44.69%**, compared with **43.95%** for always predicting FF: five additional correct predictions out of 669, a gain of roughly **0.75 percentage points**.

The improvement in probability quality is more noticeable than the improvement in the most likely class. No uncertainty analysis has established whether these gains are robust across games, seasons, or pitchers.

### B. More history is not yet proven to be consistently better

The 256-pitch window produced the best individual validation result. However, seed 2 performed less well, and only two seeds were tested per window. We cannot conclude that 256 pitches is the optimal history length.

The window comparison also changes positional-embedding parameter count. It is not a perfectly isolated comparison of information quantity alone.

### C. Results are sensitive to random seed

At 64 pitches, validation log loss ranges from **1.4077 to 1.4594** across the two seeds. At 256 pitches, it ranges from **1.3881 to 1.4231**. These differences are material relative to the gain over baseline.

A single successful run is insufficient evidence of a dependable improvement. Future comparisons should summarize multiple seeds rather than highlight only the best one.

### D. Longer training shows overfitting

In the 256-pitch, seed-1 run:

| Checkpoint / epoch | Training loss | Validation log loss |
| --- | ---: | ---: |
| Selected epoch 6 | 1.1311 | 1.3881 |
| Epoch 10 | 1.0882 | 1.4276 |
| Epoch 20 | 0.8343 | 1.6801 |

Training loss keeps falling while validation loss worsens. This is evidence that later epochs fit the training data increasingly well without improving generalization. Selecting the best validation checkpoint already protects evaluation from using the final, overfit epoch; automatic early stopping could also save computation.

### E. Fastballs and sliders are learned more successfully than other classes

For the best validation-log-loss run:

| Class | Validation support | F1 |
| --- | ---: | ---: |
| FF — four-seam fastball | 294 | 0.589 |
| SL — slider | 149 | 0.406 |
| CH — changeup | 93 | 0.104 |
| FC — cutter | 40 | 0.032 |
| CU — curveball | 93 | 0.113 |

Macro F1 is **0.249**. The model's roughly 45% accuracy therefore does not represent balanced performance across pitch types. Less frequent classes remain a major weakness. A confusion matrix and probability analysis are needed to explain the errors more precisely.

### F. Validation and test rankings differ

The best validation run has test log loss **1.4288**, test accuracy about **43.6%**, and test top-2 accuracy about **71.6%**. The 128-pitch, seed-2 run has lower test log loss (**1.4086**) despite worse validation performance.

This shows that rankings are not stable across the two evaluation seasons. It does not justify selecting a model using test results. It also does not establish a test improvement over the pitch-mix baseline, because the test baseline has not been recorded alongside these results.

### G. We have not isolated the value of the transformer or its history

The model uses context and history together. We have not yet completed matched comparisons against context-only models, short-history regression, or strong tabular baselines. We therefore cannot attribute the observed gain specifically to attention, long history, or any particular feature group.

## 6. Limitations and reproducibility gaps

1. **Single-pitcher scope.** These results describe Scherzer; they do not establish generalization to other pitchers.
2. **Small validation set.** There are only 669 validation pitches, including 40 cutters. Small changes in correct predictions can move headline metrics.
3. **Test results have already been inspected repeatedly.** The 2025 split should be described as an inspected test set. Further model decisions should use validation, with a separate untouched holdout considered for a final assessment.
4. **ID vocabulary construction uses all splits.** Player-ID mappings and batter rarity counts are currently constructed from the full supplied frame. This exposes future category presence/frequency, even though it does not directly use future labels. Fit and freeze those mappings on training data before making stronger leakage claims.
5. **Incomplete experiment metadata.** The scoreboard records training settings but omits model configuration, history length, and a unique checkpoint reference. General checkpoint filenames can be overwritten by later runs. The overnight directory preserves the six grid checkpoints separately.
6. **Demo and evaluation probabilities differ in handling unused classes.** The training evaluator restricts probabilities to the five training classes and renormalizes them; the demo currently applies softmax over all ten classes directly. Align these paths before presenting demo probabilities as identical to scored probabilities.
7. **No demonstrated advantage over a matched regression baseline yet.** The teammate's R work needs the same evaluation rows, class handling, and probability metrics before a fair comparison can be made.

## 7. Current demo model

`demo_prediction.py` explicitly loads:

```text
results/overnight/20261110_224536/checkpoints/pitch_type_w256_seed1.pt
```

This is the preserved 256-pitch, seed-1 model selected at epoch 6. It is the best recorded validation-log-loss run. The generic `results/checkpoints/pitch_type_w256.pt` file has since been replaced by another experiment, so it should not be assumed to contain the same model.

The demo displays a requested number of held-out 2025 examples, their actual pitch types, the top three predictions, and accuracy over those displayed examples. That small displayed sample is illustrative and should not substitute for full-split metrics.

## 8. Recommended next experiments

1. **Make evaluation consistent:** export common pitch IDs, labels, and class probabilities for all models; calculate both validation and test pitch-mix baselines; align demo and evaluator probability handling.
2. **Improve experiment records:** save complete model/training configuration, seed, feature version, checkpoint path, and full split metrics under a unique run ID. Save train-fitted vocabularies and scalers with each model.
3. **Compare useful baselines:** training pitch mix, count/handedness-conditioned pitch mix, and multinomial regression with recent history, all evaluated on identical rows.
4. **Run feature comparisons:** context only, context plus the last few pitches, and longer history. These establish whether history and longer windows add measurable value.
5. **Address overfitting:** try early stopping and controlled changes to model size or regularization. Change one setting at a time and repeat promising configurations across seeds.
6. **Analyze errors:** inspect confusion matrices and performance by class, count, handedness, and game. Quantify uncertainty with a method that respects grouping by game.

The next milestone is a reproducible comparison showing whether the transformer adds useful predictive information beyond simpler methods, and where that improvement occurs.

## Evidence and source files

- `results/scoreboard.jsonl`: ten recorded runs, validation metrics, and test log loss.
- `results/overnight/20261110_224536/logs/`: epoch-by-epoch logs and printed test metrics for the six-run grid.
- `results/overnight/20261110_224536/checkpoints/`: preserved grid models and their architecture metadata.
- `src/model/transformer.py`: architecture.
- `src/model/config.py`: model and training settings.
- `src/model/train.py`: optimization, checkpoint selection, probability restriction, and scoring.
- `src/model/dataset.py` and `src/features/sequence_features.py`: dataset construction and inputs.
- `docs/DESIGN_DECISIONS.md`: taxonomy, splits, eligibility rules, and project decisions.
- `tests/run_overnight_experiments.sh`: experiment grid and artifact preservation.
- `demo_prediction.py`: current demo checkpoint and prediction display.
