# Design Decisions

This file records the decisions the whole team codes against: class definitions, data scope, the split,
evaluation metrics, and filtering rules. 

## 1. Pitch type classes

Eleven classes. Each common pitch type stands alone; the super-rare types are pooled into `OTHER`.

| Code | Pitch | 2024 + 2025 pitches | 2024 + 2025 % |
| --- | --- | --- | --- |
| `FF` | Four-seam fastball | 461,253 | 31.93% |
| `SI` | Sinker | 225,986 | 15.65% |
| `SL` | Slider | 210,344 | 14.56% |
| `CH` | Changeup | 148,068 | 10.25% |
| `FC` | Cutter | 113,640 | 7.87% |
| `ST` | Sweeper | 108,323 | 7.50% |
| `CU` | Curveball | 92,771 | 6.42% |
| `FS` | Splitter | 46,383 | 3.21% |
| `KC` | Knuckle curve | 25,827 | 1.79% |
| `SV` | Slurve | 7,360 | 0.51% |
| `OTHER` | Pooled rare types (below) | 4,407 | 0.31% |

Counts are the 2024 and 2025 seasons combined, regular season plus postseason — 1,444,362 pitches, after the
drops listed below.

**What goes into `OTHER`.** Five codes, too infrequent to learn as separate classes:

| Code | Pitch |
| --- | --- |
| `EP` | Eephus |
| `FO` | Forkball |
| `CS` | Slow curve |
| `KN` | Knuckleball |
| `SC` | Screwball |


**Dropped as non-pitches.** These are not pitch-selection decisions and must be removed before training:

- `PO` — pitchout
- `IN` — intentional ball
- `UN` — unknown / classifier failure
- `FA` — generic fastball fallback

**Drop rows where `pitch_type` is null.** 0.38% of rows; the same rows are also null in `plate_x`,
`plate_z` and `release_speed`.

## 2. Pitch zone classes

All 13 Statcast zones, taken directly from the `zone` column. Zones 1–9 divide the strike zone into a 3×3
grid. Zones 11–14 are the four quadrants outside it, split by the zone's vertical and horizontal centre lines.

| Zone | Represents |
| --- | --- |
| `1` | In zone — upper third, left |
| `2` | In zone — upper third, middle |
| `3` | In zone — upper third, right |
| `4` | In zone — middle third, left |
| `5` | In zone — middle third, middle |
| `6` | In zone — middle third, right |
| `7` | In zone — lower third, left |
| `8` | In zone — lower third, middle |
| `9` | In zone — lower third, right |
| `11` | Outside — upper left quadrant |
| `12` | Outside — upper right quadrant |
| `13` | Outside — lower left quadrant |
| `14` | Outside — lower right quadrant |


## 3. Evaluation metrics

**Primary:** multiclass log loss.

**Reported alongside:**

- Accuracy
- Top-2 accuracy
- Macro-averaged precision, recall and F1
- Per-class F1

## 4. Split

**Scope: everything except spring training.** Discard `game_type == 'S'` and keep the regular season (`R`)
and postseason (`F`/`D`/`L`/`W`), in train, validation and test alike.

Spring training is dropped because it is a different data-generating process: pitchers experiment with pitches
they never throw in games, and rosters are full of non-MLB hitters. Those pitches are excluded from the
running averages in the additional notes too, since they would bias a pitcher's repertoire estimates.

Postseason is kept. A pitcher's postseason pitch mix does differ measurably from their own regular-season mix,
but the effect is small — about 1.2 percentage points of top-1 accuracy — and it is not systematic: only about
half of pitcher-seasons pitch worse-than-expected in October. Since postseason appears in train, validation and
test alike, the mixture is consistent across splits and needs no indicator feature.

| Split | Seasons |
| --- | --- |
| Train | 2016–2023 |
| Validation | 2024 |
| Test | 2025 |


**Why 2016.** Two things begin that season. Sweeper and slurve (`ST`, `SV`) exist as labels from 2016 and are
absent in 2015 and earlier, so older seasons would have sweepers silently labelled as sliders. And
`release_spin_rate`, `spin_axis`, `release_extension` and `effective_speed` go from 100% null in 2015 to ~1%
null in 2016.

**Leakage rule.** A feature for a given pitch may use information from every pitch that happened strictly
*before* it in real time, and nothing else. History crosses split boundaries — a 2025 pitch may legitimately
use that pitcher's full 2016–2024 record.

## 5. Minimum pitch thresholds

- **Training rows** — drop rows where the pitcher has fewer than 25 prior pitches. Applies to train,
  validation and test alike.
- **Demo** — only pitchers with 250 or more career-to-date pitches appear in the dashboard.

**Compute the running averages first, then filter rows.** Those first 25 pitches still count toward every
pitcher's accumulated history; they are just not used as training examples.

## 6. Feature set

The columns below are the agreed starting feature set, selected by hand-inspecting the data for what looks
predictive of the next pitch. They are split into three groups by how they may be used, which is dictated by
the leakage rule in section 4: a feature may only use information available *strictly before* the pitch being
predicted.

### 6a. Current-pitch inputs — fed directly

Known before the pitch is thrown, so they are safe as inputs for the pitch being predicted.

| Column | Notes |
| --- | --- |
| `balls`, `strikes` | The count. |
| `outs_when_up` | Outs in the half-inning. |
| `inning`, `inning_topbot` | Inning and top/bottom. Cap `inning` at 10+ (extra innings are rare and high-variance); bucket rather than feed raw. |
| `stand`, `p_throws` | Batter and pitcher handedness. |
| `on_1b`, `on_2b`, `on_3b` | Baserunner occupancy. The raw columns hold a runner's MLBAM id — convert each to a boolean (occupied / empty) in the pipeline. Presence, never identity. |
| `bat_score`, `fld_score` | Pre-pitch score for each side. Prefer the derived `bat_score_diff` / `home_score_diff` as the model input. |
| `sz_top`, `sz_bot` | Batter strike-zone top/bottom. Allowed — a property of the batter, not the pitch (see section 4). |
| `n_thruorder_pitcher` | Times through the order. |
| `n_priorpa_thisgame_player_at_bat` | Prior PAs this game for this batter. |
| `pitcher_days_since_prev_game`, `batter_days_since_prev_game` | Rest / fatigue. |
| `age_pit`, `age_bat` | Player ages. Use these, not the `*_legacy` variants. |
| `batter`, `pitcher` | MLBAM ids — fed as embeddings, never as raw integers. |
| `fielder_2` | Catcher id (embedding). Catchers influence pitch calling, so this is kept. |

### 6b. Lagged previous-pitch features — fed as history

These are measured *from the pitch*, so they are leakage as current-pitch inputs, but legitimate when they
describe pitches that already happened. The transformer consumes a window of previous pitches; each previous
pitch contributes these. This is where the flight / location / outcome columns earn their place.

| Column (of the previous pitch) | Notes |
| --- | --- |
| `pitch_type` | What was actually thrown last — the strongest sequence signal. |
| `release_speed` | Velocity of the prior pitch. |
| `plate_x`, `plate_z`, `zone` | Where the prior pitch ended up. |
| `description` | Prior-pitch result (ball / called strike / foul / swinging strike …). |

The lag is strictly backward: for pitch *t* these describe pitches *t−1, t−2, …* within the window. Windows do
not reset at at-bat, game, season or split boundaries (consistent with the running-average rule below).

### 6c. Bookkeeping — kept on the table, never fed

Used to build sequences, splits, running averages and the demo, but not model inputs.

| Column | Used for |
| --- | --- |
| `game_date`, `game_type`, `game_pk` | Ordering, split assignment, spring-training filter. |
| `at_bat_number`, `pitch_number` | Real-time pitch ordering within a game. |
| `player_name` | Human-readable id. |
| `pitch_type` | The label for the pitch-type model. |
| `plate_x`, `plate_z`, `zone` | Label for the zone model + demo heatmap. (As *current-pitch* values; their lagged forms are features per 6b.) |

### Excluded, and why

- **`bat_win_exp` / `home_win_exp`** — leakage. Win expectancy is reported *after* the pitch's result; use the
  pre-pitch score and base/out state instead.
- **`pitcher_days_until_next_game`, `batter_days_until_next_game`** — leakage by construction: they encode the
  future schedule. The `*_days_since_prev_game` versions are the valid, backward-looking ones.
- **`fielder_3`–`fielder_9`** — defensive alignment, weak signal for pitch *type*; dropped for v1. Revisit via
  `if_fielding_alignment` / `of_fielding_alignment` if ever needed.
- **`home_team`, `away_team`** — park / opponent effects on a single pitcher's pitch selection are marginal;
  left out of v1.
- **`age_pit_legacy`, `age_bat_legacy`** — superseded by `age_pit` / `age_bat`.

### v1 scope: single pitcher

The first model is trained on one pitcher (Max Scherzer, MLBAM 453286) to keep the loop fast. Consequences the
team should keep in mind:

- The label space collapses to the classes he actually throws (roughly `FF`, `SL`, `CH`, `CU`, `FC`); classes
  he never throws will not appear.
- Career-to-date and rolling features only ever see his own history, so the `pitcher` embedding carries no
  signal within this dataset. The feature stays in the schema so the pipeline generalises when more pitchers
  are added.

### Open decisions

Captured but not yet settled. Each needs a call before the feature pipeline is final; a recommendation is
noted for each.

**1. Does the 6b previous-pitch window reset at the at-bat / PA boundary?**
The running averages (below) explicitly never reset. For the *immediate* previous-pitch features (6b), the
pitch before the first pitch of an at-bat belongs to a different batter, and sometimes a different inning. That
is still strictly in the past, so it is not a hard leak, but it mixes contexts — e.g. a lagged `description` of
`hit_into_play` that actually ended the previous PA.
*Recommendation:* do **not** reset the window (keep history continuous), but add boolean flags
`is_first_pitch_of_pa` and `is_first_pitch_of_ab` so the model can discount a "previous pitch" that came from a
different context. Add these flags to 6a if adopted.

**2. How does v1 evaluation handle the collapsed label space?**
`evaluation.metrics` hardcodes all eleven `PITCH_CLASSES`, and `evaluate()` scores log loss, top-2 and
macro-F1 over all eleven. With a single pitcher, ~6 classes never occur, so macro-averaged metrics are dragged
down by structurally impossible classes and `zero_support_classes` fires every run.
*Recommendation:* for v1, fit the label set to the classes the pitcher actually throws (pass a pitcher-specific
`classes` list into `evaluate`), and treat log loss + per-class F1 as the headline metrics; keep macro metrics
but read them knowing the denominator. Revisit when the model is multi-pitcher. Do not change the global
`PITCH_CLASSES` taxonomy — scope the restriction to the v1 run.

**3. Does `batter` identity belong in a single-pitcher model?**
Over one pitcher's rows, most batters appear only a handful of times. A full batter-id embedding table would
mostly memorise noise and can hurt a small v1 model. `stand` already captures the part of the matchup that
generalises.
*Recommendation:* for v1, drop `batter` identity (keep `stand`), or map rare batters to a single `UNK` bucket
(batters seen ≥ K times keep their id, else `UNK`). Keep `batter` in the schema so it returns naturally in the
multi-pitcher model.

## Additional notes

### Long-running averages

Two windows per feature:

- Career-to-date expanding
- Rolling last ~200 pitches

Neither resets at a season or split boundary.

### No current-pitch information

Nothing measured from the pitch being predicted may be used as an input. Enforce this as an **allowlist** of
permitted columns, not a denylist — most of the 119 Statcast columns are outcomes of the pitch. Section 6 is
that allowlist; the categories below are the reasoning behind what it excludes.

Excluded as current-pitch inputs (allowed only as lagged features of previous pitches, per 6b):

- **Flight and release** — `release_speed`, `release_spin_rate`, `spin_axis`, `effective_speed`,
  `release_extension`, `release_pos_*`, `pfx_x`, `pfx_z`, `vx0`–`az0`
- **Location** — `plate_x`, `plate_z`, `zone`. These *are* the zone label. Keep them on the table for the demo
  heatmap, never as current-pitch features.
- **Outcome** — `type`, `description`, `events`, `bb_type`, `launch_speed`, `launch_angle`,
  `hit_distance_sc`, all `estimated_*`, all `woba_*`, `delta_run_exp`, `delta_home_win_exp`
- **Post-pitch state** — all `post_*` score columns, and win-expectancy columns (`bat_win_exp`,
  `home_win_exp`), which are reported after the result.

`arm_angle`, `bat_speed` and `swing_length` are excluded entirely — all three are 100% null through 2018, and
`bat_speed` is still 100% null in 2023.