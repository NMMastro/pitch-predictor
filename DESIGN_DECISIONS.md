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

## Additional notes

### Long-running averages

Two windows per feature:

- Career-to-date expanding
- Rolling last ~200 pitches

Neither resets at a season or split boundary.

### No current-pitch information

Nothing measured from the pitch being predicted may be used as an input. Enforce this as an **allowlist** of
permitted columns, not a denylist — most of the 119 Statcast columns are outcomes of the pitch.

Excluded as inputs:

- **Flight and release** — `release_speed`, `release_spin_rate`, `spin_axis`, `effective_speed`,
  `release_extension`, `release_pos_*`, `pfx_x`, `pfx_z`, `vx0`–`az0`
- **Location** — `plate_x`, `plate_z`. These *are* the zone label. Keep them on the table for the demo
  heatmap, never as features.
- **Outcome** — `type`, `description`, `events`, `bb_type`, `launch_speed`, `launch_angle`,
  `hit_distance_sc`, all `estimated_*`, all `woba_*`, `delta_run_exp`, `delta_home_win_exp`
- **Post-pitch state** — all `post_*` score columns

These same quantities *are* allowed as lagged features describing previous pitches.

`sz_top` and `sz_bot` are allowed: they are a property of the batter rather than of the pitch.

`arm_angle`, `bat_speed` and `swing_length` are excluded entirely — all three are 100% null through 2018, and
`bat_speed` is still 100% null in 2023.
