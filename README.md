# predictive-wobacon

Per-pitch (exit velocity, launch angle) → next-year pitcher xwOBAcon. A
50/50 GAM-spline + LightGBM ensemble on a dense (EV, LA) grid, Gaussian
smoothed, BIP-averaged per pitcher-year, then stretched by a
sample-size-dependent calibration b(n).

## What this is

`xwOBAcon` = expected wOBA on contact (mean of Statcast's
`estimated_woba_using_speedangle` over a pitcher's batted-ball events).
We model the *next-year* per-pitcher xwOBAcon as a function only of the
pitcher's *current-year* per-pitch (EV, LA). The output is a single 2-D
lookup grid: given any batted ball's exit velocity and launch angle, you
get a predicted contact quality, and averaging those per-BIP predictions
across a pitcher's season gives that pitcher's predicted next-year
xwOBAcon.

The model lives entirely in `src/`. Three scripts run end-to-end on a
laptop in under a minute (after the initial data pull):

```
uv run python src/fetch_savant.py    # ~minutes per year, only needed once
uv run python src/ensemble.py        # ~30s: train + cache the production grid
uv run python src/leaderboard.py     # ~15s: render every PNG from the cache
```

## Useful links
https://tangotiger.com/index.php/site/comments/stacast-lab-xwobacon-v-predictive-wobacon
https://tangotiger.com/index.php/site/comments/introducing-predictive-woba


## Install

```
uv sync
```

This project uses [uv](https://docs.astral.sh/uv/) and Python 3.11+. The
core deps are polars, numpy, scipy, scikit-learn, lightgbm, matplotlib,
and requests.

## Pipeline

### 1. `src/fetch_savant.py [years...]`

Pulls raw Statcast pitch logs from baseballsavant for each year (defaults
to all seasons in `SEASON_DATES`, currently 2016–2025). 6-thread workers
with retry/backoff, one HTTP call per day.

- **Reads** baseballsavant CSV endpoint (network).
- **Writes** `data/raw/pitches_{year}.parquet` — one row per pitch.
  Columns: `pitcher_id`, `year`, `game_date`, `ab_number`, `index_play`,
  `pitch_type`, `launch_speed`, `launch_angle`, `event_type`,
  `xwoba_value`.
- Skip if you already have the parquets.

### 2. `src/ensemble.py`

Trains two component models on per-event (EV, LA) with a custom
group-aggregated MSE loss (grouped by `(pitcher_id, year)`), blends
them, smooths, and caches.

1. **Splits.** `eval.load_splits()` partitions chronologically into
   trainval (2016–2023) and test (2024 → predicting 2025). Training
   filter `MIN_BIP_TRAINVAL=50`; test filter `min(IP, IP_next) >= 30`
   weighted by `min(IP, IP_next)`.
2. **Splines** (`train_splines`): 50×50 quantile-knot degree-1 B-splines
   on (EV, LA), per-group mean tensor-product basis, P-spline ridge
   penalty `α=1000`.
3. **LightGBM** (`train_lgbm`): 3000 rounds, `num_leaves=5`,
   `min_data_in_leaf=1000`. Custom objective aggregates per-event preds
   back to per-pitcher-year MSE before computing gradients.
4. **Production grid** (`build_smoothed_ensemble_grid`):
   1. Evaluate both models on a 481×721 (EV, LA) mesh (0.25 mph × 0.25°).
   2. `ens = 0.5 * spline_grid + 0.5 * lgbm_grid`.
   3. Gaussian smooth at `σ = (5 mph, 5°)`.

   The calibration stretch is NOT baked into the grid. Production
   predictions (`calibrated_grid_predict_per_group`) average the grid
   per BIP, then stretch each group's mean by
   `b(n) = 1 + (STRETCH_B_MAX − 1) · n / (n + STRETCH_N0)` around the
   trainval pred mean `pm`, where `n` is the group's BIP count.

- **Writes** `artifacts/ensemble_grid.npz`: production grid (`grid`),
  un-smoothed `spline_grid` / `lgbm_grid`, shared `ev_grid` / `la_grid`,
  and the stretch center (`stretch_pm`). Also three heatmap PNGs.

### 3. `src/leaderboard.py`

Reads the cached grid, scores every baseline, renders every PNG. **No
model retraining**, so iterating on visualizations is cheap.

For each of the four grain/threshold combos
(`pitch_type × {bip≥20, bip≥100}`, `pitcher_year × {ip≥20, ip≥100}`),
`run_grain` computes weighted RMSE / r (`corr(pred, actual)`) / r(self)
(`corr(pred_N, pred_N+1)`) for: `ensemble`, `gam`, `lgbm`, `naive`
(constant), `pwobacon` (12-bucket OLS over LA×EV), and univariate
baselines (`xwobacon`, `avg_ev`, `avg_la`; pitcher-year grain also shows
`k_pct`, `bb_pct`, `hr_pct` for self-stability only). Emits
`SLICE_METRIC <slug> rmse=... r=... r_self=...` to stderr for sweep
harnesses, then renders the PNG.

- **Top/bottom-20 charts** for the bip20 and ip20 grains: career
  2016–2025, filtered to `n_bip≥100` or `IP≥100`, ranked by ensemble
  pred. Pitcher names auto-fetched from MLB statsapi and cached in
  `data/raw/pitcher_names.parquet`.
- **Cronbach α** for one of each grain: empirical chronological accrual
  curves vs sample size.
- **Calibration scatter**: per-pitcher career (min_bip=10), pred vs
  actual, circle area ∝ BIP, with weighted OLS line.

## Outputs (`artifacts/`)

| File | What |
|---|---|
| `ensemble_grid.npz` | Cached production grid + components + provenance |
| `heatmap_ensemble.png` | Production grid on (EV, LA) |
| `heatmap_gam.png` / `heatmap_lgbm.png` | Un-smoothed component grids |
| `leaderboard_pitch_type_bip{20,100}.png` | Per-(pitcher × pitch_type × year) metrics table |
| `leaderboard_pitcher_year_ip{20,100}.png` | Per-(pitcher × year) metrics table |
| `top_bottom_ensemble_pitch_type_*.png` | Top/bottom 20 pitcher × pitch_type, career |
| `top_bottom_ensemble_pitcher_year_*.png` | Top/bottom 20 pitchers, career |
| `cronbach_alpha_{pitch_type,pitcher_year}.png` | Empirical reliability vs sample size |
| `calibration_scatter.png` | Per-pitcher career calibration plot |

## Key constants and tunables

All in `src/ensemble.py`:

| Constant | Default | Role |
|---|---|---|
| `SPLINE_KNOTS` | 50 | Quantile knots per axis for the GAM tensor-product basis |
| `SPLINE_DEGREE` | 1 | Piecewise-linear B-splines |
| `SPLINE_ALPHA` | 1000.0 | P-spline 2nd-difference penalty |
| `LGBM_NUM_BOOST_ROUND` | 3000 | LightGBM boosting rounds |
| `GRID_N_EV` × `GRID_N_LA` | 481 × 721 | Grid resolution (0.25 mph × 0.25°) |
| `SMOOTH_SIGMA_EV` / `SMOOTH_SIGMA_LA` | 5.0 / 5.0 | Gaussian smoothing σ |
| `STRETCH_B_MAX` / `STRETCH_N0` | 10.0 / 8000 | Sample-size-dependent calibration stretch b(n) |

### Environment overrides

- `MIN_BIP_TRAINVAL=<int>` — override the training filter (set internally
  to `50` by both entrypoints, honored by `eval.load_splits`).

## On the calibration stretch b(n)

The raw smoothed grid produces per-pitcher predictions that are far too
compressed around the league mean: weighted OLS of actual on pred on the
career calibration plot has slope ≈ 2.3. A group's grid-average is a
noisy estimate of its true contact-quality distribution, and that noise
shrinks with sample size, so the optimal decompression is
sample-size-dependent:

```
b(n) = 1 + (STRETCH_B_MAX − 1) · n / (n + STRETCH_N0)
pred = pm + b(n) · (raw − pm)
```

with `n` the group's BIP count and `pm` the trainval pred mean. A single
global `b` (the old `b = 1.3`) over-stretches noisy small-n groups and
under-stretches reliable large-n ones — it could only reach career slope
1.78 before regressing season-level RMSE. Because career aggregates
(n ≈ 1000–5000) are much larger than season groups (n ≈ 50–600), b(n)
decouples the two: season groups get the RMSE-optimal b ≈ 1.1–1.6 while
career aggregates get b ≈ 2.5–4.5, reaching 1:1 career calibration.

`(b_max, n0) = (10, 8000)` — near-linear in n over the observed range —
was chosen by sweeping both parameters against the four leaderboard
slices plus the career calibration slope. Result vs the old global
`b = 1.3`:

| Slice | RMSE Δ | r Δ | r(self) Δ |
|---|---|---|---|
| bip20 | −0.2% | +0.004 | +0.022 |
| bip100 | +0.5% | −0.001 | +0.005 |
| ip20 | −0.4% | +0.010 | +0.039 |
| ip100 | +0.2% | +0.001 | +0.008 |
| career calibration slope | 1.78 → **1.01** | | |

Unlike a global linear stretch (which leaves r / r(self) invariant),
b(n) shrinks noisy groups relative to reliable ones, which is why the
correlations move — consistently up on the noisy 20-threshold cuts.
One trade-off: the career-scatter correlation itself drops (r² 0.54 →
0.44) because the career plot partly reflects concurrent (same-season)
fit, which no next-year-calibrated stretch can recover.

## Module layout

```
src/
  data.py          # parquet loaders, OUTS_PER_EVENT, BBE filtering
  eval.py          # load_splits, weighted_rmse / weighted_r2 / weighted_corr
  fetch_savant.py  # baseballsavant scraper (network)
  ensemble.py      # train + smoothed grid + b(n) calibration, save npz
  leaderboard.py   # read npz, score baselines, render every PNG
artifacts/         # all generated outputs land here
data/raw/          # pitches_{year}.parquet and pitcher_names.parquet
```

## Methodology notes

- **Why a 2-D grid?** EV and LA are the only inputs. Two pitchers with
  identical (EV, LA) distributions get identical predictions — the
  pitcher-level signal is *purely* about how a pitcher's contact is
  distributed in (EV, LA) space. No pitch type, count, batter handedness,
  pitcher identity, or spin enters the model.
- **Why next-year, not concurrent?** Concurrent xwOBAcon is essentially
  self-correlation. Next-year forces the model to separate stable skill
  from year-specific noise — the relevant question for projection.
- **Why a custom group-aggregated LightGBM objective?** Per-event MSE
  would overweight pitchers with many BBE. The custom objective collapses
  per-event preds to per-group means before computing gradients, so each
  pitcher-year contributes proportionally to its `min(n_bip, n_bip_next)`
  weight.
- **Why the σ=5 smoothing?** LightGBM produces step-function edges that
  don't reflect the true smoothness of the underlying contact-quality
  surface. The smoothing regularizes those edges before the per-BIP
  bilinear interpolation.
- **Cronbach's α ceiling.** Empirically α plateaus around 0.55 at the
  100-threshold cuts (so √α ≈ 0.74). A well-calibrated model could push
  `std(pred) / std(actual)` toward 0.74; the current model is at ~0.33.
  Roughly 2× headroom remains for any model change that adds signal
  rather than redistributes it.
