# predictive-wobacon

Per-pitch (exit velocity, launch angle) → next-year pitcher xwOBAcon. A
50/50 GAM-spline + LightGBM ensemble on a dense (EV, LA) grid, Gaussian
smoothed, post-hoc linear-stretched, then BIP-averaged per pitcher-year.

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
them, smooths, stretches, and caches.

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
   4. Mean-preserving linear stretch by `b = LINEAR_STRETCH_B = 1.3`
      around the trainval per-group pred mean.

- **Writes** `artifacts/ensemble_grid.npz`: production grid (`grid`),
  un-smoothed `spline_grid` / `lgbm_grid`, shared `ev_grid` / `la_grid`,
  and stretch provenance (`stretch_b`, `stretch_pm`, `stretch_a`). Also
  three heatmap PNGs.

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
| `LINEAR_STRETCH_B` | 1.3 | Post-hoc calibration stretch around trainval pred mean |

### Environment overrides

- `SCALE_B=<float>` — override `LINEAR_STRETCH_B` for one run. `SCALE_B=1.0`
  disables the stretch (useful for ablation).
- `MIN_BIP_TRAINVAL=<int>` — override the training filter (set internally
  to `50` by both entrypoints, honored by `eval.load_splits`).

## On the calibration stretch (b=1.3)

The unscaled smoothed grid produces per-pitcher predictions that are
~3× too compressed around the league mean vs the actuals (slope of
weighted OLS on the career calibration plot ≈ 2.3). We can't recover
that full slope without breaking RMSE — pitcher skill is noisy
year-over-year, so over-stretching predictions actively hurts
predictive RMSE on the noisier slices.

`b = 1.3` was chosen by sweeping `b ∈ {1.0, 1.1, ..., 2.0}` against all
four leaderboard slices and picking the largest `b` that doesn't
regress the 20-threshold cuts. Result vs unscaled baseline:

| Slice | RMSE Δ |
|---|---|
| bip100 | **−1.9%** |
| ip100 | **−0.9%** |
| bip20 | −0.2% |
| ip20 | +0.2% (essentially tied) |

R² and R²(self) are mathematically invariant under linear stretch, so
this is a pure RMSE-side fix — it does not lift the model's signal
ceiling.

## Module layout

```
src/
  data.py          # parquet loaders, OUTS_PER_EVENT, BBE filtering
  eval.py          # load_splits, weighted_rmse / weighted_r2 / weighted_corr
  fetch_savant.py  # baseballsavant scraper (network)
  ensemble.py      # train + smoothed + stretched grid, save npz
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
