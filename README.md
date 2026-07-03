# predictive-wobacon

**Projecting a pitcher's next-year contact quality from exit velocity and launch angle alone.**

`xwOBAcon` is expected wOBA on contact — Statcast's `estimated_woba_using_speedangle` averaged over a pitcher's batted balls. This project predicts a pitcher's *next-year* xwOBAcon from nothing but the exit velocity (EV) and launch angle (LA) of the contact they allowed *this* year.

The entire model is a single 2-D lookup surface over (EV, LA): score every batted ball, average per pitcher, and you have a projection. No pitch type, count, spin, batter handedness, or pitcher identity — the only signal is *how a pitcher's contact is distributed in (EV, LA) space*.

## Quickstart

```bash
uv sync                              # Python 3.11+, deps via uv
uv run python src/fetch_savant.py    # pull Statcast (once; minutes per year)
uv run python src/ensemble.py        # train + cache the grid (~2-3 min)
uv run python src/leaderboard.py     # score baselines + render charts (~1 min)
```

## How it works

1. **Two component models**, both fit at the `(pitcher, year)` level so every pitcher-season counts proportionally rather than by batted-ball volume:
   - a **GAM** — 50×50 quantile-knot B-spline tensor product over (EV, LA), P-spline penalty, fit by weighted ridge on per-group mean basis activations; and
   - **LightGBM** — with a custom objective that aggregates per-pitch predictions to per-group means *before* computing gradients.
2. **Ensemble grid** — average the two surfaces 50/50 on a dense 481×721 (EV, LA) mesh and Gaussian-smooth (σ = 5 mph, 5°). This cached grid *is* the production model.
3. **Sample-size-aware calibration** applied at prediction time (see below).

Everything downstream reads the one cached grid — no retraining — so iterating on metrics and charts is cheap.

## The calibration trick

Raw grid predictions are far too compressed toward the league mean (calibration slope ≈ 2.3). But a pitcher's grid-average is a *noisy* estimate whose noise shrinks with sample size, so the right amount of decompression depends on how many batted balls back it:

```
b(n) = 1 + (b_max − 1) · n / (n + n₀)     # b_max = 10, n₀ = 8000
pred = pm + b(n) · (raw − pm)             # pm = league pred mean
```

A single global stretch can't win everywhere — it over-corrects noisy season lines and under-corrects reliable career totals. `b(n)` decouples them: seasons (n ≈ 50–600) get b ≈ 1.1–1.6 (RMSE-optimal), while career aggregates (n ≈ 1000–5000) get b ≈ 2.5–4.5, pulling career calibration from slope 1.78 to **1.01** with neutral-to-better season RMSE.

## Results

Held out chronologically — train 2016–2023, test **2024 → predict 2025**, min(IP, IP_next) ≥ 30, weighted by the min of the pair. The ensemble is benchmarked against Tango's pwOBAcon, Max's pwOBAcon+, raw xwOBAcon carryover, a per-bucket OLS, and a naive constant, at both pitcher-season and pitcher × pitch-type grains.

| Slice | RMSE | r | r (self) |
|---|---|---|---|
| Pitcher-year, IP ≥ 100 | .0181 | .50 | .71 |
| Pitcher-year, IP ≥ 20 | .0261 | .37 | .57 |
| Pitch-type, BIP ≥ 100 | .0324 | .55 | .81 |
| Pitch-type, BIP ≥ 20 | .0485 | .38 | .61 |

`r` = corr(prediction, next-year actual); `r (self)` = year-to-year stability of the predictor.

`leaderboard.py` also renders (EV, LA) contact-quality heatmaps, top/bottom-20 pitcher boards, per-pitcher calibration scatter, and Cronbach's-α reliability-vs-sample-size curves into `artifacts/`.

## Layout

```
src/
  fetch_savant.py  # Statcast scraper → data/raw/pitches_{year}.parquet
  data.py          # parquet loaders, BBE filtering, season rates
  eval.py          # chronological splits + weighted metrics
  ensemble.py      # train GAM + LGBM, build/cache the grid, b(n) calibration
  leaderboard.py   # score every baseline, render every chart
```

Key tunables live at the top of `src/ensemble.py`; the source is documented inline.

## References

- [Statcast Lab: xwOBAcon vs predictive wOBA](https://tangotiger.com/index.php/site/comments/stacast-lab-xwobacon-v-predictive-wobacon)
- [Introducing predictive wOBA](https://tangotiger.com/index.php/site/comments/introducing-predictive-woba)
