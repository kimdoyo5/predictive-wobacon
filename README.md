# predictive-wobacon

**Projecting a pitcher's next-year contact quality from exit velocity and launch angle alone.**

`xwOBAcon` is expected wOBA on contact: Statcast's `estimated_woba_using_speedangle` averaged over a pitcher's batted balls. It predicts a pitcher's *next-year* xwOBAcon from only the exit velocity (EV) and launch angle (LA) of the contact they allowed *this* year.

The whole model is one 2-D lookup surface over (EV, LA): score every batted ball, average per pitcher, done. No pitch type, count, spin, handedness, or identity. The only signal is *how a pitcher's contact is distributed in (EV, LA) space*.

## Quickstart

```bash
uv sync                              # Python 3.11+, deps via uv
uv run python src/fetch_savant.py    # pull Statcast (once; minutes per year)
uv run python src/ensemble.py        # train + cache the grid (~2-3 min)
uv run python src/leaderboard.py     # score baselines + render charts (~1 min)
```

## How it works

1. **Two component models**, fit at the `(pitcher, year)` level so a pitcher-season counts proportionally, not by batted-ball volume:
   - a **GAM**: 50×50 quantile-knot B-spline tensor product over (EV, LA), P-spline penalty, fit by weighted ridge on per-group mean basis activations; and
   - **LightGBM**: custom objective aggregating per-pitch predictions to per-group means *before* computing gradients.
2. **Ensemble grid**: 50/50 average on a 481×721 (EV, LA) mesh, Gaussian-smoothed (σ = 5 mph, 5°). This cached grid *is* the production model.
3. **Sample-size-aware calibration** at prediction time (below).

Everything downstream reads the one cached grid (no retraining), so iterating on metrics and charts is cheap.

## The group-aggregated objective

LightGBM optimizes a sum of *per-sample* losses and wants a gradient and Hessian for each batted ball. But the target lives at the *group* level (a pitcher-season is scored by the **mean** of its per-ball predictions, not ball by ball), and weighting balls equally would let high-volume pitchers dominate the fit. So the loss is defined on group means and backpropagated to events.

Let ball $e$ belong to group $p$ (a pitcher-year) with $N_p$ balls, weight $w_p$, and next-year target $y_p$. LightGBM's raw output for a ball is $f_e$, and the group prediction is their mean:

$$m_p = \frac{1}{N_p}\sum_{e \in p} f_e, \qquad L = \sum_p w_p\,(m_p - y_p)^2 .$$

At the group level the gradient and curvature are immediate:

$$\frac{\partial L}{\partial m_p} = 2 w_p (m_p - y_p), \qquad \frac{\partial^2 L}{\partial m_p^2} = 2 w_p .$$

Since $\partial m_p / \partial f_e = 1/N_p$, the chain rule gives the per-ball gradient:

$$g_e = \frac{\partial L}{\partial f_e} = \frac{\partial L}{\partial m_p}\cdot\frac{1}{N_p} = \frac{2 w_p (m_p - y_p)}{N_p} .$$

For the Hessian we **spread the group curvature evenly** over its balls, rather than take the naive per-ball second derivative $\big(\partial^2 L/\partial f_e^2 = 2 w_p / N_p^2\big)$:

$$h_e = \frac{1}{N_p}\frac{\partial^2 L}{\partial m_p^2} = \frac{2 w_p}{N_p} .$$

This is the deliberate part. A boosting leaf that covers all $N_p$ balls of a group takes the Newton step

$$-\frac{\sum_{e \in p} g_e}{\sum_{e \in p} h_e} = -\frac{2 w_p (m_p - y_p)}{2 w_p} = -(m_p - y_p),$$

moving the group mean exactly onto its target. Because the per-ball Hessians sum back to the group curvature, $\sum_{e \in p} h_e = \partial^2 L/\partial m_p^2$, tree-wise Newton updates behave as if they were optimizing the per-pitcher means directly. (See `lgbm_aggregated_objective` in `src/ensemble.py`.)

## The calibration trick

Raw grid predictions are too compressed toward the league mean (calibration slope ≈ 2.3). But a pitcher's grid-average is a *noisy* estimate whose noise shrinks with sample size, so the right decompression grows with n:

```
b(n) = 1 + (b_max − 1) · n / (n + n₀)     # b_max = 10, n₀ = 8000
pred = pm + b(n) · (raw − pm)             # pm = league pred mean
```

A single global stretch can't win everywhere: it over-corrects noisy season lines and under-corrects reliable career totals. `b(n)` decouples them: seasons (n ≈ 50–600) get b ≈ 1.1–1.6 (RMSE-optimal), career aggregates (n ≈ 1000–5000) get b ≈ 2.5–4.5, pulling career calibration from slope 1.78 to **1.01** with neutral-to-better season RMSE.

## Results

Held out chronologically: train 2016–2023, test **2024 → predict 2025**, min(IP, IP_next) ≥ 30, min-of-pair weighted. Benchmarked against Tango's pwOBAcon, Max's pwOBAcon+, xwOBAcon carryover, a per-bucket OLS, and a naive constant.

| Slice | RMSE | r | r (self) |
|---|---|---|---|
| Pitcher-year, IP ≥ 100 | .0181 | .50 | .71 |
| Pitcher-year, IP ≥ 20 | .0261 | .37 | .57 |
| Pitch-type, BIP ≥ 100 | .0324 | .55 | .81 |
| Pitch-type, BIP ≥ 20 | .0485 | .38 | .61 |

`r` = corr(prediction, next-year actual); `r (self)` = year-to-year stability.

`leaderboard.py` also renders (EV, LA) heatmaps, top/bottom-20 boards, a calibration scatter, and Cronbach's-α reliability curves into `artifacts/`.

## Layout

```
src/
  fetch_savant.py  # Statcast scraper → data/raw/pitches_{year}.parquet
  data.py          # loaders, BBE filtering, season rates
  eval.py          # chronological splits + weighted metrics
  ensemble.py      # train GAM + LGBM, cache the grid, b(n) calibration
  leaderboard.py   # score baselines, render charts
```

Tunables live at the top of `src/ensemble.py`.

## References

- [Statcast Lab: xwOBAcon vs predictive wOBA](https://tangotiger.com/index.php/site/comments/stacast-lab-xwobacon-v-predictive-wobacon)
- [Introducing predictive wOBA](https://tangotiger.com/index.php/site/comments/introducing-predictive-woba)
