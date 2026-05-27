"""Final pitcher-year predictive-xwOBAcon ensemble (Gaussian-smoothed).

Two component models trained on per-event (EV, LA) with a custom group-
aggregated MSE loss, grouped by (pitcher_id, year):

  1. GAM splines: 50×50 degree-1 B-splines, P-spline penalty α=1000.
  2. LGBM: 3000 rounds, num_leaves=5, min_data_in_leaf=1000.

Their 50/50 average is evaluated on a dense 481×721 (EV, LA) grid (0.25
mph × 0.25°), then Gaussian-smoothed at σ=(5 mph, 5°) to regularize
LGBM's step-function edges. The smoothed grid is THE production ensemble:
per-group predictions = bilinear interpolation per BIP, averaged per group.

Filters and weights: see eval.load_splits (this module overrides
MIN_BIP_TRAINVAL=50 for training; trainval = train ∪ val with no val
held out).

Usage:
  uv run python src/ensemble.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from sklearn.preprocessing import SplineTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import load_batted_balls
from eval import load_splits, weighted_rmse, weighted_r2

ART = ROOT / "artifacts"

# Component hyperparams (final).
SPLINE_KNOTS  = 50
SPLINE_DEGREE = 1
SPLINE_ALPHA  = 1000.0
LGBM_NUM_BOOST_ROUND = 3000

# Cached (EV, LA) ensemble grid + Gaussian smoothing.
GRID_N_EV = 481         # 0..120 mph at 0.25 mph
GRID_N_LA = 721         # -90..90 deg at 0.25°
SMOOTH_SIGMA_EV = 5.0   # mph
SMOOTH_SIGMA_LA = 5.0   # degrees

# Default group key used by all model-training functions. Override via
# `group_keys=("pitcher_id", "pitch_type", "year")` for the pitch-type grain.
PITCHER_YEAR_KEYS = ("pitcher_id", "year")


# ---------- Spline helpers ----------

def fit_spline_transformers(ev: np.ndarray, la: np.ndarray):
    st_ev = SplineTransformer(
        n_knots=SPLINE_KNOTS, degree=SPLINE_DEGREE, knots="quantile",
        include_bias=False, extrapolation="constant",
    ).fit(ev.reshape(-1, 1))
    st_la = SplineTransformer(
        n_knots=SPLINE_KNOTS, degree=SPLINE_DEGREE, knots="quantile",
        include_bias=False, extrapolation="constant",
    ).fit(la.reshape(-1, 1))
    return st_ev, st_la


def design_matrix(bbe: pl.DataFrame, st_ev, st_la,
                   group_keys=PITCHER_YEAR_KEYS):
    """Per group: mean tensor-product B-spline activation."""
    ev = bbe["launch_speed"].to_numpy()
    la = bbe["launch_angle"].to_numpy()
    B_ev = st_ev.transform(ev.reshape(-1, 1)).astype(np.float64)
    B_la = st_la.transform(la.reshape(-1, 1)).astype(np.float64)
    K_ev, K_la = B_ev.shape[1], B_la.shape[1]
    df = bbe.with_row_index("row").sort(*group_keys)
    perm = df["row"].to_numpy()
    B_ev_s = B_ev[perm]; B_la_s = B_la[perm]
    grp = (df.group_by(list(group_keys), maintain_order=True)
             .agg(pl.len().alias("n_p"),
                  pl.col("xwobacon_next").first().alias("y_p"),
                  pl.col("n_bip_next").first().cast(pl.Float64).alias("w_p")))
    n_p = grp["n_p"].to_numpy().astype(np.int64)
    y_p = grp["y_p"].to_numpy().astype(np.float64)
    w_p = grp["w_p"].to_numpy().astype(np.float64)
    boundaries = np.concatenate([[0], np.cumsum(n_p)])
    M = np.zeros((len(n_p), K_ev * K_la), dtype=np.float64)
    for g in range(len(n_p)):
        lo, hi = boundaries[g], boundaries[g + 1]
        M[g] = (B_ev_s[lo:hi].T @ B_la_s[lo:hi]).ravel() / (hi - lo)
    return M, y_p, w_p, K_ev, K_la, grp


def diff2_op(K: int) -> np.ndarray:
    D = np.zeros((K - 2, K))
    for i in range(K - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def p_spline_penalty(K_ev: int, K_la: int) -> np.ndarray:
    """Anisotropic 2nd-difference penalty for flattened (K_ev × K_la) coefs."""
    D_ev = diff2_op(K_ev); D_la = diff2_op(K_la)
    return (np.kron(D_ev.T @ D_ev, np.eye(K_la))
            + np.kron(np.eye(K_ev), D_la.T @ D_la))


def weighted_ridge(X, y, w, alpha, P):
    """Sample-weighted ridge with generalized penalty α · β·P·β.

    Centering uses the weighted mean (not the sample mean of sqrt(w)·X) so
    the intercept is correct under heterogeneous weights.
    """
    K = X.shape[1]
    W = w / w.sum()
    x_bar = W @ X
    y_bar = float(W @ y)
    Xc = X - x_bar
    yc = y - y_bar
    XtWX = (Xc.T * w) @ Xc
    XtWy = (Xc.T * w) @ yc
    beta = np.linalg.solve(XtWX + alpha * P, XtWy)
    intercept = float(y_bar - x_bar @ beta)
    return beta, intercept


# ---------- LGBM helpers ----------

def make_grouped_lgbm(bbe: pl.DataFrame, group_keys=PITCHER_YEAR_KEYS):
    """Sort BBE by group_keys; return per-event + per-group arrays + grp frame."""
    df = bbe.sort(*group_keys)
    grp = (df.group_by(list(group_keys), maintain_order=True)
             .agg(pl.len().alias("n_p"),
                  pl.col("xwobacon_next").first().alias("y_p"),
                  pl.col("n_bip_next").first().cast(pl.Float64).alias("w_p")))
    n_p = grp["n_p"].to_numpy().astype(np.int64)
    y_p = grp["y_p"].to_numpy().astype(np.float64)
    w_p = grp["w_p"].to_numpy().astype(np.float64)
    group_starts = np.concatenate([[0], np.cumsum(n_p)])
    X = df.select("launch_speed", "launch_angle").to_numpy()
    y_row = np.repeat(y_p, n_p)
    return X, y_row, group_starts, n_p, y_p, w_p, grp


def lgbm_aggregated_objective(group_starts, n_p, y_p, w_p):
    """Group-aggregated MSE: g_e = 2 w_p (mean_p − y_p)/N_p, h_e = 2 w_p/N_p."""
    n_p_f = n_p.astype(np.float64)
    def obj(y_pred, dataset):
        sums = np.add.reduceat(y_pred, group_starts[:-1])
        means = sums / n_p_f
        resid = means - y_p
        g = np.repeat(2.0 * w_p * resid / n_p_f, n_p)
        h = np.repeat(2.0 * w_p / n_p_f, n_p)
        return g, h
    return obj


# ---------- Train + predict (per-group, closed-form) ----------

def train_splines(train: pl.DataFrame, test: pl.DataFrame,
                   group_keys=PITCHER_YEAR_KEYS):
    st_ev, st_la = fit_spline_transformers(
        train["launch_speed"].to_numpy(), train["launch_angle"].to_numpy())
    X_tr, y_tr, w_tr, K_ev, K_la, _      = design_matrix(train, st_ev, st_la, group_keys)
    X_te, y_te, w_te, _,    _,    grp_te = design_matrix(test,  st_ev, st_la, group_keys)
    P = p_spline_penalty(K_ev, K_la)
    beta, intercept = weighted_ridge(X_tr, y_tr, w_tr, SPLINE_ALPHA, P)
    pred = X_te @ beta + intercept
    return {
        "pred_te": pred, "y_te": y_te, "w_te": w_te, "grp_te": grp_te,
        "beta": beta, "intercept": intercept, "K_ev": K_ev, "K_la": K_la,
        "st_ev": st_ev, "st_la": st_la,
    }


def train_lgbm(train: pl.DataFrame, test: pl.DataFrame,
                group_keys=PITCHER_YEAR_KEYS):
    X_tr, yrow_tr, gs_tr, np_tr, yp_tr, wp_tr, _      = make_grouped_lgbm(train, group_keys)
    X_te, _,       gs_te, np_te, yp_te, wp_te, grp_te = make_grouped_lgbm(test,  group_keys)
    init_score = float((wp_tr * yp_tr).sum() / wp_tr.sum())
    obj = lgbm_aggregated_objective(gs_tr, np_tr, yp_tr, wp_tr)
    dtr = lgb.Dataset(X_tr, label=yrow_tr,
                       init_score=np.full(X_tr.shape[0], init_score),
                       free_raw_data=False)
    booster = lgb.train(
        params={"objective": obj, "num_leaves": 5, "min_data_in_leaf": 1000,
                "verbose": -1, "force_row_wise": True},
        train_set=dtr, num_boost_round=LGBM_NUM_BOOST_ROUND,
        callbacks=[lgb.log_evaluation(0)],
    )
    pred_event = booster.predict(X_te) + init_score
    sums = np.add.reduceat(pred_event, gs_te[:-1])
    pred = sums / np_te.astype(np.float64)
    return {
        "pred_te": pred, "y_te": yp_te, "w_te": wp_te, "grp_te": grp_te,
        "booster": booster, "init_score": init_score,
    }


# ---------- Cached + smoothed ensemble grid (the production model) ----------

def _component_grids(s, l):
    """Evaluate spline and LGBM surfaces on the shared (EV, LA) grid."""
    ev_grid = np.linspace(0.0, 120.0, GRID_N_EV)
    la_grid = np.linspace(-90.0, 90.0, GRID_N_LA)
    B_ev = s["st_ev"].transform(ev_grid.reshape(-1, 1))
    B_la = s["st_la"].transform(la_grid.reshape(-1, 1))
    beta_mat = s["beta"].reshape(s["K_ev"], s["K_la"])
    spline_grid = s["intercept"] + B_ev @ beta_mat @ B_la.T
    EV, LA = np.meshgrid(ev_grid, la_grid, indexing="ij")
    lgbm_grid = (l["booster"].predict(np.column_stack([EV.ravel(), LA.ravel()]))
                  + l["init_score"]).reshape(EV.shape)
    return ev_grid, la_grid, spline_grid, lgbm_grid


def build_smoothed_ensemble_grid(s, l,
                                   sigma_ev: float = SMOOTH_SIGMA_EV,
                                   sigma_la: float = SMOOTH_SIGMA_LA):
    """0.5*spline + 0.5*LGBM on (EV, LA), then Gauss-smoothed.

    Returns (ev_grid, la_grid, smoothed_grid).
    """
    ev_grid, la_grid, spline_grid, lgbm_grid = _component_grids(s, l)
    ens_grid = 0.5 * spline_grid + 0.5 * lgbm_grid
    dev = ev_grid[1] - ev_grid[0]
    dla = la_grid[1] - la_grid[0]
    smoothed = gaussian_filter(ens_grid,
                                sigma=(sigma_ev / dev, sigma_la / dla),
                                mode="nearest")
    return ev_grid, la_grid, smoothed


def grid_predict_per_bip(grid: np.ndarray, ev_grid: np.ndarray,
                          la_grid: np.ndarray,
                          evs: np.ndarray, las: np.ndarray) -> np.ndarray:
    """Bilinear-interpolate `grid` at per-BIP (EV, LA) coordinates."""
    interp = RegularGridInterpolator((ev_grid, la_grid), grid,
                                       bounds_error=False, fill_value=None)
    evs_c = np.clip(evs, ev_grid[0], ev_grid[-1])
    las_c = np.clip(las, la_grid[0], la_grid[-1])
    return interp(np.column_stack([evs_c, las_c]))


def grid_predict_per_group(bbe: pl.DataFrame, grid: np.ndarray,
                            ev_grid: np.ndarray, la_grid: np.ndarray,
                            group_keys, grp_te: pl.DataFrame) -> np.ndarray:
    """Per-BIP `grid` lookup, mean per `group_keys`, aligned to `grp_te`."""
    p_e = grid_predict_per_bip(grid, ev_grid, la_grid,
                                bbe["launch_speed"].to_numpy(),
                                bbe["launch_angle"].to_numpy())
    df = bbe.with_columns(pl.Series("p_e", p_e))
    agg = df.group_by(list(group_keys)).agg(pl.col("p_e").mean().alias("p"))
    return (grp_te.select(list(group_keys))
                   .join(agg, on=list(group_keys), how="left")["p"].to_numpy())


# ---------- Heatmap ----------

def render_heatmap(out_path: Path, ev_grid: np.ndarray, la_grid: np.ndarray,
                    grid: np.ndarray, title: str,
                    density_bbe: pl.DataFrame | None = None) -> None:
    """Heatmap of `grid` on (EV, LA), masked by BBE density when provided."""
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.ticker import MultipleLocator

    EV, LA = np.meshgrid(ev_grid, la_grid, indexing="ij")
    if density_bbe is not None:
        H, xe, ye = np.histogram2d(
            density_bbe["launch_speed"].to_numpy(),
            density_bbe["launch_angle"].to_numpy(),
            bins=[120, 120], range=[[0, 120], [-90, 90]],
        )
        ix = np.clip(np.searchsorted(xe, ev_grid, side="right") - 1, 0, H.shape[0] - 1)
        iy = np.clip(np.searchsorted(ye, la_grid, side="right") - 1, 0, H.shape[1] - 1)
        Hgrid = H[np.ix_(ix, iy)]
        mask = Hgrid < max(1, H.sum() * 1e-6)
        grid_m = np.ma.array(grid, mask=mask)
    else:
        grid_m = grid

    fig, ax = plt.subplots(figsize=(8, 6.3), constrained_layout=True)
    league_xwobacon = 0.371
    half = 0.18
    norm = mcolors.TwoSlopeNorm(vmin=league_xwobacon - half,
                                 vcenter=league_xwobacon,
                                 vmax=league_xwobacon + half)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#dddddd")
    im = ax.pcolormesh(EV, LA, grid_m, shading="auto", cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("predicted next-year pitcher xwOBAcon", fontsize=14)
    cbar.ax.yaxis.set_major_locator(MultipleLocator(0.05))
    cbar.ax.tick_params(labelsize=12)
    ax.set_xlabel("Exit velocity (mph)", fontsize=15)
    ax.set_ylabel("Launch angle (°)", fontsize=15)
    ax.set_title(title, fontsize=16, pad=10)
    ax.set_xlim(0, 120); ax.set_ylim(-90, 90)
    ax.tick_params(labelsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------- Main ----------

def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ART.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MIN_BIP_TRAINVAL", "50")
    # Reimport eval after env-var seed.
    import importlib, eval as eval_mod
    importlib.reload(eval_mod)
    from eval import load_splits

    t0 = time.time()
    train, val, test, _ = load_splits()
    train_full = pl.concat([train, val])
    print(f"trainval BBE: {train_full.height:,}, test BBE: {test.height:,}")
    print(f"trainval groups: {train_full.unique(['pitcher_id', 'year']).height}, "
          f"test groups: {test.unique(['pitcher_id', 'year']).height}")
    print(f"MIN_BIP_TRAINVAL={os.environ['MIN_BIP_TRAINVAL']}")
    print()

    print(f"=== Splines (K={SPLINE_KNOTS}×{SPLINE_KNOTS}, deg={SPLINE_DEGREE}, "
          f"P-spline α={SPLINE_ALPHA:g}) ===")
    t = time.time()
    s = train_splines(train_full, test)
    rmse_s = weighted_rmse(s["y_te"], s["pred_te"], s["w_te"])
    r2_s = weighted_r2(s["y_te"], s["pred_te"], s["w_te"])
    print(f"  METRIC splines_test_rmse={rmse_s:.5f}")
    print(f"  METRIC splines_test_r2={r2_s:.5f}")
    print(f"  ({time.time()-t:.0f}s)")

    print(f"\n=== LGBM (defaults, {LGBM_NUM_BOOST_ROUND} rounds) ===")
    t = time.time()
    l = train_lgbm(train_full, test)
    rmse_l = weighted_rmse(l["y_te"], l["pred_te"], l["w_te"])
    r2_l = weighted_r2(l["y_te"], l["pred_te"], l["w_te"])
    print(f"  METRIC lgbm_test_rmse={rmse_l:.5f}")
    print(f"  METRIC lgbm_test_r2={r2_l:.5f}")
    print(f"  ({time.time()-t:.0f}s)")

    assert np.allclose(s["y_te"], l["y_te"])
    assert np.allclose(s["w_te"], l["w_te"])

    print(f"\n=== Ensemble = Gauss(0.5*spline + 0.5*LGBM), "
          f"σ=({SMOOTH_SIGMA_EV:g} mph, {SMOOTH_SIGMA_LA:g}°) on "
          f"{GRID_N_EV}×{GRID_N_LA} grid ===")
    t = time.time()
    ev_grid, la_grid, smoothed_grid = build_smoothed_ensemble_grid(s, l)
    pred_ens = grid_predict_per_group(test, smoothed_grid, ev_grid, la_grid,
                                        PITCHER_YEAR_KEYS, s["grp_te"])
    rmse_e = weighted_rmse(s["y_te"], pred_ens, s["w_te"])
    r2_e = weighted_r2(s["y_te"], pred_ens, s["w_te"])
    print(f"  METRIC ensemble_test_rmse={rmse_e:.5f}")
    print(f"  METRIC ensemble_test_r2={r2_e:.5f}")
    print(f"  ({time.time()-t:.0f}s)")

    print(f"\n  Splines:  {rmse_s:.5f}, r²={r2_s:+.4f}")
    print(f"  LGBM:     {rmse_l:.5f}, r²={r2_l:+.4f}")
    print(f"  Ensemble: {rmse_e:.5f}, r²={r2_e:+.4f}")

    # Compute the un-smoothed component grids — needed both for heatmaps and
    # for downstream consumers (src/leaderboard.py) that score GAM and LGBM
    # rows from cached grids instead of retraining.
    _, _, spline_grid, lgbm_grid = _component_grids(s, l)

    # Save the production model: smoothed ensemble + un-smoothed component
    # grids on the shared (EV, LA) axes.
    np.savez(ART / "ensemble_grid.npz",
             ev_grid=ev_grid, la_grid=la_grid,
             grid=smoothed_grid,
             spline_grid=spline_grid, lgbm_grid=lgbm_grid,
             sigma_ev=SMOOTH_SIGMA_EV, sigma_la=SMOOTH_SIGMA_LA)
    print(f"\nsaved ensemble_grid.npz")

    # Heatmaps: ensemble (= the smoothed model) + raw components.
    density_bbe = load_batted_balls()

    render_heatmap(
        ART / "heatmap_ensemble.png", ev_grid, la_grid, smoothed_grid,
        "Pitcher Predictive xwOBAcon (Ensemble)",
        density_bbe=density_bbe,
    )
    print("saved heatmap_ensemble.png")
    render_heatmap(
        ART / "heatmap_gam.png", ev_grid, la_grid, spline_grid,
        "Pitcher Predictive xwOBAcon (GAM)",
        density_bbe=density_bbe,
    )
    print("saved heatmap_gam.png")
    render_heatmap(
        ART / "heatmap_lgbm.png", ev_grid, la_grid, lgbm_grid,
        "Pitcher Predictive xwOBAcon (LGBM)",
        density_bbe=density_bbe,
    )
    print("saved heatmap_lgbm.png")

    print(f"\ntotal: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
