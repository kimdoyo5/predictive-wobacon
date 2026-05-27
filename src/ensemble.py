"""Final pitcher-year predictive-xwOBAcon ensemble.

Two component models trained on per-event (EV, LA) with a custom group-
aggregated MSE loss, grouped by (pitcher_id, year):

  1. GAM splines: 50×50 degree-1 B-splines, P-spline penalty α=1000.
  2. LGBM: 3000 rounds, num_leaves=5, min_data_in_leaf=1000.

Output: equal-weight 50/50 average of the two predictions.

Filters and weights: see eval.load_splits (this module overrides
MIN_BIP_TRAINVAL=50 for training; trainval = train ∪ val with no val
held out).

Usage:
  uv run python src/ensemble.py   # train + evaluate + render heatmaps
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.preprocessing import SplineTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data import load_batted_balls
from eval import load_splits, weighted_rmse, weighted_r2

ART = ROOT / "artifacts"

# Hyperparams (final).
SPLINE_KNOTS  = 50
SPLINE_DEGREE = 1
SPLINE_ALPHA  = 1000.0
LGBM_NUM_BOOST_ROUND = 3000

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


# ---------- Train + predict ----------

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


# ---------- Heatmap ----------

def render_heatmap(spline_res, lgbm_res, out_path: Path, which: str = "ensemble"):
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.ticker import MultipleLocator

    ev_grid = np.linspace(0, 120, 240)
    la_grid = np.linspace(-90, 90, 240)
    # Spline predictions on grid
    B_ev = spline_res["st_ev"].transform(ev_grid.reshape(-1, 1))
    B_la = spline_res["st_la"].transform(la_grid.reshape(-1, 1))
    beta_mat = spline_res["beta"].reshape(spline_res["K_ev"], spline_res["K_la"])
    pred_s = spline_res["intercept"] + B_ev @ beta_mat @ B_la.T
    # LGBM predictions on grid
    EV, LA = np.meshgrid(ev_grid, la_grid, indexing="ij")
    Xg = np.column_stack([EV.ravel(), LA.ravel()])
    pred_l = (lgbm_res["booster"].predict(Xg) + lgbm_res["init_score"]).reshape(EV.shape)

    if which == "splines":
        pred = pred_s
        title_model = f"GAM splines (K=50, deg=1, P-spline α={SPLINE_ALPHA:g})"
    elif which == "lgbm":
        pred = pred_l
        title_model = f"LightGBM (defaults, {LGBM_NUM_BOOST_ROUND} rounds)"
    else:
        pred = 0.5 * pred_s + 0.5 * pred_l
        title_model = (f"50/50 ensemble (Splines + LGBM)\n"
                       f"Splines: K=50, deg=1, P-spline α={SPLINE_ALPHA:g}    "
                       f"LGBM: defaults, {LGBM_NUM_BOOST_ROUND} rounds")

    # Density mask
    bb = load_batted_balls()
    H, xe, ye = np.histogram2d(
        bb["launch_speed"].to_numpy(), bb["launch_angle"].to_numpy(),
        bins=[120, 120], range=[[0, 120], [-90, 90]],
    )
    ix = np.clip(np.searchsorted(xe, ev_grid, side="right") - 1, 0, H.shape[0] - 1)
    iy = np.clip(np.searchsorted(ye, la_grid, side="right") - 1, 0, H.shape[1] - 1)
    Hgrid = H[np.ix_(ix, iy)]
    mask = Hgrid < max(1, H.sum() * 1e-6)
    pred_m = np.ma.array(pred, mask=mask)

    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    league_xwobacon = 0.371
    half_range = 0.2
    vmin = league_xwobacon - half_range
    vmax = league_xwobacon + half_range
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=league_xwobacon, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#dddddd")
    im = ax.pcolormesh(EV, LA, pred_m, shading="auto", cmap=cmap, norm=norm)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("predicted next-year pitcher xwOBAcon")
    cbar.ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_xlabel("Exit velocity (mph)")
    ax.set_ylabel("Launch angle (°)")
    ax.set_title(f"Pitcher-year predictive xwOBAcon: {title_model}")
    ax.set_xlim(0, 120); ax.set_ylim(-90, 90)
    fig.savefig(out_path, dpi=150)


# ---------- Main ----------

def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ART.mkdir(parents=True, exist_ok=True)

    # Default to BIP=50 if not set.
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

    # Sanity: same target alignment.
    assert np.allclose(s["y_te"], l["y_te"])
    assert np.allclose(s["w_te"], l["w_te"])

    pred_ens = 0.5 * s["pred_te"] + 0.5 * l["pred_te"]
    rmse_e = weighted_rmse(s["y_te"], pred_ens, s["w_te"])
    r2_e = weighted_r2(s["y_te"], pred_ens, s["w_te"])
    print(f"\n=== 50/50 Ensemble ===")
    print(f"  METRIC ensemble_test_rmse={rmse_e:.5f}")
    print(f"  METRIC ensemble_test_r2={r2_e:.5f}")
    print(f"\n  Splines:  {rmse_s:.5f}, r²={r2_s:+.4f}")
    print(f"  LGBM:     {rmse_l:.5f}, r²={r2_l:+.4f}")
    print(f"  Ensemble: {rmse_e:.5f}, r²={r2_e:+.4f}")

    # Save model artifacts.
    np.savez(ART / "ensemble_splines.npz",
             beta=s["beta"].reshape(s["K_ev"], s["K_la"]),
             intercept=s["intercept"])
    l["booster"].save_model(str(ART / "ensemble_lgbm.txt"))
    np.savez(ART / "ensemble_lgbm_meta.npz", init_score=l["init_score"])
    print(f"\nsaved ensemble_splines.npz, ensemble_lgbm.txt, ensemble_lgbm_meta.npz")

    for which, fname in [("ensemble", "heatmap_ensemble.png"),
                          ("splines",  "heatmap_gam.png"),
                          ("lgbm",     "heatmap_lgbm.png")]:
        out = ART / fname
        render_heatmap(s, l, out, which=which)
        print(f"saved {out.name}")

    print(f"\ntotal: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
