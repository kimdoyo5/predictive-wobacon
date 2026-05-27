"""Leaderboard at pitcher-year (default) or pitcher × pitch_type × year grain.

Columns (BIP-weighted on the held-out 2024 → 2025 test set):
  rmse       — RMSE on year-N+1 xwobacon prediction
  r²         — corr²(predictor, target); the linear-calibration-free ceiling
               (== r²(self) by construction for univariate predictors equal to the target).
  r² (self)  — corr²(predictor_n, predictor_n+1); year-to-year stability.

Rows: ensemble (50/50 splines + LGBM), gam, lgbm, tango, xwobacon, avg_ev,
      avg_la, hr_rate.

Usage:
  uv run python src/leaderboard.py    # runs both grains, writes both artifacts
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MIN_BIP_TRAINVAL", "50")
import importlib, eval as eval_mod; importlib.reload(eval_mod)
from eval import load_splits, weighted_rmse, weighted_corr
from ensemble import train_splines, train_lgbm

RAW = ROOT / "data" / "raw"

# Pitch-type grain config.
PT_MIN_BIP = 20
PT_TRAINVAL_YEARS = tuple(range(2016, 2024))
PT_TEST_YEAR = 2024

METRICS = ["xwobacon", "avg_ev", "avg_la", "hr_rate"]


# --- Tango 12-bucket OLS (3 LA × 4 EV) ---

LA_BREAKS = [8.0, 32.0]
EV_BREAKS = [95.0, 100.0, 105.0]
N_LA = len(LA_BREAKS) + 1
N_EV = len(EV_BREAKS) + 1
N_BUCKETS = N_LA * N_EV


def bucket_id(ev, la):
    iev = np.searchsorted(EV_BREAKS, ev, side="right")
    ila = np.searchsorted(LA_BREAKS, la, side="right")
    return ila * N_EV + iev


def tango_design(bbe: pl.DataFrame, group_keys):
    """Per group: bucket-frequency vector + key. Targets (y, w) attached if
    `xwobacon_next` and `n_bip_next` columns are present on `bbe`, else NaN.
    """
    bb = bbe.with_columns(
        pl.Series("bucket", bucket_id(bbe["launch_speed"].to_numpy(),
                                       bbe["launch_angle"].to_numpy()))
    )
    counts = (bb.group_by([*group_keys, "bucket"]).len()
                .pivot(on="bucket", index=list(group_keys), values="len",
                       aggregate_function="first")
                .fill_null(0))
    for k in range(N_BUCKETS):
        if str(k) not in counts.columns:
            counts = counts.with_columns(pl.lit(0, dtype=pl.UInt32).alias(str(k)))
    ordered = [str(k) for k in range(N_BUCKETS)]
    counts = counts.select([*group_keys, *ordered])
    Xc = counts.select(ordered).to_numpy().astype(np.float64)
    n_bip = Xc.sum(axis=1, keepdims=True)
    X = Xc / np.maximum(n_bip, 1)
    key = counts.select(list(group_keys))
    if "xwobacon_next" in bb.columns and "n_bip_next" in bb.columns:
        nxt = (bb.select([*group_keys, "xwobacon_next", "n_bip_next"])
                  .unique(subset=list(group_keys)))
        keyed = key.join(nxt, on=list(group_keys), how="left")
        y = keyed["xwobacon_next"].to_numpy()
        w = keyed["n_bip_next"].to_numpy().astype(np.float64)
    else:
        y = np.full(X.shape[0], np.nan)
        w = np.full(X.shape[0], np.nan)
    return X, y, w, key


def tango_fit(train_full: pl.DataFrame, group_keys) -> np.ndarray:
    X_tr, y_tr, w_tr, _ = tango_design(train_full, group_keys)
    sw = np.sqrt(w_tr)
    beta, *_ = np.linalg.lstsq(X_tr * sw[:, None], y_tr * sw, rcond=None)
    return beta


# --- Univariate per-season metrics ---

def season_metrics(bbe: pl.DataFrame, group_keys) -> pl.DataFrame:
    """Per group: n_bip, xwobacon, avg_ev, avg_la, hr_rate."""
    return bbe.group_by(list(group_keys)).agg(
        pl.len().alias("n_bip"),
        pl.col("xwoba_value").mean().alias("xwobacon"),
        pl.col("launch_speed").mean().alias("avg_ev"),
        pl.col("launch_angle").mean().alias("avg_la"),
        (pl.col("event_type") == "home_run").mean().alias("hr_rate"),
    ).sort(*group_keys)


def fit_univariate(x_tr, y_tr, w_tr) -> tuple[float, float]:
    """Weighted univariate OLS with intercept."""
    W = w_tr.sum()
    xbar = float((w_tr * x_tr).sum() / W)
    ybar = float((w_tr * y_tr).sum() / W)
    cov = float((w_tr * (x_tr - xbar) * (y_tr - ybar)).sum() / W)
    var = float((w_tr * (x_tr - xbar) ** 2).sum() / W)
    b = cov / var if var > 0 else 0.0
    a = ybar - b * xbar
    return a, b


# --- Per-grain data loaders ---

def load_pitcher_grain():
    """Pitcher-year grain: defers to eval.load_splits()."""
    train, val, test, test_next = load_splits()
    train_full = pl.concat([train, val])
    group_keys = ("pitcher_id", "year")
    self_keys  = ("pitcher_id",)   # year-N+1 alignment drops the year
    header_tmpl = ("Pitcher-year test leaderboard "
                   "(n={n} pitchers, IP≥30 both yrs, min(IP, IP_next) weighted)")
    return train_full, test, test_next, group_keys, self_keys, header_tmpl


def load_pitch_type_grain():
    """Pitch-type grain: load all BBE with pitch_type, build (N, N+1) pairs.

    Both train and test filter on min(n_bip, n_bip_next) >= PT_MIN_BIP per
    (pitcher, pitch_type). Weight (in the `n_bip_next` column) is
    min(n_bip, n_bip_next).
    """
    frames = []
    for y in range(2016, 2026):
        frames.append(
            pl.scan_parquet(RAW / f"pitches_{y}.parquet")
            .select("pitcher_id", "year", "pitch_type",
                    "launch_speed", "launch_angle", "event_type", "xwoba_value")
            .filter(
                pl.col("launch_speed").is_not_null()
                & pl.col("launch_angle").is_not_null()
                & pl.col("xwoba_value").is_not_null()
                & pl.col("pitch_type").is_not_null()
            )
        )
    bb = pl.concat(frames, how="vertical").collect()

    pst = season_metrics(bb, ("pitcher_id", "pitch_type", "year")) \
            .select("pitcher_id", "pitch_type", "year", "n_bip", "xwobacon")
    nxt = (pst.with_columns((pl.col("year") - 1).alias("year"))
              .rename({"n_bip": "n_bip_next", "xwobacon": "xwobacon_next"}))
    pairs = pst.join(nxt, on=["pitcher_id", "pitch_type", "year"], how="inner")

    bip_min_ok = pl.min_horizontal("n_bip", "n_bip_next") >= PT_MIN_BIP
    trainval  = pairs.filter(pl.col("year").is_in(list(PT_TRAINVAL_YEARS)) & bip_min_ok)
    test_keys = pairs.filter((pl.col("year") == PT_TEST_YEAR) & bip_min_ok)

    target_cols = pairs.select(
        "pitcher_id", "pitch_type", "year", "xwobacon_next",
        pl.min_horizontal("n_bip", "n_bip_next").alias("n_bip_next"),
    )
    bbe_all = bb.join(target_cols, on=["pitcher_id", "pitch_type", "year"], how="inner")
    train_full = bbe_all.join(trainval.select("pitcher_id", "pitch_type", "year"),
                               on=["pitcher_id", "pitch_type", "year"], how="inner")
    test = bbe_all.join(test_keys.select("pitcher_id", "pitch_type", "year"),
                         on=["pitcher_id", "pitch_type", "year"], how="inner")
    test_next = (bb.filter(pl.col("year") == PT_TEST_YEAR + 1)
                   .join(test_keys.select("pitcher_id", "pitch_type"),
                         on=["pitcher_id", "pitch_type"], how="inner"))

    group_keys = ("pitcher_id", "pitch_type", "year")
    self_keys  = ("pitcher_id", "pitch_type")
    header_tmpl = ("Pitcher × pitch_type × year test leaderboard "
                   f"(n={{n}} (pitcher, pitch_type) combos, "
                   f"min(n_bip, n_bip_next)≥{PT_MIN_BIP}, min(n_bip, n_bip_next) weighted)")
    return train_full, test, test_next, group_keys, self_keys, header_tmpl


# --- Main ---

def run_grain(loader, out_filename: str) -> None:
    train_full, test, test_next, group_keys, self_keys, header_tmpl = loader()

    # --- Train models ---
    print("Training splines + LGBM ...", file=sys.stderr, flush=True)
    s = train_splines(train_full, test, group_keys=group_keys)
    l = train_lgbm(train_full, test, group_keys=group_keys)
    assert np.allclose(s["y_te"], l["y_te"])
    assert np.allclose(s["w_te"], l["w_te"])

    y_te   = s["y_te"]
    w_te   = s["w_te"]
    grp_te = s["grp_te"]   # has group_keys columns + n_p/y_p/w_p
    pred_gam  = s["pred_te"]
    pred_lgbm = l["pred_te"]
    pred_ens  = 0.5 * pred_gam + 0.5 * pred_lgbm

    # Year-N+1 predictions for r²(self): apply f to year-N+1 BBE, mean per
    # self_keys group, aligned to grp_te.
    def model_pred_n1(bbe: pl.DataFrame, kind: str) -> np.ndarray:
        evs = bbe["launch_speed"].to_numpy()
        las = bbe["launch_angle"].to_numpy()
        if kind == "splines":
            B_ev = s["st_ev"].transform(evs.reshape(-1, 1))
            B_la = s["st_la"].transform(las.reshape(-1, 1))
            beta_mat = s["beta"].reshape(s["K_ev"], s["K_la"])
            pred_e = s["intercept"] + (B_ev @ beta_mat * B_la).sum(axis=1)
        else:
            X_e = np.column_stack([evs, las])
            pred_e = l["booster"].predict(X_e) + l["init_score"]
        df = bbe.with_columns(pl.Series("pred_e", pred_e))
        agg = df.group_by(list(self_keys)).agg(pl.col("pred_e").mean().alias("p"))
        keyed = grp_te.select(list(self_keys)).join(agg, on=list(self_keys), how="left")
        return keyed["p"].to_numpy()

    pred_gam_n1  = model_pred_n1(test_next, "splines")
    pred_lgbm_n1 = model_pred_n1(test_next, "lgbm")
    pred_ens_n1  = 0.5 * pred_gam_n1 + 0.5 * pred_lgbm_n1

    rows: list[dict] = []
    def add(name, pred_n, pred_n1):
        ok = np.isfinite(pred_n) & np.isfinite(pred_n1)
        rmse  = weighted_rmse(y_te[ok], pred_n[ok], w_te[ok])
        r2    = weighted_corr(pred_n[ok], y_te[ok], w_te[ok]) ** 2
        rself = weighted_corr(pred_n[ok], pred_n1[ok], w_te[ok]) ** 2
        rows.append({"name": name, "rmse": rmse, "r2": r2, "r2_self": rself})

    add("ensemble", pred_ens, pred_ens_n1)
    add("gam",      pred_gam, pred_gam_n1)
    add("lgbm",     pred_lgbm, pred_lgbm_n1)

    # --- Tango ---
    beta_tango = tango_fit(train_full, group_keys)
    X_n,  _, _, key_n  = tango_design(test,      group_keys)
    X_n1, _, _, key_n1 = tango_design(test_next, group_keys)

    def to_key_tuples(df: pl.DataFrame, cols) -> list[tuple]:
        return list(zip(*[df[c].to_list() for c in cols]))

    pred_n_map  = dict(zip(to_key_tuples(key_n,  group_keys), (X_n  @ beta_tango).tolist()))
    pred_n1_map = dict(zip(to_key_tuples(key_n1, self_keys),  (X_n1 @ beta_tango).tolist()))
    keys_te_n   = to_key_tuples(grp_te, group_keys)
    keys_te_n1  = to_key_tuples(grp_te, self_keys)
    p_n  = np.array([pred_n_map.get(k,  np.nan) for k in keys_te_n])
    p_n1 = np.array([pred_n1_map.get(k, np.nan) for k in keys_te_n1])
    add("tango", p_n, p_n1)

    # --- Univariate linregs on per-season metrics ---
    pst_trv = season_metrics(train_full, group_keys)
    nxt_target = (train_full.unique(list(group_keys))
                            .select([*group_keys, "xwobacon_next", "n_bip_next"]))
    pst_trv = pst_trv.join(nxt_target, on=list(group_keys), how="inner")
    y_trv = pst_trv["xwobacon_next"].to_numpy()
    w_trv = pst_trv["n_bip_next"].to_numpy().astype(np.float64)

    pst_te    = season_metrics(test,      group_keys)
    pst_te_n1 = season_metrics(test_next, self_keys)  # drop "year" for alignment
    paired_n = grp_te.select(list(group_keys)).join(
        pst_te.select([*group_keys, *METRICS]),
        on=list(group_keys), how="left",
    )
    paired_n1 = grp_te.select(list(self_keys)).join(
        pst_te_n1.select([*self_keys, *METRICS]),
        on=list(self_keys), how="left",
    )

    for metric in METRICS:
        x_trv = pst_trv[metric].to_numpy()
        a, b = fit_univariate(x_trv, y_trv, w_trv)
        x_n  = paired_n[metric].to_numpy()
        x_n1 = paired_n1[metric].to_numpy()
        yhat_n = a + b * x_n
        ok = np.isfinite(yhat_n) & np.isfinite(x_n1)
        rmse  = weighted_rmse(y_te[ok], yhat_n[ok], w_te[ok])
        r2    = weighted_corr(x_n[ok], y_te[ok], w_te[ok]) ** 2
        rself = weighted_corr(x_n[ok], x_n1[ok], w_te[ok]) ** 2
        rows.append({"name": metric, "rmse": rmse, "r2": r2, "r2_self": rself})

    # --- Print + write artifact ---
    rows.sort(key=lambda r: r["rmse"])
    lines = [
        "",
        header_tmpl.format(n=len(y_te)),
        f"{'rank':>4}  {'name':<12s}  {'rmse':>8s}  {'r²':>8s}  {'r² (self)':>10s}",
        "-" * 50,
    ]
    for i, r in enumerate(rows, 1):
        lines.append(f"{i:>4}  {r['name']:<12s}  {r['rmse']:.5f}  {r['r2']:+.4f}  {r['r2_self']:+.6f}")
    out = "\n".join(lines) + "\n"
    print(out, end="")

    art = ROOT / "artifacts"
    art.mkdir(exist_ok=True)
    (art / out_filename).write_text(out, encoding="utf-8")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    run_grain(load_pitcher_grain,    "leaderboard_pitcher_year.txt")
    run_grain(load_pitch_type_grain, "leaderboard_pitch_type.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
