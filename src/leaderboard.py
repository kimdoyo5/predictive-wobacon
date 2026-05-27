"""Leaderboard at pitcher-year (default) or pitcher × pitch_type × year grain.

Columns (BIP-weighted on the held-out 2024 → 2025 test set):
  rmse       — RMSE on year-N+1 xwobacon prediction
  r²         — corr²(predictor, target); the linear-calibration-free ceiling
               (== r²(self) by construction for univariate predictors equal to the target).
  r² (self)  — corr²(predictor_n, predictor_n+1); year-to-year stability.

Rows: ensemble (50/50 splines + LGBM), gam, lgbm, tango, xwobacon, avg_ev,
      avg_la, hr_rate, naive (constant training-mean target).

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
from eval import load_splits, weighted_rmse, weighted_corr, TEST_N_YEAR
from ensemble import train_splines, train_lgbm
from data import load_batted_balls, pitcher_season_ip

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

def load_pitcher_grain(min_ip: int = 30):
    """Pitcher-year grain: defers to eval.load_splits().

    `test_alpha` is the unfiltered test-year BBE slice — all pitchers, no IP
    threshold, no pair requirement — used for the Cronbach's α plot.
    """
    train, val, test, test_next = load_splits(min_ip=min_ip)
    train_full = pl.concat([train, val])
    test_alpha = load_batted_balls().filter(pl.col("year") == TEST_N_YEAR)
    group_keys = ("pitcher_id", "year")
    self_keys  = ("pitcher_id",)   # year-N+1 alignment drops the year
    header_tmpl = ("Pitcher-year test leaderboard "
                   f"(n={{n}} pitchers, IP≥{min_ip} both yrs, min(IP, IP_next) weighted)")
    return train_full, test, test_next, group_keys, self_keys, header_tmpl, test_alpha


def load_pitch_type_grain(min_bip: int = PT_MIN_BIP):
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

    bip_min_ok_trainval = pl.min_horizontal("n_bip", "n_bip_next") >= PT_MIN_BIP
    bip_min_ok_test     = pl.min_horizontal("n_bip", "n_bip_next") >= min_bip
    trainval  = pairs.filter(pl.col("year").is_in(list(PT_TRAINVAL_YEARS)) & bip_min_ok_trainval)
    test_keys = pairs.filter((pl.col("year") == PT_TEST_YEAR) & bip_min_ok_test)

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
    test_alpha = bb.filter(pl.col("year") == PT_TEST_YEAR)

    group_keys = ("pitcher_id", "pitch_type", "year")
    self_keys  = ("pitcher_id", "pitch_type")
    header_tmpl = ("Pitcher × pitch_type × year test leaderboard "
                   f"(n={{n}} (pitcher, pitch_type) combos, "
                   f"min(n_bip, n_bip_next)≥{min_bip}, min(n_bip, n_bip_next) weighted)")
    return train_full, test, test_next, group_keys, self_keys, header_tmpl, test_alpha


# --- Cronbach's α curves ---

def per_bip_predictors(test: pl.DataFrame, s, l, beta_tango) -> dict[str, np.ndarray]:
    """For each leaderboard model, predict at the BIP level on `test`."""
    evs = test["launch_speed"].to_numpy()
    las = test["launch_angle"].to_numpy()

    B_ev = s["st_ev"].transform(evs.reshape(-1, 1))
    B_la = s["st_la"].transform(las.reshape(-1, 1))
    beta_mat = s["beta"].reshape(s["K_ev"], s["K_la"])
    p_gam = s["intercept"] + (B_ev @ beta_mat * B_la).sum(axis=1)
    p_lgbm = l["booster"].predict(np.column_stack([evs, las])) + l["init_score"]

    return {
        "ensemble": 0.5 * p_gam + 0.5 * p_lgbm,
        "gam":      p_gam,
        "lgbm":     p_lgbm,
        "tango":    beta_tango[bucket_id(evs, las)],
        "xwobacon": test["xwoba_value"].to_numpy(),
        "avg_ev":   evs,
        "avg_la":   las,
        "hr_rate":  (test["event_type"] == "home_run").to_numpy().astype(np.float64),
    }


def alpha_curve(per: pl.DataFrame, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Spearman-Brown reliability α(n) from per-group (n, mean, var).

    σ²_e: mean of per-group within-variances (each group weighted equally).
    σ²_t: max(Var(group_means) − mean(σ²_e / n_p), 0).
    α(n) = n σ²_t / (n σ²_t + σ²_e).
    """
    n   = per["n"].to_numpy()
    mu  = per["mean"].to_numpy()
    var = per["var"].to_numpy()  # null where n=1

    mask = (n >= 2) & np.isfinite(var)
    sigma2_e = float(var[mask].mean()) if mask.any() else 0.0

    var_means = float(np.var(mu, ddof=1)) if len(mu) >= 2 else 0.0
    sigma2_t  = max(var_means - float((sigma2_e / n).mean()), 0.0)

    ns = np.arange(1, n_max + 1)
    alphas = ns * sigma2_t / (ns * sigma2_t + sigma2_e) if sigma2_e > 0 else np.ones_like(ns, dtype=float)
    return ns, alphas


PITCHER_NAME_CACHE = ROOT / "data" / "raw" / "pitcher_names.parquet"


def load_pitcher_names(needed_ids) -> dict[int, str]:
    """pitcher_id → fullName via MLB stats API; cached on disk."""
    import requests

    needed_ids = {int(i) for i in needed_ids}
    if PITCHER_NAME_CACHE.exists():
        cache = pl.read_parquet(PITCHER_NAME_CACHE)
        names = dict(zip(cache["pitcher_id"].to_list(), cache["name"].to_list()))
    else:
        names = {}

    missing = sorted(needed_ids - set(names.keys()))
    if missing:
        for i in range(0, len(missing), 100):
            batch = missing[i : i + 100]
            r = requests.get(
                "https://statsapi.mlb.com/api/v1/people",
                params={"personIds": ",".join(str(x) for x in batch)},
                timeout=30,
            )
            r.raise_for_status()
            for person in r.json().get("people", []):
                names[int(person["id"])] = person.get("fullName", str(person["id"]))
        PITCHER_NAME_CACHE.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {"pitcher_id": list(names.keys()),
             "name":       list(names.values())},
            schema={"pitcher_id": pl.Int64, "name": pl.Utf8},
        ).write_parquet(PITCHER_NAME_CACHE)

    for pid in needed_ids - set(names.keys()):
        names[pid] = str(pid)
    return names


def _ensemble_per_event(s, l, evs: np.ndarray, las: np.ndarray) -> np.ndarray:
    B_ev = s["st_ev"].transform(evs.reshape(-1, 1))
    B_la = s["st_la"].transform(las.reshape(-1, 1))
    beta_mat = s["beta"].reshape(s["K_ev"], s["K_la"])
    p_gam = s["intercept"] + (B_ev @ beta_mat * B_la).sum(axis=1)
    p_lgbm = l["booster"].predict(np.column_stack([evs, las])) + l["init_score"]
    return 0.5 * p_gam + 0.5 * p_lgbm


def write_year_top_bottom(s, l, group_keys, year: int, min_threshold: int,
                          out_path: Path, header: str, n_show: int = 20) -> None:
    """Rank pitchers (or pitcher × pitch_type) in `year` by ensemble pred xwobacon.

    Filter: n_bip ≥ `min_threshold` for pitch_type grain, IP ≥ `min_threshold`
    for pitcher_year grain. n_bip is the year's own BIP count (no pairing).
    """
    has_pt = "pitch_type" in group_keys
    cols = ["pitcher_id", "year", "launch_speed", "launch_angle",
            "event_type", "xwoba_value"] + (["pitch_type"] if has_pt else [])
    bb = (pl.scan_parquet(RAW / f"pitches_{year}.parquet")
            .select(cols)
            .filter(
                pl.col("launch_speed").is_not_null()
                & pl.col("launch_angle").is_not_null()
                & pl.col("xwoba_value").is_not_null()
                & (pl.col("pitch_type").is_not_null() if has_pt else pl.lit(True))
            ).collect())

    p_ens = _ensemble_per_event(s, l,
                                 bb["launch_speed"].to_numpy(),
                                 bb["launch_angle"].to_numpy())
    bb = bb.with_columns(pl.Series("p_ens", p_ens))

    agg_keys = ["pitcher_id"] + (["pitch_type"] if has_pt else [])
    agg = bb.group_by(agg_keys).agg(
        pl.len().alias("n_bip"),
        pl.col("p_ens").mean().alias("pred_xwobacon"),
        pl.col("xwoba_value").mean().alias("xwobacon_actual"),
    )

    if has_pt:
        agg = agg.filter(pl.col("n_bip") >= min_threshold)
        show_cols = ["pitcher_name", "pitch_type", "n_bip", "pred_xwobacon", "xwobacon_actual"]
    else:
        ip = (pitcher_season_ip(years=(year,))
                .filter(pl.col("year") == year)
                .select("pitcher_id", "ip"))
        agg = (agg.join(ip, on="pitcher_id", how="left")
                  .with_columns(pl.col("ip").fill_null(0.0))
                  .filter(pl.col("ip") >= min_threshold))
        show_cols = ["pitcher_name", "ip", "n_bip", "pred_xwobacon", "xwobacon_actual"]

    names = load_pitcher_names(agg["pitcher_id"].to_list())
    agg = agg.with_columns(
        pl.col("pitcher_id")
          .map_elements(lambda i: names.get(int(i), str(i)), return_dtype=pl.Utf8)
          .alias("pitcher_name")
    )

    df_sorted = agg.sort("pred_xwobacon")
    top = df_sorted.head(n_show)
    bot = df_sorted.tail(n_show).reverse()

    col_labels = {"pitcher_name": "pitcher", "pitch_type": "pitch_type",
                  "n_bip": "n_bip", "ip": "ip",
                  "pred_xwobacon": "pred_xwobacon", "xwobacon_actual": "xwobacon"}

    def cell(c, v):
        if c in ("pred_xwobacon", "xwobacon_actual"):
            return "—" if v is None or not np.isfinite(v) else f"{v:.5f}"
        if c == "ip":
            return f"{v:.1f}"
        if c == "n_bip":
            return f"{v:d}"
        return str(v)

    body_rows = list(df_sorted.iter_rows(named=True))
    widths = {c: max(len(col_labels[c]),
                     max((len(cell(c, r[c])) for r in body_rows), default=0))
              for c in show_cols}

    def fmt_header_row():
        parts = [f"{'rank':>4}"] + [f"{col_labels[c]:>{widths[c]}s}" for c in show_cols]
        return "  ".join(parts)

    def fmt_row(rank, row):
        parts = [f"{rank:>4}"] + [f"{cell(c, row[c]):>{widths[c]}s}" for c in show_cols]
        return "  ".join(parts)

    sep_len = len(fmt_header_row())
    lines = [header, ""]
    for title, sub in (
        (f"Top {n_show} (lowest predicted xwobacon)", top),
        (f"Bottom {n_show} (highest predicted xwobacon)", bot),
    ):
        lines.append(title)
        lines.append("-" * sep_len)
        lines.append(fmt_header_row())
        for i, row in enumerate(sub.iter_rows(named=True), 1):
            lines.append(fmt_row(i, row))
        lines.append("")

    out_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def render_alpha_plot(test: pl.DataFrame, self_keys, s, l, beta_tango,
                      ordered_names: list[str], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    preds = per_bip_predictors(test, s, l, beta_tango)
    test_p = test.with_columns([pl.Series(k, v) for k, v in preds.items()])

    sizes = test_p.group_by(list(self_keys)).agg(pl.len().alias("n"))["n"].to_numpy()
    n_max = max(50, int(np.percentile(sizes, 95)))

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for name in ordered_names:
        per = test_p.group_by(list(self_keys)).agg(
            pl.len().alias("n"),
            pl.col(name).mean().alias("mean"),
            pl.col(name).var(ddof=1).alias("var"),
        )
        ns, alphas = alpha_curve(per, n_max=n_max)
        ax.plot(ns, alphas, label=name, linewidth=1.5)

    for thr in (0.5, 0.7):
        ax.axhline(thr, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Number of BIP per group (n)")
    ax.set_ylabel("Cronbach's α (reliability)")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xlim(1, n_max)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- Main ---

def run_grain(loader, out_filename: str, alpha_filename: str | None = None,
              top_bottom_year: int | None = None,
              top_bottom_min: int | None = None,
              top_bottom_filename: str | None = None) -> None:
    train_full, test, test_next, group_keys, self_keys, header_tmpl, test_alpha = loader()

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
        rmse = weighted_rmse(y_te[ok], pred_n[ok], w_te[ok])
        if np.ptp(pred_n[ok]) == 0 or np.ptp(pred_n1[ok]) == 0:
            r2 = rself = float("nan")  # constant predictor — corr undefined
        else:
            r2    = weighted_corr(pred_n[ok], y_te[ok], w_te[ok]) ** 2
            rself = weighted_corr(pred_n[ok], pred_n1[ok], w_te[ok]) ** 2
        rows.append({"name": name, "rmse": rmse, "r2": r2, "r2_self": rself})

    add("ensemble", pred_ens, pred_ens_n1)
    add("gam",      pred_gam, pred_gam_n1)
    add("lgbm",     pred_lgbm, pred_lgbm_n1)

    # Naive baseline: training-set weighted mean of the per-group target.
    trn_tgt = train_full.unique(list(group_keys)).select("xwobacon_next", "n_bip_next")
    xt = trn_tgt["xwobacon_next"].to_numpy()
    wt = trn_tgt["n_bip_next"].to_numpy().astype(np.float64)
    naive_mean = float((wt * xt).sum() / wt.sum())
    pred_naive = np.full_like(y_te, naive_mean, dtype=np.float64)
    add("naive", pred_naive, pred_naive)

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
    def fmt(v: float, spec: str) -> str:
        width = int(spec.split(".")[0].lstrip("+")) if "." in spec else len(f"{0:{spec}}")
        return f"{'—':>{width}}" if not np.isfinite(v) else f"{v:{spec}}"
    for i, r in enumerate(rows, 1):
        lines.append(f"{i:>4}  {r['name']:<12s}  {r['rmse']:.5f}  "
                     f"{fmt(r['r2'], '+8.4f')}  {fmt(r['r2_self'], '+10.6f')}")
    out = "\n".join(lines) + "\n"
    print(out, end="")

    art = ROOT / "artifacts"
    art.mkdir(exist_ok=True)
    (art / out_filename).write_text(out, encoding="utf-8")

    if top_bottom_year is not None:
        has_pt = "pitch_type" in group_keys
        thresh_label = (f"n_bip ≥ {top_bottom_min}" if has_pt
                        else f"IP ≥ {top_bottom_min}")
        grain_label = ("Pitcher × pitch_type" if has_pt else "Pitcher")
        header = (f"{grain_label} top/bottom 20 in {top_bottom_year} "
                  f"by ensemble (50/50 splines + LGBM) predicted xwobacon "
                  f"({thresh_label}, {top_bottom_year} stats only).")
        write_year_top_bottom(
            s, l, group_keys,
            year=top_bottom_year,
            min_threshold=top_bottom_min,
            out_path=art / top_bottom_filename,
            header=header,
        )
        print(f"saved {top_bottom_filename}")

    if alpha_filename is not None:
        n_alpha_groups = test_alpha.select(list(self_keys)).unique().height
        render_alpha_plot(
            test_alpha, self_keys, s, l, beta_tango,
            ordered_names=[r["name"] for r in rows if r["name"] not in ("gam", "lgbm", "naive")],
            out_path=art / alpha_filename,
            title=(f"{TEST_N_YEAR} BBE, no IP/BIP filter "
                   f"(n={n_alpha_groups} groups)\nCronbach's α by BIP per group"),
        )
        print(f"saved {alpha_filename}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    for i, min_bip in enumerate((20, 100)):
        run_grain(lambda mb=min_bip: load_pitch_type_grain(min_bip=mb),
                  f"leaderboard_pitch_type_bip{min_bip}.txt",
                  alpha_filename="cronbach_alpha_pitch_type.png" if i == 0 else None,
                  top_bottom_year=2025 if min_bip == 20 else None,
                  top_bottom_min=20 if min_bip == 20 else None,
                  top_bottom_filename=("top_bottom_ensemble_pitch_type_2025_bip20.txt"
                                       if min_bip == 20 else None))
    for i, min_ip in enumerate((20, 100)):
        run_grain(lambda mi=min_ip: load_pitcher_grain(min_ip=mi),
                  f"leaderboard_pitcher_year_ip{min_ip}.txt",
                  alpha_filename="cronbach_alpha_pitcher_year.png" if i == 0 else None,
                  top_bottom_year=2025 if min_ip == 20 else None,
                  top_bottom_min=20 if min_ip == 20 else None,
                  top_bottom_filename=("top_bottom_ensemble_pitcher_year_2025_ip20.txt"
                                       if min_ip == 20 else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
