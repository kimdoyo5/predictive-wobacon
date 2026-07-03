"""Leaderboard at pitcher-year (default) or pitcher × pitch_type × year grain.

Columns (BIP-weighted on the held-out 2024 → 2025 test set):
  rmse       — RMSE on year-N+1 xwobacon prediction
  r          — corr(predictor, target); the linear-calibration-free ceiling
               (== r(self) by construction for univariate predictors equal to the target).
  r (self)   — corr(predictor_n, predictor_n+1); year-to-year stability.

Rows: ensemble (smoothed 50/50 splines + LGBM — the production model
      defined in src/ensemble.py), gam, lgbm, pwobacon, xwobacon, avg_ev,
      avg_la, naive (constant training-mean target). At pitcher-year grain
      only: K%, BB%, HR% (PA-level rates, shown for self-stability only).

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
from ensemble import (grid_predict_per_bip, grid_predict_per_group,
                       calibrated_grid_predict_per_group, stretch_b,
                       SMOOTH_SIGMA_EV, SMOOTH_SIGMA_LA)
from data import load_batted_balls, pitcher_season_ip, pitcher_season_pa_rates

ART = ROOT / "artifacts"
RAW = ROOT / "data" / "raw"


def load_cached_grids() -> dict:
    """Load the (EV, LA) component + smoothed-ensemble grids saved by
    src/ensemble.py — the production model. All four grain/threshold configs
    score against these same cached grids, so this is loaded once per run.
    """
    cache_path = ART / "ensemble_grid.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} not found — run `uv run python src/ensemble.py` first.")
    z = np.load(cache_path)
    required = {"ev_grid", "la_grid", "grid", "spline_grid", "lgbm_grid",
                "stretch_pm"}
    missing = required - set(z.files)
    if missing:
        raise RuntimeError(
            f"{cache_path} is stale (missing {sorted(missing)}). "
            "Rerun src/ensemble.py to refresh the cache.")
    return {
        "ev_grid":       z["ev_grid"],
        "la_grid":       z["la_grid"],
        "smoothed_grid": z["grid"],
        "spline_grid":   z["spline_grid"],
        "lgbm_grid":     z["lgbm_grid"],
        "pm":            float(z["stretch_pm"]),
    }


def test_targets(test: pl.DataFrame, group_keys):
    """Per-group test targets aligned to a group key — replaces what
    `train_splines`/`train_lgbm` used to compute as a side effect.

    Returns (y_te, w_te, grp_te) where grp_te is keyed by `group_keys` with
    one row per group, used downstream for joining year-N+1 predictions.
    """
    df = test.sort(*group_keys)
    grp = (df.group_by(list(group_keys), maintain_order=True)
             .agg(pl.len().alias("n_p"),
                  pl.col("xwobacon_next").first().alias("y_p"),
                  pl.col("n_bip_next").first().cast(pl.Float64).alias("w_p")))
    return (grp["y_p"].to_numpy().astype(np.float64),
            grp["w_p"].to_numpy().astype(np.float64),
            grp)

# Pitch-type grain config.
PT_MIN_BIP = 20
PT_TRAINVAL_YEARS = tuple(range(2016, 2024))
PT_TEST_YEAR = 2024

METRICS = ["xwobacon", "avg_ev", "avg_la"]
# PA-level rates (per plate appearance). Only meaningful at pitcher-year grain.
PA_METRICS = ["k_pct", "bb_pct", "hr_pct"]
# Per-pitcher-year metrics loaded from the Max pwOBAcon+ CSV. Same units as
# xwobacon (predicted wOBA on contact), so they compete on RMSE / xwOBAcon.
# Only meaningful at pitcher-year grain (CSV is keyed at that grain).
MAX_METRICS = ["pwobacon_max"]
MAX_CSV = "pitcher_pwobacon_plus_2020_26.csv"


def load_max_pwobacon() -> pl.DataFrame:
    """Per (pitcher_id, year): Max pwOBAcon. CSV pitcher column is
    "<mlbam_id><L|R>" — strip the handedness suffix to get pitcher_id.
    """
    df = pl.read_csv(RAW / MAX_CSV)
    return df.with_columns(
        pl.col("pitcher").str.replace(r"[LR]$", "").cast(pl.Int64).alias("pitcher_id"),
        pl.col("game_year").alias("year"),
        pl.col("pwobacon").alias("pwobacon_max"),
    ).select("pitcher_id", "year", "pwobacon_max")


# --- pwobacon 12-bucket OLS (3 LA × 4 EV) ---

LA_BREAKS = [8.0, 32.0]
EV_BREAKS = [95.0, 100.0, 105.0]
N_LA = len(LA_BREAKS) + 1
N_EV = len(EV_BREAKS) + 1
N_BUCKETS = N_LA * N_EV


def bucket_id(ev, la):
    iev = np.searchsorted(EV_BREAKS, ev, side="right")
    ila = np.searchsorted(LA_BREAKS, la, side="right")
    return ila * N_EV + iev


def pwobacon_design(bbe: pl.DataFrame, group_keys):
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


def pwobacon_fit(train_full: pl.DataFrame, group_keys) -> np.ndarray:
    X_tr, y_tr, w_tr, _ = pwobacon_design(train_full, group_keys)
    sw = np.sqrt(w_tr)
    beta, *_ = np.linalg.lstsq(X_tr * sw[:, None], y_tr * sw, rcond=None)
    return beta


# --- Univariate per-season metrics ---

def season_metrics(bbe: pl.DataFrame, group_keys) -> pl.DataFrame:
    """Per group: n_bip, xwobacon, avg_ev, avg_la."""
    return bbe.group_by(list(group_keys)).agg(
        pl.len().alias("n_bip"),
        pl.col("xwoba_value").mean().alias("xwobacon"),
        pl.col("launch_speed").mean().alias("avg_ev"),
        pl.col("launch_angle").mean().alias("avg_la"),
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

    `alpha_bbe` is the full 2016-2025 BBE — every pitcher, every year, no IP
    threshold or pair requirement — used for the chronological Cronbach's α
    accrual plot.
    """
    train, val, test, test_next = load_splits(min_ip=min_ip)
    train_full = pl.concat([train, val])
    alpha_bbe = load_batted_balls()
    group_keys = ("pitcher_id", "year")
    self_keys  = ("pitcher_id",)   # year-N+1 alignment drops the year
    png_main = f"{TEST_N_YEAR} Correlation to {TEST_N_YEAR + 1}"
    png_sub  = (f"Pitcher-year  ·  n={{n}} pitchers, IP ≥ {min_ip} both years  ·  "
                "min(IP, IP_next)-weighted")
    return (train_full, test, test_next, group_keys, self_keys,
            (png_main, png_sub), alpha_bbe)


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
            .select("pitcher_id", "year", "game_date", "ab_number", "index_play",
                    "pitch_type", "launch_speed", "launch_angle",
                    "event_type", "xwoba_value")
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
    # Full 2016-2025 BBE with non-null pitch_type, for the chronological α plot.
    alpha_bbe = bb

    group_keys = ("pitcher_id", "pitch_type", "year")
    self_keys  = ("pitcher_id", "pitch_type")
    png_main = f"{TEST_N_YEAR} Correlation to {TEST_N_YEAR + 1}"
    png_sub  = (f"Pitcher × Pitch Type × Year  ·  n={{n}} combos  ·  "
                f"min(n_bip, n_bip_next) ≥ {min_bip}, weighted")
    return (train_full, test, test_next, group_keys, self_keys,
            (png_main, png_sub), alpha_bbe)


# --- Cronbach's α curves ---

def per_bip_predictors(test: pl.DataFrame, grids: dict,
                       beta_pwobacon) -> dict[str, np.ndarray]:
    """For each leaderboard model, predict at the BIP level on `test` by
    bilinear-interpolating the cached (EV, LA) grids loaded from
    `ensemble_grid.npz` — no model retraining.
    """
    evs = test["launch_speed"].to_numpy()
    las = test["launch_angle"].to_numpy()
    return {
        "ensemble": grid_predict_per_bip(grids["smoothed_grid"],
                                          grids["ev_grid"], grids["la_grid"],
                                          evs, las),
        "gam":      grid_predict_per_bip(grids["spline_grid"],
                                          grids["ev_grid"], grids["la_grid"],
                                          evs, las),
        "lgbm":     grid_predict_per_bip(grids["lgbm_grid"],
                                          grids["ev_grid"], grids["la_grid"],
                                          evs, las),
        "pwobacon": beta_pwobacon[bucket_id(evs, las)],
        "xwobacon": test["xwoba_value"].to_numpy(),
        "avg_ev":   evs,
        "avg_la":   las,
    }


def chronological_alpha_curve(values_per_group: list[np.ndarray],
                              ns: np.ndarray) -> np.ndarray:
    """Empirical one-way-ANOVA reliability evaluated at accumulating BIPs.

    Each element of `values_per_group` is the metric series for one group
    (pitcher or pitcher × pitch_type), ordered chronologically. For each
    n in `ns`, restrict to groups with ≥n BIPs, take their first n values,
    and compute:

        σ²_e(n)       = mean within-group sample variance of those n values
        σ²_between(n) = Var across groups of [mean of those n values]
        σ²_t(n)       = max(σ²_between(n) − σ²_e(n)/n, 0)
        α(n)          = σ²_t(n) / σ²_between(n)

    α(n) is NaN where n<2 or fewer than 5 groups qualify.
    """
    if not values_per_group:
        return np.full(len(ns), np.nan)

    lens = np.array([len(v) for v in values_per_group], dtype=np.int64)
    max_len = int(lens.max()) if lens.size else 0
    n_groups = len(values_per_group)

    cs  = np.full((n_groups, max_len), np.nan)
    cs2 = np.full((n_groups, max_len), np.nan)
    for i, vals in enumerate(values_per_group):
        L = lens[i]
        if L == 0:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        cs[i,  :L] = arr.cumsum()
        cs2[i, :L] = (arr * arr).cumsum()

    alphas = np.full(len(ns), np.nan)
    for j, n in enumerate(ns):
        if n < 2 or n > max_len:
            continue
        col = n - 1
        mask = lens >= n
        if int(mask.sum()) < 5:
            continue
        s  = cs[mask,  col]
        ss = cs2[mask, col]
        m  = s / n
        v  = (ss - s * s / n) / (n - 1)
        sigma2_e       = float(np.nanmean(v))
        sigma2_between = float(np.var(m, ddof=1))
        if sigma2_between <= 0:
            continue
        sigma2_t = max(sigma2_between - sigma2_e / n, 0.0)
        alphas[j] = sigma2_t / sigma2_between
    return alphas


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


TOP_BOTTOM_YEARS = tuple(range(2016, 2026))
TOP_BOTTOM_MIN   = 100   # n_bip for pitch_type grain, IP for pitcher_year grain


def render_career_top_bottom_png(grids: dict, group_keys, out_path: Path,
                                   png_main_title: str, png_subtitle: str,
                                   n_show: int = 20) -> None:
    """Rank pitchers (or pitcher × pitch_type) over 2016-2025 by mean
    ensemble-predicted xwobacon (each BIP weighted equally), and render the
    top/bottom-N PNG.

    Filter: total n_bip ≥ TOP_BOTTOM_MIN for pitch_type grain, total IP ≥
    TOP_BOTTOM_MIN for pitcher_year grain.
    """
    has_pt = "pitch_type" in group_keys
    cols = ["pitcher_id", "year", "launch_speed", "launch_angle",
            "event_type", "xwoba_value"] + (["pitch_type"] if has_pt else [])
    frames = [
        pl.scan_parquet(RAW / f"pitches_{y}.parquet")
        .select(cols)
        .filter(
            pl.col("launch_speed").is_not_null()
            & pl.col("launch_angle").is_not_null()
            & pl.col("xwoba_value").is_not_null()
            & (pl.col("pitch_type").is_not_null() if has_pt else pl.lit(True))
        )
        for y in TOP_BOTTOM_YEARS
    ]
    bb = pl.concat(frames, how="vertical").collect()

    p_ens = grid_predict_per_bip(
        grids["smoothed_grid"], grids["ev_grid"], grids["la_grid"],
        bb["launch_speed"].to_numpy(), bb["launch_angle"].to_numpy(),
    )
    bb = bb.with_columns(pl.Series("p_ens", p_ens))

    agg_keys = ["pitcher_id"] + (["pitch_type"] if has_pt else [])
    agg = bb.group_by(agg_keys).agg(
        pl.len().alias("n_bip"),
        pl.col("p_ens").mean().alias("pred_xwobacon"),
        pl.col("xwoba_value").mean().alias("xwobacon_actual"),
    )
    pm = grids["pm"]
    n_career = agg["n_bip"].to_numpy().astype(np.float64)
    agg = agg.with_columns(pl.Series(
        "pred_xwobacon", pm + stretch_b(n_career) * (agg["pred_xwobacon"].to_numpy() - pm)))

    if has_pt:
        agg = agg.filter(pl.col("n_bip") >= TOP_BOTTOM_MIN)
    else:
        ip = (pitcher_season_ip(years=TOP_BOTTOM_YEARS)
                .group_by("pitcher_id")
                .agg(pl.col("ip").sum().alias("ip")))
        agg = (agg.join(ip, on="pitcher_id", how="left")
                  .with_columns(pl.col("ip").fill_null(0.0))
                  .filter(pl.col("ip") >= TOP_BOTTOM_MIN))

    names = load_pitcher_names(agg["pitcher_id"].to_list())
    agg = agg.with_columns(
        pl.col("pitcher_id")
          .map_elements(lambda i: names.get(int(i), str(i)), return_dtype=pl.Utf8)
          .alias("pitcher_name")
    )

    df_sorted = agg.sort("pred_xwobacon")
    top = df_sorted.head(n_show)
    bot = df_sorted.tail(n_show).reverse()

    render_top_bottom_png(
        main_title=png_main_title, subtitle=png_subtitle,
        has_pt=has_pt, top=top, bot=bot,
        out_path=out_path, n_show=n_show,
    )


# --- Mobile-friendly PNG renderers for the leaderboards ---

def _display_name(name: str) -> str:
    """Map internal row name (used in data + txt) to the proper display name
    shown in the PNG output.
    """
    mapping = {
        "ensemble": "Ensemble",
        "lgbm":     "LightGBM",
        "gam":      "GAM",
        "tango":           "pwOBAcon (Tango)",
        "pwobacon":        "pwOBAcon (Tango)",
        "pwobacon_max":    "pwOBAcon (Max)",
        "xwobacon": "xwOBAcon",
        "avg_ev":   "Avg EV",
        "avg_la":   "Avg LA",
        "k_pct":    "K%",
        "bb_pct":   "BB%",
        "hr_pct":   "HR%",
        "naive":    "Naive",
    }
    if name in mapping:
        return mapping[name]
    if name.startswith("ensemble (smoothed"):
        return f"Ensemble (smoothed, σ={SMOOTH_SIGMA_EV:g}/{SMOOTH_SIGMA_LA:g})"
    if name.startswith("lgbm (smoothed"):
        return f"LightGBM (smoothed, σ={SMOOTH_SIGMA_EV:g}/{SMOOTH_SIGMA_LA:g})"
    return name


PNG_DPI = 220
# Editorial palette with maroon highlights.
HEAD_BG = "#1a1a1a"     # near-black header band
HEAD_FG = "#ffffff"
ALT_BG  = "#f5f5f4"     # warm light gray alternating rows
GRID    = "#ffffff"
HL_FG   = "#8b2635"     # rich maroon for bolded rows / columns
TITLE_FG    = "#0c0a09" # near-black main title
SUBTITLE_FG = "#57534e" # warm gray subtitle
ACCENT      = "#8b2635" # maroon divider rule
SECTION_FG  = "#0c0a09" # near-black "Top 20 / Bottom 20" labels

TITLE_BLOCK_IN = 0.85   # vertical inches reserved for main title + subtitle + divider


def _draw_title_block(fig, main_title: str, subtitle: str) -> None:
    """Draw a two-line title block (main + subtitle) and a thin accent divider
    at the top of `fig`. Caller is responsible for reserving TITLE_BLOCK_IN
    inches at the top of the figure for this block.
    """
    import matplotlib.lines as mlines

    fig_h = fig.get_figheight()
    y_main = 1 - 0.25 / fig_h
    y_sub  = 1 - 0.52 / fig_h
    y_rule = 1 - 0.74 / fig_h

    fig.text(0.5, y_main, main_title, ha="center", va="top",
             fontsize=14, fontweight="bold", color=TITLE_FG)
    if subtitle:
        fig.text(0.5, y_sub, subtitle, ha="center", va="top",
                 fontsize=8.5, color=SUBTITLE_FG, style="italic")
    fig.add_artist(mlines.Line2D(
        [0.08, 0.92], [y_rule, y_rule],
        transform=fig.transFigure,
        color=ACCENT, linewidth=1.5, alpha=0.85, solid_capstyle="round",
    ))


def _style_table(table, n_data_rows: int, n_cols: int,
                 bold_data_rows: set[int] | None = None,
                 bold_data_cols: set[int] | None = None,
                 left_align_cols: set[int] | None = None) -> None:
    bold_data_rows = bold_data_rows or set()
    bold_data_cols = bold_data_cols or set()
    left_align_cols = left_align_cols or set()

    for c in range(n_cols):
        cell = table[(0, c)]
        cell.set_facecolor(HEAD_BG)
        cell.set_edgecolor(HEAD_BG)
        cell.set_text_props(color=HEAD_FG, weight="bold",
                            ha=("left" if c in left_align_cols else "center"))

    for i in range(n_data_rows):
        bg = ALT_BG if i % 2 == 0 else "white"
        row_bold = i in bold_data_rows
        for c in range(n_cols):
            cell = table[(i + 1, c)]
            cell.set_facecolor(bg)
            cell.set_edgecolor(GRID)
            col_bold = c in bold_data_cols
            bold = row_bold or col_bold
            cell.set_text_props(
                weight="bold" if bold else "normal",
                color=HL_FG if bold else "black",
                ha=("left" if c in left_align_cols else "center"),
            )


def render_main_leaderboard_png(main_title: str, subtitle: str,
                                 rows: list[dict], out_path: Path) -> None:
    """Render the per-grain leaderboard table to a mobile-friendly PNG.

    Bolds rows whose `name` is "ensemble" or "xwobacon".
    """
    import matplotlib.pyplot as plt

    col_labels = ["#", "Model", "RMSE", "xwOBAcon", "Self"]
    col_widths = [0.06, 0.37, 0.19, 0.19, 0.19]

    cell_text: list[list[str]] = []
    bold_rows: set[int] = set()
    for i, r in enumerate(rows):
        cell_text.append([
            str(i + 1),
            _display_name(r["name"]),
            "—" if not np.isfinite(r["rmse"])   else f"{r['rmse']:.4f}",
            "—" if not np.isfinite(r["r"])      else f"{r['r']:.3f}",
            "—" if not np.isfinite(r["r_self"]) else f"{r['r_self']:.3f}",
        ])
        if r["name"].startswith("ensemble") or r["name"] == "xwobacon":
            bold_rows.add(i)

    n = len(rows)
    row_h = 0.36
    fig_w = 4.8
    fig_h = TITLE_BLOCK_IN + row_h * (n + 1) + 0.12
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=PNG_DPI)
    ax.set_axis_off()
    ax.set_position([0.02, 0.03, 0.96, 1 - (TITLE_BLOCK_IN / fig_h) - 0.03])

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        colWidths=col_widths,
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    _style_table(
        table, n_data_rows=n, n_cols=len(col_labels),
        bold_data_rows=bold_rows,
        left_align_cols={1},
    )

    _draw_title_block(fig, main_title, subtitle)
    fig.savefig(out_path, dpi=PNG_DPI, bbox_inches="tight",
                facecolor="white", pad_inches=0.08)
    plt.close(fig)


def render_top_bottom_png(main_title: str, subtitle: str, has_pt: bool,
                           top: pl.DataFrame, bot: pl.DataFrame,
                           out_path: Path, n_show: int = 20) -> None:
    """Render top/bottom-N ensemble leaderboards as a mobile-friendly PNG.

    Two stacked tables (top, bottom).
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if has_pt:
        col_labels = ["#", "pitcher", "Pitch Type", "n_bip", "pred", "actual"]
        col_widths = [0.07, 0.35, 0.17, 0.11, 0.15, 0.15]
        meta_col, meta_fmt = "pitch_type", (lambda v: str(v))
    else:
        col_labels = ["#", "pitcher", "IP", "n_bip", "pred", "actual"]
        col_widths = [0.07, 0.40, 0.11, 0.12, 0.15, 0.15]
        meta_col, meta_fmt = "ip", (lambda v: f"{v:.1f}")

    def to_cells(df: pl.DataFrame) -> list[list[str]]:
        rows: list[list[str]] = []
        for i, r in enumerate(df.iter_rows(named=True), 1):
            pred = r["pred_xwobacon"]
            act  = r["xwobacon_actual"]
            rows.append([
                str(i),
                r["pitcher_name"],
                meta_fmt(r[meta_col]),
                f"{r['n_bip']:d}",
                "—" if pred is None or not np.isfinite(pred) else f"{pred:.4f}",
                "—" if act  is None or not np.isfinite(act)  else f"{act:.4f}",
            ])
        return rows

    top_rows = to_cells(top)
    bot_rows = to_cells(bot)

    fig_w = 5.6
    row_h = 0.32
    section_title_h = 0.32
    section_h = section_title_h + row_h * (n_show + 1)
    fig_h = TITLE_BLOCK_IN + 2 * section_h + 0.4

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=PNG_DPI)
    gs = GridSpec(2, 1, figure=fig,
                  top=1 - TITLE_BLOCK_IN / fig_h, bottom=0.03,
                  left=0.03, right=0.97, hspace=0.20)

    sections = [
        (gs[0, 0], f"Top {n_show} (lowest predicted xwobacon)", top_rows),
        (gs[1, 0], f"Bottom {n_show} (highest predicted xwobacon)", bot_rows),
    ]
    for slot, sub_title, cells in sections:
        ax = fig.add_subplot(slot)
        ax.set_axis_off()
        ax.text(0.0, 1.0, sub_title, transform=ax.transAxes,
                ha="left", va="top",
                fontsize=10.5, fontweight="bold", color=SECTION_FG)
        table = ax.table(
            cellText=cells,
            colLabels=col_labels,
            cellLoc="center",
            colLoc="center",
            colWidths=col_widths,
            bbox=[0, 0, 1, 0.92],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        _style_table(
            table, n_data_rows=len(cells), n_cols=len(col_labels),
            bold_data_cols=set(),
            left_align_cols={1},
        )

    _draw_title_block(fig, main_title, subtitle)
    fig.savefig(out_path, dpi=PNG_DPI, bbox_inches="tight",
                facecolor="white", pad_inches=0.18)
    plt.close(fig)


def render_alpha_plot(bbe: pl.DataFrame, self_keys, grids: dict, beta_pwobacon,
                      ordered_names: list[str], out_path: Path, title: str,
                      min_bip_pool: int) -> None:
    """Chronological-accrual Cronbach's α plot.

    Sort BBE chronologically within each group (pitcher or pitcher × pitch_type),
    fix a cohort of groups with ≥`min_bip_pool` BIPs over 2016-2025, and plot
    α(n) for n=2..min_bip_pool using `chronological_alpha_curve`.
    """
    import matplotlib.pyplot as plt

    sort_cols = list(self_keys) + ["game_date", "ab_number", "index_play"]
    bbe = bbe.sort(sort_cols)

    preds = per_bip_predictors(bbe, grids, beta_pwobacon)
    bbe = bbe.with_columns([pl.Series(k, v) for k, v in preds.items()])

    counts = bbe.group_by(list(self_keys)).agg(pl.len().alias("N"))
    cohort = counts.filter(pl.col("N") >= min_bip_pool).select(list(self_keys))
    n_pool = cohort.height
    if n_pool < 5:
        print(f"  α plot: only {n_pool} groups have ≥{min_bip_pool} BIPs; skipping",
              file=sys.stderr)
        return
    bbe = bbe.join(cohort, on=list(self_keys), how="inner")

    grouped = bbe.group_by(list(self_keys), maintain_order=True).agg(
        [pl.col(name).alias(name) for name in ordered_names]
    )

    # Sub-sample the n grid on a logish spacing so the curve is fast to render
    # while keeping every point at small n where reliability moves fastest.
    ns_dense  = np.arange(2, min(50, min_bip_pool) + 1)
    if min_bip_pool > 50:
        ns_sparse = np.unique(np.geomspace(51, min_bip_pool, num=200).astype(int))
        ns = np.concatenate([ns_dense, ns_sparse])
    else:
        ns = ns_dense

    fig, ax = plt.subplots(figsize=(8, 6.3), constrained_layout=True)
    final_pts: list[tuple[str, float, tuple]] = []
    for name in ordered_names:
        vals_per_group = [np.asarray(v, dtype=np.float64)
                          for v in grouped[name].to_list()]
        alphas = chronological_alpha_curve(vals_per_group, ns)
        finite = np.isfinite(alphas)
        if not finite.any():
            continue
        last = float(alphas[finite][-1])
        line, = ax.plot(ns[finite], alphas[finite],
                        label=f"{name} (α={last:.2f})",
                        linewidth=2.2, linestyle="-")
        final_pts.append((name, last, line.get_color()))

    for thr in (0.5, 0.7):
        ax.axhline(thr, color="gray", linestyle=":", linewidth=1.2)
    ax.set_xlabel("BIP accumulated chronologically per group (n)", fontsize=15)
    ax.set_ylabel("Cronbach's α (reliability)", fontsize=15)
    ax.set_title(f"{title}\n(cohort: {n_pool} groups with ≥{min_bip_pool} career BIPs, 2016-2025)",
                 fontsize=15)
    ax.set_ylim(-0.02, 1.0)
    ax.set_xlim(1, min_bip_pool * 1.13)
    ax.tick_params(labelsize=13)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=13, ncol=2)
    x_anchor = min_bip_pool
    for name, y, color in final_pts:
        ax.annotate(name, xy=(x_anchor, y), xytext=(4, 0),
                    textcoords="offset points", fontsize=10, color=color,
                    va="center")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# --- Main ---

def run_grain(loader, grids: dict, out_filename: str,
              alpha_filename: str | None = None,
              alpha_min_bip_pool: int = 1000,
              top_bottom_png: str | None = None) -> None:
    train_full, test, test_next, group_keys, self_keys, header_info, alpha_bbe = loader()
    png_main, png_sub_tmpl = header_info

    # All spline/LGBM/ensemble predictions come from the cached (EV, LA)
    # grids in `grids` (built once by src/ensemble.py), so this script
    # does no model retraining per grain.
    y_te, w_te, grp_te = test_targets(test, group_keys)

    def predict_n(grid):
        return grid_predict_per_group(test, grid, grids["ev_grid"],
                                       grids["la_grid"], group_keys, grp_te)

    def predict_n1(grid):
        return grid_predict_per_group(test_next, grid, grids["ev_grid"],
                                       grids["la_grid"], self_keys, grp_te)

    pred_gam,  pred_gam_n1  = predict_n(grids["spline_grid"]),  predict_n1(grids["spline_grid"])
    pred_lgbm, pred_lgbm_n1 = predict_n(grids["lgbm_grid"]),    predict_n1(grids["lgbm_grid"])
    # Production ensemble: smoothed grid + b(n_bip) calibration stretch.
    pred_ens    = calibrated_grid_predict_per_group(
        test, grids["smoothed_grid"], grids["ev_grid"], grids["la_grid"],
        group_keys, grp_te, grids["pm"])
    pred_ens_n1 = calibrated_grid_predict_per_group(
        test_next, grids["smoothed_grid"], grids["ev_grid"], grids["la_grid"],
        self_keys, grp_te, grids["pm"])

    rows: list[dict] = []
    def add(name, pred_n, pred_n1):
        ok = np.isfinite(pred_n) & np.isfinite(pred_n1)
        rmse = weighted_rmse(y_te[ok], pred_n[ok], w_te[ok])
        if np.ptp(pred_n[ok]) == 0 or np.ptp(pred_n1[ok]) == 0:
            r_val = rself = float("nan")  # constant predictor — corr undefined
        else:
            r_val = weighted_corr(pred_n[ok], y_te[ok], w_te[ok])
            rself = weighted_corr(pred_n[ok], pred_n1[ok], w_te[ok])
        rows.append({"name": name, "rmse": rmse, "r": r_val, "r_self": rself})

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

    # --- pwobacon ---
    beta_pwobacon = pwobacon_fit(train_full, group_keys)
    X_n,  _, _, key_n  = pwobacon_design(test,      group_keys)
    X_n1, _, _, key_n1 = pwobacon_design(test_next, group_keys)

    def to_key_tuples(df: pl.DataFrame, cols) -> list[tuple]:
        return list(zip(*[df[c].to_list() for c in cols]))

    pred_n_map  = dict(zip(to_key_tuples(key_n,  group_keys), (X_n  @ beta_pwobacon).tolist()))
    pred_n1_map = dict(zip(to_key_tuples(key_n1, self_keys),  (X_n1 @ beta_pwobacon).tolist()))
    keys_te_n   = to_key_tuples(grp_te, group_keys)
    keys_te_n1  = to_key_tuples(grp_te, self_keys)
    p_n  = np.array([pred_n_map.get(k,  np.nan) for k in keys_te_n])
    p_n1 = np.array([pred_n1_map.get(k, np.nan) for k in keys_te_n1])
    add("pwobacon", p_n, p_n1)

    # --- Univariate linregs on per-season metrics ---
    pst_trv = season_metrics(train_full, group_keys)
    nxt_target = (train_full.unique(list(group_keys))
                            .select([*group_keys, "xwobacon_next", "n_bip_next"]))
    pst_trv = pst_trv.join(nxt_target, on=list(group_keys), how="inner")

    pst_te    = season_metrics(test,      group_keys)
    pst_te_n1 = season_metrics(test_next, self_keys)  # drop "year" for alignment

    # PA-level rates + Max pwOBAcon — only at pitcher-year grain.
    metrics_iter = list(METRICS)
    if group_keys == ("pitcher_id", "year"):
        pa_rates = pitcher_season_pa_rates()
        pa_n  = pa_rates.select(["pitcher_id", "year", *PA_METRICS])
        pa_n1 = (pa_rates.filter(pl.col("year") == TEST_N_YEAR + 1)
                          .select(["pitcher_id", *PA_METRICS]))
        pst_trv   = pst_trv.join(pa_n,   on=["pitcher_id", "year"], how="left")
        pst_te    = pst_te.join(pa_n,    on=["pitcher_id", "year"], how="left")
        pst_te_n1 = pst_te_n1.join(pa_n1, on=["pitcher_id"], how="left")
        metrics_iter.extend(PA_METRICS)

        max_pw = load_max_pwobacon()
        mx_n  = max_pw.select(["pitcher_id", "year", *MAX_METRICS])
        mx_n1 = (max_pw.filter(pl.col("year") == TEST_N_YEAR + 1)
                        .select(["pitcher_id", *MAX_METRICS]))
        pst_trv   = pst_trv.join(mx_n,   on=["pitcher_id", "year"], how="left")
        pst_te    = pst_te.join(mx_n,    on=["pitcher_id", "year"], how="left")
        pst_te_n1 = pst_te_n1.join(mx_n1, on=["pitcher_id"], how="left")
        metrics_iter.extend(MAX_METRICS)

    y_trv = pst_trv["xwobacon_next"].to_numpy()
    w_trv = pst_trv["n_bip_next"].to_numpy().astype(np.float64)
    paired_n = grp_te.select(list(group_keys)).join(
        pst_te.select([*group_keys, *metrics_iter]),
        on=list(group_keys), how="left",
    )
    paired_n1 = grp_te.select(list(self_keys)).join(
        pst_te_n1.select([*self_keys, *metrics_iter]),
        on=list(self_keys), how="left",
    )

    for metric in metrics_iter:
        x_n  = paired_n[metric].to_numpy()
        x_n1 = paired_n1[metric].to_numpy()
        if metric in PA_METRICS:
            # PA-level rates aren't in xwobacon units, so RMSE and
            # corr(x, xwobacon) aren't meaningful — only self-stability is.
            rmse  = float("nan")
            r_val = float("nan")
        else:
            x_trv = pst_trv[metric].to_numpy()
            f_trv = np.isfinite(x_trv) & np.isfinite(y_trv) & np.isfinite(w_trv)
            a, b = fit_univariate(x_trv[f_trv], y_trv[f_trv], w_trv[f_trv])
            yhat_n = a + b * x_n
            ok_y   = np.isfinite(yhat_n) & np.isfinite(x_n1)
            rmse   = weighted_rmse(y_te[ok_y], yhat_n[ok_y], w_te[ok_y])
            r_val  = weighted_corr(x_n[ok_y], y_te[ok_y], w_te[ok_y])
        ok_self = np.isfinite(x_n) & np.isfinite(x_n1)
        rself = weighted_corr(x_n[ok_self], x_n1[ok_self], w_te[ok_self])
        rows.append({"name": metric, "rmse": rmse, "r": r_val, "r_self": rself})

    # NaN rmse rows (the PA-level baselines) sort to the bottom.
    rows.sort(key=lambda r: (not np.isfinite(r["rmse"]), r["rmse"]))
    # Emit ensemble row metrics to stdout for sweep harnesses.
    ens_row = next((r for r in rows if r["name"] == "ensemble"), None)
    if ens_row is not None:
        slug = Path(out_filename).stem
        print(f"SLICE_METRIC {slug} rmse={ens_row['rmse']:.5f} "
              f"r={ens_row['r']:.4f} r_self={ens_row['r_self']:.4f}",
              file=sys.stderr)
    art = ROOT / "artifacts"
    art.mkdir(exist_ok=True)
    png_path = art / (Path(out_filename).stem + ".png")
    render_main_leaderboard_png(
        main_title=png_main,
        subtitle=png_sub_tmpl.format(n=len(y_te)),
        rows=rows,
        out_path=png_path,
    )
    print(f"saved {png_path.name}", file=sys.stderr)

    if top_bottom_png is not None:
        has_pt = "pitch_type" in group_keys
        thresh_label = (f"n_bip ≥ {TOP_BOTTOM_MIN}" if has_pt
                        else f"IP ≥ {TOP_BOTTOM_MIN}")
        png_grain_label = ("Pitcher × Pitch Type" if has_pt else "Pitcher")
        yr_label = f"{TOP_BOTTOM_YEARS[0]}-{TOP_BOTTOM_YEARS[-1]}"
        png_main_tb = f"Top / Bottom 20 — {png_grain_label}, {yr_label}"
        png_sub_tb  = ("Ranked by ensemble (smoothed splines + LGBM) "
                       f"predicted xwobacon  ·  {thresh_label}")
        tb_png_path = art / top_bottom_png
        render_career_top_bottom_png(
            grids, group_keys,
            out_path=tb_png_path,
            png_main_title=png_main_tb,
            png_subtitle=png_sub_tb,
        )
        print(f"saved {tb_png_path.name}", file=sys.stderr)

    if alpha_filename is not None:
        # α plot has always omitted gam/lgbm/naive — show ensemble + the
        # univariate baselines + pwobacon. PA-level rates (K%/BB%/HR%) and
        # Max pwOBAcon are per-pitcher-year aggregates, not per-BIP, so
        # they can't be plotted on the BIP-accrual axis.
        alpha_skip = {"gam", "lgbm", "naive", *PA_METRICS, *MAX_METRICS}
        render_alpha_plot(
            alpha_bbe, self_keys, grids, beta_pwobacon,
            ordered_names=[r["name"] for r in rows
                            if r["name"] not in alpha_skip],
            out_path=art / alpha_filename,
            title=("Chronological Cronbach's α accrual"),
            min_bip_pool=alpha_min_bip_pool,
        )
        print(f"saved {alpha_filename}")


def render_calibration_scatter(grids: dict, out_path: Path,
                                 min_bip: int = 10) -> None:
    """Per-pitcher career calibration scatter (2016-2025).

    x = predicted xwobacon (mean per-BIP ensemble grid lookup, equal-BIP weight)
    y = actual    xwobacon (mean per-BIP xwoba_value, equal-BIP weight)
    size ∝ n_bip
    """
    import matplotlib.pyplot as plt

    cols = ["pitcher_id", "launch_speed", "launch_angle", "xwoba_value"]
    frames = [
        pl.scan_parquet(RAW / f"pitches_{y}.parquet")
        .select(cols)
        .filter(
            pl.col("launch_speed").is_not_null()
            & pl.col("launch_angle").is_not_null()
            & pl.col("xwoba_value").is_not_null()
        )
        for y in TOP_BOTTOM_YEARS
    ]
    bb = pl.concat(frames, how="vertical").collect()

    p_ens = grid_predict_per_bip(
        grids["smoothed_grid"], grids["ev_grid"], grids["la_grid"],
        bb["launch_speed"].to_numpy(), bb["launch_angle"].to_numpy(),
    )
    bb = bb.with_columns(pl.Series("p_ens", p_ens))
    agg = (bb.group_by("pitcher_id")
              .agg(pl.len().alias("n_bip"),
                   pl.col("p_ens").mean().alias("pred"),
                   pl.col("xwoba_value").mean().alias("actual"))
              .filter(pl.col("n_bip") >= min_bip))

    pm = grids["pm"]
    n_bip = agg["n_bip"].to_numpy().astype(np.float64)
    pred  = pm + stretch_b(n_bip) * (agg["pred"].to_numpy() - pm)
    actual = agg["actual"].to_numpy()

    # Weighted OLS slope/intercept of actual on pred (for calibration line).
    w = n_bip
    pm = (w * pred).sum() / w.sum()
    am = (w * actual).sum() / w.sum()
    b = ((w * (pred - pm) * (actual - am)).sum()
         / (w * (pred - pm) ** 2).sum())
    a = am - b * pm
    r = (((w * (pred - pm) * (actual - am)).sum())
         / np.sqrt((w * (pred - pm) ** 2).sum()
                   * (w * (actual - am) ** 2).sum()))

    fig, ax = plt.subplots(figsize=(8.5, 7.0), constrained_layout=True)
    sizes = 4.0 + 0.8 * np.sqrt(n_bip)
    ax.scatter(pred, actual, s=sizes, alpha=0.35, color="#1f3a93",
                edgecolors="none")

    lo = float(min(pred.min(), actual.min())) - 0.005
    hi = float(max(pred.max(), actual.max())) + 0.005
    xs = np.linspace(lo, hi, 200)
    ax.plot(xs, xs, color="black", linestyle="--", linewidth=1.0,
             label="perfect calibration (y = x)")
    ax.plot(xs, a + b * xs, color="crimson", linewidth=1.8,
             label=f"weighted OLS: y = {a:.3f} + {b:.3f}·x   (r²={r*r:.3f})")

    ax.set_xlabel("Predicted xwOBAcon (ensemble, career-mean per-BIP)", fontsize=13)
    ax.set_ylabel("Actual xwOBAcon (career-mean per-BIP)", fontsize=13)
    ax.set_title(
        f"Pitcher Calibration  ·  career 2016-2025  ·  n={len(pred)} pitchers, "
        f"min BIP = {min_bip}\ncircle area ∝ BIP",
        fontsize=14,
    )
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=11)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    grids = load_cached_grids()
    yr_tag = f"{TOP_BOTTOM_YEARS[0]}_{TOP_BOTTOM_YEARS[-1]}"

    render_calibration_scatter(grids, ART / "calibration_scatter.png", min_bip=10)
    print("saved calibration_scatter.png", file=sys.stderr)
    for i, min_bip in enumerate((20, 100)):
        run_grain(lambda mb=min_bip: load_pitch_type_grain(min_bip=mb),
                  grids,
                  f"leaderboard_pitch_type_bip{min_bip}.txt",
                  alpha_filename="cronbach_alpha_pitch_type.png" if i == 0 else None,
                  alpha_min_bip_pool=500,
                  top_bottom_png=(f"top_bottom_ensemble_pitch_type_{yr_tag}_bip{TOP_BOTTOM_MIN}.png"
                                   if min_bip == 20 else None))
    for i, min_ip in enumerate((20, 100)):
        run_grain(lambda mi=min_ip: load_pitcher_grain(min_ip=mi),
                  grids,
                  f"leaderboard_pitcher_year_ip{min_ip}.txt",
                  alpha_filename="cronbach_alpha_pitcher_year.png" if i == 0 else None,
                  alpha_min_bip_pool=1500,
                  top_bottom_png=(f"top_bottom_ensemble_pitcher_year_{yr_tag}_ip{TOP_BOTTOM_MIN}.png"
                                   if min_ip == 20 else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
