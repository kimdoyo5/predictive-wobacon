"""Shared train/val/test split + min-of-pair-weighted metrics.

TEST is held out chronologically: N = 2024 → 2024->2025 target.
TRAIN/VAL pool is N in 2016..2023, then split RANDOMLY by (pitcher, year)
pair into train (80%) and val (20%) with a fixed seed.

Filters:
    train, val:  min(n_bip, n_bip_next) >= MIN_BIP_TRAINVAL (env-tunable).
    test:        min(ip, ip_next) >= MIN_IP.

Weights (stored in the `n_bip_next` column for downstream compatibility):
    train, val:  min(n_bip, n_bip_next).
    test:        min(ip, ip_next).

load_splits() returns (train, val, test, test_next):
    train, val, test:  year-N BBE rows with target/IP columns attached.
    test_next:         year-(N+1) BBE rows for the same test pitchers (used
                       to compute year-N+1 metrics + model autocorrelation).
"""
from __future__ import annotations

import os
import numpy as np
import polars as pl

from data import (load_batted_balls, pitcher_season_xwobacon,
                  pitcher_season_ip)

TRAINVAL_N_YEARS = (2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023)
TEST_N_YEAR      = 2024

VAL_FRAC = 0.20
SEED     = 42

MIN_IP           = 30
MIN_BIP_TRAINVAL = int(os.environ.get("MIN_BIP_TRAINVAL", "1"))


def load_splits(min_ip: int = MIN_IP) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    bb = load_batted_balls()
    ps = pitcher_season_xwobacon(bb, min_bip=1)         # no filter here
    ip = pitcher_season_ip().select("pitcher_id", "year", "ip")

    # ps + ip per (pitcher, year)
    ps_ip = ps.join(ip, on=["pitcher_id", "year"], how="left").with_columns(
        pl.col("ip").fill_null(0.0)
    )

    # Build (N, N+1) pairs carrying ip and ip_next.
    nxt = (ps_ip.with_columns((pl.col("year") - 1).alias("year"))
                .rename({"n_bip": "n_bip_next",
                         "xwobacon": "xwobacon_next",
                         "ip": "ip_next"}))
    pairs = ps_ip.join(nxt, on=["pitcher_id", "year"], how="inner")
    # pairs: pitcher_id, year(N), n_bip, xwobacon, ip,
    #                              n_bip_next, xwobacon_next, ip_next

    bip_ok = pl.min_horizontal("n_bip", "n_bip_next") >= MIN_BIP_TRAINVAL
    ip_ok  = pl.min_horizontal("ip", "ip_next") >= min_ip
    trainval  = pairs.filter(pl.col("year").is_in(list(TRAINVAL_N_YEARS)) & bip_ok)
    test_keys = pairs.filter((pl.col("year") == TEST_N_YEAR) & ip_ok)

    # Random 80/20 split of trainval by (pitcher, year).
    rng = np.random.default_rng(SEED)
    n_pool = trainval.height
    perm = rng.permutation(n_pool)
    n_val = int(round(n_pool * VAL_FRAC))
    val_keys   = trainval[np.sort(perm[:n_val])].select("pitcher_id", "year")
    train_keys = trainval[np.sort(perm[n_val:])].select("pitcher_id", "year")

    # Attach target + weight to per-event BBE rows. The `n_bip_next` column
    # stores the *weight* (downstream code reads it as the row weight):
    #   train/val: min(n_bip, n_bip_next) — effective-sample weighting.
    #   test:      min(ip, ip_next)       — leaderboard metric weight.
    target_cols_trainval = pairs.select(
        "pitcher_id", "year", "xwobacon_next",
        pl.min_horizontal("n_bip", "n_bip_next").alias("n_bip_next"),
        "ip", "ip_next",
    )
    target_cols_test = pairs.select(
        "pitcher_id", "year", "xwobacon_next",
        pl.min_horizontal("ip", "ip_next").alias("n_bip_next"),
        "ip", "ip_next",
    )
    bbe_trainval = bb.join(target_cols_trainval, on=["pitcher_id", "year"], how="inner")
    bbe_test     = bb.join(target_cols_test,     on=["pitcher_id", "year"], how="inner")

    train = bbe_trainval.join(train_keys, on=["pitcher_id", "year"], how="inner")
    val   = bbe_trainval.join(val_keys,   on=["pitcher_id", "year"], how="inner")
    test  = bbe_test.join(test_keys.select("pitcher_id", "year"),
                          on=["pitcher_id", "year"], how="inner")

    # test_next: year-(N+1)=2025 BBE for the SAME test pitchers.
    test_pitcher_ids = test_keys.select("pitcher_id")
    test_next = (bb.filter(pl.col("year") == TEST_N_YEAR + 1)
                   .join(test_pitcher_ids, on="pitcher_id", how="inner"))
    return train, val, test, test_next


# --- weighted metrics ---

def weighted_rmse(y_true, y_pred, w) -> float:
    err = y_true - y_pred
    return float(np.sqrt((w * err * err).sum() / w.sum()))


def weighted_r2(y_true, y_pred, w) -> float:
    ybar = (w * y_true).sum() / w.sum()
    ss_res = (w * (y_true - y_pred) ** 2).sum()
    ss_tot = (w * (y_true - ybar) ** 2).sum()
    return float(1.0 - ss_res / ss_tot)


def weighted_corr(x, y, w) -> float:
    sw = w.sum()
    xbar = (w * x).sum() / sw
    ybar = (w * y).sum() / sw
    cov  = (w * (x - xbar) * (y - ybar)).sum() / sw
    vx   = (w * (x - xbar) ** 2).sum() / sw
    vy   = (w * (y - ybar) ** 2).sum() / sw
    denom = np.sqrt(vx * vy)
    return float(cov / denom) if denom > 0 else float("nan")
