"""Read local Statcast parquets and surface the minimal data the model needs.

Parquets live at data/raw/pitches_<year>.parquet. Each row is one pitch.
Columns we care about: pitcher_id, year, launch_speed, launch_angle,
xwoba_value (= Statcast estimated_woba_using_speedangle), event_type.

`load_batted_balls`: one row per BBE with EV/LA/xwoba_value.
`pitcher_season_xwobacon`: per (pitcher_id, year): n_bip, xwobacon.
`pitcher_season_pa_rates`: per (pitcher_id, year): n_pa, k_pct, bb_pct, hr_pct.
`pitcher_season_ip`: per (pitcher_id, year): outs, ip (approx).
"""
from __future__ import annotations

from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
ALL_YEARS = tuple(range(2016, 2026))

KEEP_COLS = ["pitcher_id", "year", "game_date", "ab_number", "index_play",
             "pitch_type", "launch_speed", "launch_angle",
             "event_type", "xwoba_value"]


def load_batted_balls(years: tuple[int, ...] = ALL_YEARS) -> pl.DataFrame:
    """One row per batted ball (launch_speed, launch_angle, xwoba_value all
    non-null). Statcast only emits xwoba_value for BBE.
    """
    missing = [y for y in years if not (RAW / f"pitches_{y}.parquet").exists()]
    if missing:
        raise FileNotFoundError(
            f"missing parquets for years {missing}. "
            f"Run `uv run python src/fetch_savant.py {' '.join(map(str, missing))}`."
        )
    lf = pl.concat(
        [
            pl.scan_parquet(RAW / f"pitches_{y}.parquet")
            .select(KEEP_COLS)
            .filter(
                pl.col("launch_speed").is_not_null()
                & pl.col("launch_angle").is_not_null()
                & pl.col("xwoba_value").is_not_null()
            )
            for y in years
        ],
        how="vertical",
    )
    return lf.collect()


def pitcher_season_xwobacon(bb: pl.DataFrame, min_bip: int = 1) -> pl.DataFrame:
    """Per (pitcher_id, year): n_bip, xwobacon (mean xwoba_value)."""
    return (
        bb.group_by("pitcher_id", "year")
        .agg(
            pl.len().alias("n_bip"),
            pl.col("xwoba_value").mean().alias("xwobacon"),
        )
        .filter(pl.col("n_bip") >= min_bip)
        .sort("pitcher_id", "year")
    )


# event_type -> outs recorded. event_type is populated only on the pitch
# that ends a plate appearance, so each non-null row represents one PA.
OUTS_PER_EVENT = {
    "strikeout":                 1,
    "strikeout_double_play":     2,
    "strikeout_triple_play":     3,
    "field_out":                 1,
    "force_out":                 1,
    "fielders_choice_out":       1,
    "fielders_choice":           0,
    "grounded_into_double_play": 2,
    "double_play":               2,
    "triple_play":               3,
    "sac_fly":                   1,
    "sac_fly_double_play":       2,
    "sac_bunt":                  1,
    "sac_bunt_double_play":      2,
    "other_out":                 1,
    # Hits, walks, HBP, errors -> 0 outs (default below).
}


PA_K_EVENTS  = ("strikeout", "strikeout_double_play", "strikeout_triple_play")
PA_BB_EVENTS = ("walk", "intent_walk")


def pitcher_season_pa_rates(years: tuple[int, ...] = ALL_YEARS) -> pl.DataFrame:
    """Per (pitcher_id, year): n_pa, k_pct, bb_pct, hr_pct.

    PA = count of non-null event_type rows (one per plate appearance).
    Intentional walks count toward BB%; HBP does not.
    """
    frames = [
        pl.scan_parquet(RAW / f"pitches_{y}.parquet")
        .select("pitcher_id", "year", "event_type")
        .filter(pl.col("event_type").is_not_null())
        for y in years
    ]
    return (
        pl.concat(frames, how="vertical")
        .collect()
        .group_by("pitcher_id", "year")
        .agg(
            pl.len().alias("n_pa"),
            pl.col("event_type").is_in(list(PA_K_EVENTS)).sum().alias("n_k"),
            pl.col("event_type").is_in(list(PA_BB_EVENTS)).sum().alias("n_bb"),
            (pl.col("event_type") == "home_run").sum().alias("n_hr"),
        )
        .with_columns(
            (pl.col("n_k")  / pl.col("n_pa")).alias("k_pct"),
            (pl.col("n_bb") / pl.col("n_pa")).alias("bb_pct"),
            (pl.col("n_hr") / pl.col("n_pa")).alias("hr_pct"),
        )
        .select("pitcher_id", "year", "n_pa", "k_pct", "bb_pct", "hr_pct")
        .sort("pitcher_id", "year")
    )


def pitcher_season_ip(years: tuple[int, ...] = ALL_YEARS) -> pl.DataFrame:
    """Per (pitcher_id, year): outs and IP from event_type counts.
    Ignores pickoff/CS credited to the pitcher (rare).
    """
    frames = [
        pl.scan_parquet(RAW / f"pitches_{y}.parquet")
        .select("pitcher_id", "year", "event_type")
        .filter(pl.col("event_type").is_not_null())
        for y in years
    ]
    outs_map = pl.DataFrame(
        {"event_type": list(OUTS_PER_EVENT.keys()),
         "outs":       list(OUTS_PER_EVENT.values())},
        schema={"event_type": pl.Utf8, "outs": pl.Int32},
    )
    return (
        pl.concat(frames, how="vertical")
        .collect()
        .join(outs_map, on="event_type", how="left")
        .with_columns(pl.col("outs").fill_null(0))
        .group_by("pitcher_id", "year")
        .agg(pl.col("outs").sum().alias("outs"))
        .with_columns((pl.col("outs") / 3.0).alias("ip"))
        .sort("pitcher_id", "year")
    )
