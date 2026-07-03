"""Day-by-day Statcast scraper -> data/raw/pitches_<year>.parquet.

Pulls Baseball Savant's per-pitch CSV in `type=details` mode and slims it to
the columns we model on: pitcher_id, year, launch_speed, launch_angle,
xwoba_value, event_type, pitch_type. Threaded (6 workers) with retry/backoff.

Usage:
    uv run python src/fetch_savant.py             # 2016-2025
    uv run python src/fetch_savant.py 2024 2025   # specific years
"""
from __future__ import annotations

import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

ENDPOINT = "https://baseballsavant.mlb.com/statcast_search/csv"
N_WORKERS = 6
RETRIES = 4
BACKOFF = 2.0

SEASON_DATES = {
    # (start, end) in YYYY-MM-DD; covers reg + post.
    2016: ("2016-04-03", "2016-11-02"),
    2017: ("2017-04-02", "2017-11-01"),
    2018: ("2018-03-29", "2018-10-28"),
    2019: ("2019-03-20", "2019-10-30"),
    2020: ("2020-07-23", "2020-10-27"),
    2021: ("2021-04-01", "2021-11-02"),
    2022: ("2022-04-07", "2022-11-05"),
    2023: ("2023-03-30", "2023-11-01"),
    2024: ("2024-03-20", "2024-10-30"),
    2025: ("2025-03-18", "2025-11-01"),
}

KEEP = ["pitcher", "game_year", "launch_speed", "launch_angle",
        "estimated_woba_using_speedangle", "events", "pitch_type"]


def daterange(start: str, end: str):
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    n = (d1 - d0).days + 1
    for i in range(n):
        yield (d0 + timedelta(days=i)).isoformat()


def fetch_day(day: str, season: int) -> pl.DataFrame | None:
    """One day of pitch-level CSV. Returns empty/None on truly empty days."""
    params = {
        "all": "true",
        "hfSea": f"{season}|",
        "player_type": "pitcher",
        "game_date_gt": day,
        "game_date_lt": day,
        "type": "details",
        "min_pitches": "0", "min_results": "0", "min_pas": "0",
        "group_by": "name", "sort_col": "pitches",
        "sort_order": "desc",
    }
    last_err: Exception | None = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(ENDPOINT, params=params, timeout=60)
            r.raise_for_status()
            text = r.text.strip()
            if not text or text.startswith("<"):
                return None
            df = pl.read_csv(io.StringIO(text), ignore_errors=True,
                              null_values=["null", "NA", ""])
            if df.is_empty():
                return None
            return df.select([c for c in KEEP if c in df.columns])
        except Exception as e:
            last_err = e
            time.sleep(BACKOFF * (attempt + 1))
    print(f"  ! {day} failed after {RETRIES} retries: {last_err}",
          file=sys.stderr)
    return None


def fetch_year(year: int) -> None:
    start, end = SEASON_DATES[year]
    days = list(daterange(start, end))
    print(f"[{year}] {len(days)} days, {start} .. {end}")

    frames: list[pl.DataFrame] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(fetch_day, d, year): d for d in days}
        done = 0
        for fut in as_completed(futures):
            df = fut.result()
            done += 1
            if df is not None and not df.is_empty():
                frames.append(df)
            if done % 25 == 0:
                print(f"  [{year}] {done}/{len(days)} "
                      f"({time.time()-t0:.0f}s)", file=sys.stderr, flush=True)

    if not frames:
        print(f"[{year}] no data, skipping write", file=sys.stderr)
        return

    df = pl.concat(frames, how="diagonal_relaxed")
    # Normalise schema: pitcher -> pitcher_id, game_year -> year,
    # estimated_woba_using_speedangle -> xwoba_value, events -> event_type.
    df = df.rename({
        "pitcher": "pitcher_id",
        "game_year": "year",
        "estimated_woba_using_speedangle": "xwoba_value",
        "events": "event_type",
    }).with_columns(
        pl.col("pitcher_id").cast(pl.Int64),
        pl.col("year").cast(pl.Int32),
        pl.col("launch_speed").cast(pl.Float64),
        pl.col("launch_angle").cast(pl.Float64),
        pl.col("xwoba_value").cast(pl.Float64),
        pl.col("pitch_type").cast(pl.Utf8),
        pl.col("event_type").cast(pl.Utf8),
    )

    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"pitches_{year}.parquet"
    df.write_parquet(out)
    print(f"[{year}] wrote {out.name}: {df.height:,} rows ({time.time()-t0:.0f}s)")


def main() -> int:
    years = [int(a) for a in sys.argv[1:]] or list(SEASON_DATES.keys())
    for y in years:
        if y not in SEASON_DATES:
            print(f"unknown year {y}; known: {sorted(SEASON_DATES)}",
                  file=sys.stderr)
            return 1
    for y in years:
        fetch_year(y)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
