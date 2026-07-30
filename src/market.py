import pandas as pd
import numpy as np
import logging
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from src.config import ASSETS, MARKET_DIR, ASSET_CATEGORIES

logger = logging.getLogger(__name__)

CACHE_MAX_AGE_HOURS = 24


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(hours=CACHE_MAX_AGE_HOURS)


def fetch_all_assets(
    start: str = "1985-01-01",
    end: str | None = None,
    save: bool = True,
    category: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    keys = (
        list(ASSETS.keys())
        if category is None
        else ASSET_CATEGORIES.get(category, [])
    )

    cache_paths = {k: MARKET_DIR / f"{k}.parquet" for k in keys}
    cached = {
        k for k, p in cache_paths.items()
        if not force and _cache_is_fresh(p)
    }

    if cached:
        logger.info("Using cached data for %d/%d assets", len(cached), len(keys))

    if save:
        MARKET_DIR.mkdir(parents=True, exist_ok=True)

    needs_fetch = [k for k in keys if k not in cached]
    frames = {}

    for key in cached:
        df = pd.read_parquet(cache_paths[key])
        s = df.iloc[:, 0]
        s.name = key
        frames[key] = s

    if needs_fetch:
        tickers = [ASSETS[k]["ticker"] for k in needs_fetch]
        logger.info("Fetching %d assets via yfinance: %s", len(tickers), tickers)
        end = end or datetime.today().strftime("%Y-%m-%d")

        data = yf.download(
            " ".join(tickers),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )

        if data.empty:
            logger.warning("yfinance returned no data")
            return pd.DataFrame()

        for key in needs_fetch:
            ticker = ASSETS[key]["ticker"]
            try:
                if ticker in data.columns.get_level_values(0):
                    series = data[ticker]["Close"].squeeze().dropna()
                elif "Close" in data.columns:
                    series = data["Close"].squeeze().dropna()
                else:
                    logger.warning("No Close column for %s (%s)", ticker, key)
                    continue

                series.name = key
                series.index = pd.to_datetime(series.index)

                frames[key] = series

                if save:
                    try:
                        series.to_frame().to_parquet(cache_paths[key])
                    except Exception as e:
                        logger.warning("Failed to cache %s (%s): %s", ticker, key, e)
            except Exception as e:
                logger.warning("Failed to parse %s (%s): %s", ticker, key, e)

    if not frames:
        logger.error("No market data loaded")
        return pd.DataFrame()

    result = pd.DataFrame(frames)
    result = result.dropna(how="all").sort_index()
    logger.info("Market data: %d assets × %d rows", len(result.columns), len(result))
    return result


def compute_returns(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    if freq == "D":
        return df.pct_change().dropna()
    elif freq == "W":
        return df.resample("W").last().pct_change().dropna()
    elif freq == "ME":
        return df.resample("ME").last().pct_change().dropna()
    else:
        return df.pct_change().dropna()


def load_saved_assets(category: str | None = None) -> pd.DataFrame:
    pattern = list(ASSETS.keys())
    if category:
        pattern = ASSET_CATEGORIES.get(category, pattern)

    frames = {}
    for key in pattern:
        path = MARKET_DIR / f"{key}.parquet"
        if path.exists():
            s = pd.read_parquet(path).iloc[:, 0]
            s.name = key
            frames[key] = s
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def align_to_events(
    prices: pd.DataFrame,
    event_dates: list[str],
    window: tuple[int, int] = (-5, 10),
) -> dict:
    results = {}
    for evt_date in event_dates:
        dt = pd.Timestamp(evt_date)
        start = dt + pd.Timedelta(days=window[0])
        end = dt + pd.Timedelta(days=window[1])
        mask = (prices.index >= start) & (prices.index <= end)
        window_data = prices.loc[mask].copy()
        if not window_data.empty:
            window_data = window_data / window_data.iloc[0] - 1
        results[evt_date] = window_data
    return results


if __name__ == "__main__":
    print("Fetching all assets (this may take a minute)...")
    df = fetch_all_assets(start="2000-01-01")
    print(f"Fetched {len(df.columns)} assets with {len(df)} rows")
    print(df.head())
