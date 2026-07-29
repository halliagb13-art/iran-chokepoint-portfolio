import logging
import pandas as pd
import numpy as np
import requests
from io import StringIO
from pathlib import Path
from src.config import GPR_PUBLISHED_URL, GPR_DIR
from src.gpr.base import GPRAdapter
from src.events import EVENTS_DF

logger = logging.getLogger(__name__)


def _build_synthetic_gpr(start: str = "2000-01-01", end: str | None = None) -> pd.Series:
    """
    Build a synthetic GPR index from the curated event database.
    Each event contributes a severity-weighted spike that decays over 60 days.
    Used when the published GPR data is unreachable.
    """
    dates = pd.date_range(start, end or pd.Timestamp.today(), freq="D")
    gpr = pd.Series(0.0, index=dates, name="gpr_synthetic")

    for _, evt in EVENTS_DF.iterrows():
        evt_date = pd.Timestamp(evt["date"])
        if evt_date < gpr.index[0] or evt_date > gpr.index[-1]:
            continue
            severity = evt["severity"] / 5.0
            offsets = np.arange(60)
            decays = np.exp(-offsets / 15.0)
            event_dates = evt_date + pd.to_timedelta(offsets, unit="D")
            mask = event_dates.isin(gpr.index)
            gpr.loc[event_dates[mask]] += severity * decays[mask] * 100

    gpr += np.random.default_rng(42).normal(0, 5, len(gpr))
    gpr = gpr.clip(lower=0)
    return gpr


class PublishedGPRAdapter(GPRAdapter):

    def __init__(self, use_mideast: bool = True):
        self._use_mideast = use_mideast
        self._cached: pd.Series | None = None

    def name(self) -> str:
        return "Caldara-Iacoviello Published GPR" + (
            " (MENA)" if self._use_mideast else " (Global)"
        )

    def fetch_index(self, start: str = "2000-01-01", end: str | None = None) -> pd.Series:
        if self._cached is not None:
            return self._cached

        local_path = GPR_DIR / "gpr_data.csv"
        df = None

        if local_path.exists():
            df = pd.read_csv(local_path)
        else:
            try:
                resp = requests.get(GPR_PUBLISHED_URL, timeout=10)
                resp.raise_for_status()
                df = pd.read_csv(StringIO(resp.text))
                local_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(local_path, index=False)
            except Exception as e:
                logger.warning("GPR download failed (%s). Falling back to synthetic index.", e)

        if df is not None:
            df.columns = [c.strip().lower() for c in df.columns]
            if self._use_mideast and "gpr_mideast" in df.columns:
                col = "gpr_mideast"
            elif "gpr" in df.columns:
                col = "gpr"
            else:
                col = df.columns[1]

            if "date" in df.columns:
                dates = pd.to_datetime(df["date"], errors="coerce")
            elif "year" in df.columns and "month" in df.columns:
                dates = pd.to_datetime(
                    df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01"
                )
            else:
                raise KeyError("No date column found in GPR data.")

            series = pd.Series(df[col].values, index=dates, name="gpr")
            series = series.sort_index()
        else:
            series = _build_synthetic_gpr(start=start, end=end)

        series = series[series.index >= start]
        if end:
            series = series[series.index <= end]
        series = series.astype(float)
        self._cached = series
        return series
