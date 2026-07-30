import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass
from src.config import EVENT_WINDOW, ESTIMATION_WINDOW


@dataclass
class EventStudyResult:
    asset: str
    event_dates: list[str]
    aar: pd.Series
    car: pd.Series
    aar_t_stats: pd.Series
    aar_p_values: pd.Series
    car_p_value: float
    hit_rate: float
    positive_ratio: float


def estimate_normal_returns(
    prices: pd.Series,
    event_date: pd.Timestamp,
    est_window: tuple[int, int],
) -> float:
    start = event_date + pd.Timedelta(days=est_window[0])
    end = event_date + pd.Timedelta(days=est_window[1])
    window = prices.loc[start:end]
    if len(window) < 10:
        return 0.0
    returns = window.pct_change().dropna()
    return returns.mean()


def run_event_study(
    prices: pd.DataFrame,
    event_dates: list[str | pd.Timestamp],
    event_window: tuple[int, int] = EVENT_WINDOW,
    est_window: tuple[int, int] = ESTIMATION_WINDOW,
) -> dict[str, EventStudyResult]:
    results = {}

    if prices.empty or prices.columns.empty:
        return results

    for col in prices.columns:
        series = prices[col].dropna()
        ser_returns = series.pct_change().dropna()

        all_abnormal = []

        for evt_str in event_dates:
            evt = pd.Timestamp(evt_str)
            mu = estimate_normal_returns(series, evt, est_window)

            w_start = evt + pd.Timedelta(days=event_window[0])
            w_end = evt + pd.Timedelta(days=event_window[1])

            window_returns = ser_returns.loc[w_start:w_end]
            abnormal = window_returns - mu
            if not abnormal.empty:
                all_abnormal.append(abnormal)

        if not all_abnormal:
            continue

        aligned_relative = []
        for abnormal in all_abnormal:
            rel = pd.Series(
                abnormal.values,
                index=range(event_window[0], event_window[0] + len(abnormal)),
            )
            aligned_relative.append(rel)

        min_len = min(len(a) for a in aligned_relative)
        trimmed = [a.iloc[:min_len] for a in aligned_relative]

        abnormal_matrix = pd.concat(trimmed, axis=1)
        abnormal_matrix.columns = [str(d) for d in event_dates[: len(trimmed)]]

        aar = abnormal_matrix.mean(axis=1)
        car = aar.cumsum()

        n_events = abnormal_matrix.shape[1]
        aar_std = abnormal_matrix.std(axis=1, ddof=1) / np.sqrt(n_events)
        aar_std = aar_std.replace(0, np.nan)

        aar_t_stats = aar / aar_std
        aar_p_values = 2 * (1 - stats.t.cdf(np.abs(aar_t_stats), df=n_events - 1))

        car_std = aar_std * np.sqrt(len(aar))
        car_t = car.iloc[-1] / car_std.iloc[-1] if car_std.iloc[-1] > 0 else 0
        car_p_value = 2 * (1 - stats.t.cdf(np.abs(car_t), df=n_events - 1))

        final_abnormal = abnormal_matrix.iloc[-1]
        hit_rate = (final_abnormal > 0).mean()
        positive_ratio = (final_abnormal > 0).sum()

        results[col] = EventStudyResult(
            asset=col,
            event_dates=list(abnormal_matrix.columns),
            aar=aar,
            car=car,
            aar_t_stats=aar_t_stats,
            aar_p_values=aar_p_values,
            car_p_value=car_p_value,
            hit_rate=hit_rate,
            positive_ratio=positive_ratio,
        )

    return results


def summarize_results(results: dict[str, EventStudyResult]) -> pd.DataFrame:
    cols = [
        "asset", "car_event_window", "car_p_value", "hit_rate",
        "num_events", "aar_peak", "aar_trough", "vol_abnormal",
    ]
    if not results:
        return pd.DataFrame(columns=cols)
    rows = []
    for asset, res in results.items():
        rows.append({
            "asset": asset,
            "car_event_window": round(res.car.iloc[-1], 4),
            "car_p_value": round(res.car_p_value, 4),
            "hit_rate": round(res.hit_rate, 3),
            "num_events": len(res.event_dates),
            "aar_peak": round(res.aar.max(), 4),
            "aar_trough": round(res.aar.min(), 4),
            "vol_abnormal": round(res.aar.std(), 4),
        })
    if not rows:
        return pd.DataFrame(columns=cols)
    try:
        return pd.DataFrame(rows).sort_values("car_event_window", key=abs, ascending=False)
    except KeyError:
        return pd.DataFrame(rows, columns=cols)
