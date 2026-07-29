import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    total_return: float
    annualised_return: float
    annualised_vol: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    hit_ratio: float
    asymmetry: float
    breadth_estimate: float
    var_95: float
    cvar_95: float
    skewness: float
    kurtosis: float


def _effective_bets(returns: pd.Series, n_lags: int = 1) -> float:
    """
    Estimate the effective number of independent bets (breadth) per year
    using the inverse of average pairwise serial correlation.

    In the Grinold-Kahn framework, breadth is the number of independent
    forecasts per year.  Here we compute it as:

        BR = 1 / avg(|rho|)

    where rho is the average pairwise absolute autocorrelation of
    monthly non-overlapping returns.  Higher autocorrelation → fewer
    independent bets.
    """
    if not isinstance(returns.index, pd.DatetimeIndex):
        return 6.0

    monthly = returns.resample("ME").sum()
    if len(monthly) < 12:
        return 1.0

    autocorrs = []
    for lag in range(1, min(n_lags + 1, len(monthly) // 4)):
        ac = monthly.autocorr(lag=lag)
        if not np.isnan(ac):
            autocorrs.append(abs(ac))

    if not autocorrs:
        return 12.0

    avg_abs_corr = np.mean(autocorrs)
    return max(1.0, min(1.0 / max(avg_abs_corr, 0.01), 52.0))


def compute_metrics(returns: pd.Series, risk_free_rate: float = 0.05) -> PerformanceMetrics:
    returns = returns.dropna()
    if len(returns) < 5:
        return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    ann_vol = returns.std() * np.sqrt(252)

    rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1
    excess = returns - rf_daily
    sharpe = (excess.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = (ann_ret - risk_free_rate) / downside_std if downside_std > 0 else 0.0

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min()

    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    hit_ratio = (returns > 0).mean()

    pos_returns = returns[returns > 0]
    neg_returns = returns[returns < 0]
    avg_gain = pos_returns.mean() if len(pos_returns) > 0 else 0
    avg_loss = abs(neg_returns.mean()) if len(neg_returns) > 0 else 1e-6
    asymmetry = avg_gain / avg_loss if avg_loss > 0 else 1.0

    breadth_estimate = _effective_bets(returns)

    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    return PerformanceMetrics(
        total_return=round(total_ret, 4),
        annualised_return=round(ann_ret, 4),
        annualised_vol=round(ann_vol, 4),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        calmar_ratio=round(calmar, 4),
        max_drawdown=round(max_dd, 4),
        hit_ratio=round(hit_ratio, 4),
        asymmetry=round(asymmetry, 4),
        breadth_estimate=round(breadth_estimate, 2),
        var_95=round(var_95, 6),
        cvar_95=round(cvar_95, 6),
        skewness=round(skewness, 4),
        kurtosis=round(kurtosis, 4),
    )


def compare_strategies(
    strategy_returns: dict[str, pd.Series],
) -> pd.DataFrame:
    rows = []
    for name, rets in strategy_returns.items():
        m = compute_metrics(rets)
        rows.append({"strategy": name, **m.__dict__})
    return pd.DataFrame(rows).set_index("strategy").T


def table_to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(floatfmt=".4f")
