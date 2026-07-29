import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from src.config import PORTFOLIO_CONFIG, ASSET_CATEGORIES

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    weights: pd.DataFrame
    portfolio_returns: pd.Series
    cumulative_returns: pd.Series
    turnover: pd.Series
    transaction_costs: pd.Series
    gross_returns: pd.Series
    regime_history: pd.Series | None = None


def risk_parity_weights(cov: pd.DataFrame) -> np.ndarray:
    """
    Compute equal risk contribution (risk parity) weights.
    Uses the iterative approach.
    """
    n = cov.shape[0]
    try:
        inv_vol = 1.0 / np.sqrt(np.diag(cov))
        w = inv_vol / inv_vol.sum()
        for _ in range(100):
            mrc = (cov @ w) * w
            target = mrc.mean()
            w = w * (target / mrc)
            w = np.clip(w, 0.0, 1.0)
            w = w / w.sum()
            if np.max(np.abs(mrc - target)) < 1e-6:
                break
        return w
    except np.linalg.LinAlgError:
        return np.ones(n) / n


def regime_weights(
    regime_label: str,
    risky: pd.DataFrame,
    safe: pd.DataFrame,
    commodities: pd.DataFrame,
) -> dict[str, float]:
    """
    Regime-dependent weight templates based on the MENA risk-aware strategy.
    - Normal: standard risk parity
    - Elevated: tilt to commodities (oil/gold), reduce equities
    - Crisis: overweight oil, gold, USD; underweight equities, bonds
    """
    n_risky = len(risky.columns)
    n_safe = len(safe.columns)
    n_commod = len(commodities.columns)
    total = n_risky + n_safe + n_commod

    if regime_label == "Normal":
        w_risky = 0.35
        w_safe = 0.40
        w_commod = 0.25
    elif regime_label == "Elevated Risk":
        w_risky = 0.20
        w_safe = 0.35
        w_commod = 0.45
    elif regime_label == "Crisis":
        w_risky = 0.10
        w_safe = 0.30
        w_commod = 0.60
    else:
        w_risky = 0.35
        w_safe = 0.40
        w_commod = 0.25

    weights = {}
    if n_risky > 0:
        for c in risky.columns:
            weights[c] = w_risky / n_risky
    if n_safe > 0:
        for c in safe.columns:
            weights[c] = w_safe / n_safe
    if n_commod > 0:
        for c in commodities.columns:
            weights[c] = w_commod / n_commod

    total_w = sum(weights.values())
    if total_w > 0:
        for k in weights:
            weights[k] /= total_w

    return weights


def run_regime_backtest(
    prices: pd.DataFrame,
    regimes: pd.Series | None = None,
    regime_labels: pd.Series | None = None,
    start: str = "2005-01-01",
    end: str | None = None,
    vol_target: float = PORTFOLIO_CONFIG["vol_target"],
    rebalance_freq: str = PORTFOLIO_CONFIG["rebalance_freq"],
    transaction_cost: float = PORTFOLIO_CONFIG["transaction_cost"],
    vol_lookback: int = 60,
) -> BacktestResult:
    prices = prices.loc[start:end] if end else prices.loc[start:]
    returns = prices.pct_change().dropna()

    risky_assets = [c for c in prices.columns if c in ASSET_CATEGORIES["risky"]]
    safe_assets = [c for c in prices.columns if c in ASSET_CATEGORIES["safe"]]
    commod_assets = [c for c in prices.columns if c in ASSET_CATEGORIES["commodities"]]
    available_cols = [c for c in prices.columns if c in returns.columns]
    risky = returns[[c for c in risky_assets if c in available_cols]]
    safe = returns[[c for c in safe_assets if c in available_cols]]
    commod = returns[[c for c in commod_assets if c in available_cols]]

    rebase_dates = returns.resample(rebalance_freq).apply(
        lambda x: x.index[0] if len(x) else None
    ).dropna()

    weight_records = []
    port_rets = []
    gross_rets = []

    for i, (current_date, current_returns) in enumerate(returns.iterrows()):
        if regimes is not None and regime_labels is not None:
            label = regime_labels.loc[current_date] if current_date in regime_labels.index else "Normal"
        else:
            label = "Normal"

        if current_date in rebase_dates.values or len(weight_records) == 0:
            cov_window = returns.loc[:current_date].tail(252).dropna()
            if len(cov_window) > 20:
                cov = cov_window.cov() * 252
                try:
                    rp_w = risk_parity_weights(cov.loc[available_cols, available_cols])
                    base_weights = pd.Series(rp_w, index=available_cols)
                except Exception:
                    base_weights = pd.Series(1.0 / len(available_cols), index=available_cols)
            else:
                base_weights = pd.Series(1.0 / len(available_cols), index=available_cols)

            rw = regime_weights(label, risky, safe, commod)
            for col in available_cols:
                if col in rw:
                    base_weights[col] = rw[col]

            base_weights = base_weights / base_weights.sum()

        gross_ret = (current_returns[available_cols] * base_weights[available_cols]).sum()
        gross_rets.append(gross_ret)
        weight_records.append(base_weights.to_dict())
        port_rets.append(gross_ret)

    port_df = pd.DataFrame(port_rets, index=returns.index, columns=["portfolio_return"])
    weights_df = pd.DataFrame(weight_records, index=returns.index)
    gross_series = pd.Series(gross_rets, index=returns.index)

    turnover = weights_df.diff().abs().sum(axis=1).fillna(0.0)
    tc = turnover * transaction_cost
    net_returns = port_df["portfolio_return"] - tc
    port_df["portfolio_return"] = net_returns

    trailing_vol = net_returns.rolling(vol_lookback, min_periods=20).std().clip(lower=1e-8)
    scaling_factor = vol_target / (trailing_vol * np.sqrt(252))
    port_df["portfolio_return"] = net_returns * scaling_factor.clip(upper=3.0)

    cum_ret = (1 + port_df["portfolio_return"]).cumprod()

    logger.info(
        "Backtest complete — Sharpe ex-ante vol target: %.0f%%, TC: %.4f bps/day",
        vol_target * 100, tc.mean() * 10000,
    )

    return BacktestResult(
        weights=weights_df,
        portfolio_returns=port_df["portfolio_return"],
        cumulative_returns=cum_ret,
        turnover=turnover,
        transaction_costs=tc,
        gross_returns=gross_series,
        regime_history=regime_labels if regime_labels is not None else None,
    )


def benchmark_returns(
    prices: pd.DataFrame,
    weights: dict[str, float] | None = None,
    start: str = "2005-01-01",
    end: str | None = None,
) -> pd.Series:
    prices = prices.loc[start:end] if end else prices.loc[start:]
    returns = prices.pct_change().dropna()

    if weights is None:
        cols = returns.columns.tolist()
        n = len(cols)
        weights = {c: 1.0 / n for c in cols}

    port = pd.Series(0.0, index=returns.index)
    for col, w in weights.items():
        if col in returns.columns:
            port += returns[col] * w

    return port
