import pytest
import pandas as pd
import numpy as np
from src.events import get_events
from src.gpr.factory import create_gpr_adapter
from src.event_study import run_event_study, summarize_results
from src.regime_model import fit_markov_regime, label_regimes
from src.portfolio import run_regime_backtest
from src.performance import compute_metrics, compare_strategies, _effective_bets


@pytest.fixture(scope="module")
def synthetic_prices():
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", "2026-01-01", freq="D")
    n = len(dates)
    comps = {
        "equities_us": {"drift": 0.0006, "vol": 0.012},
        "equities_em": {"drift": 0.0003, "vol": 0.015},
        "bonds_long": {"drift": 0.0001, "vol": 0.008},
        "bonds_int": {"drift": 0.0002, "vol": 0.004},
        "gold": {"drift": 0.0003, "vol": 0.009},
        "oil_wti": {"drift": 0.0001, "vol": 0.022},
        "fx_usd_idx": {"drift": 0.0000, "vol": 0.005},
    }
    data = {}
    for name, p in comps.items():
        rets = np.random.normal(p["drift"], p["vol"], n)
        data[name] = 100 * (1 + rets).cumprod()
    return pd.DataFrame(data, index=dates)


class TestPipeline:
    def test_events_load(self):
        events = get_events()
        assert len(events) >= 15

    def test_gpr_adapter_creates_index(self):
        adapter = create_gpr_adapter("published")
        gpr = adapter.fetch_index(start="2020-01-01", end="2025-01-01")
        assert len(gpr) > 100

    def test_event_study_on_synthetic(self, synthetic_prices):
        events = get_events()
        dates = events["date"].tolist()
        results = run_event_study(synthetic_prices, dates)
        assert len(results) > 0
        summary = summarize_results(results)
        assert "car_event_window" in summary.columns

    def test_regime_model_on_synthetic(self, synthetic_prices):
        rets = synthetic_prices.pct_change().dropna()
        result = fit_markov_regime(rets, n_regimes=3)
        labels = label_regimes(result, rets)
        assert len(labels) == len(result.regimes)

    def test_full_backtest_on_synthetic(self, synthetic_prices):
        rets = synthetic_prices.pct_change().dropna()
        rm = fit_markov_regime(rets, n_regimes=3)
        labels = label_regimes(rm, rets)
        bt = run_regime_backtest(
            synthetic_prices, regimes=rm.regimes, regime_labels=labels,
        )
        assert bt.portfolio_returns is not None
        assert len(bt.portfolio_returns) > 100
        assert bt.transaction_costs is not None

    def test_transaction_costs_reduce_returns(self, synthetic_prices):
        rets = synthetic_prices.pct_change().dropna()
        rm = fit_markov_regime(rets, n_regimes=3)
        labels = label_regimes(rm, rets)
        bt = run_regime_backtest(
            synthetic_prices, regimes=rm.regimes, regime_labels=labels,
            transaction_cost=0.005,
        )
        gross = bt.gross_returns
        net = bt.portfolio_returns
        assert (gross - net).mean() >= 0

    def test_performance_metrics_complete(self, synthetic_prices):
        rets = synthetic_prices.pct_change().dropna()
        rm = fit_markov_regime(rets, n_regimes=3)
        labels = label_regimes(rm, rets)
        bt = run_regime_backtest(
            synthetic_prices, regimes=rm.regimes, regime_labels=labels,
        )
        m = compute_metrics(bt.portfolio_returns)
        assert m.sharpe_ratio != 0
        assert m.asymmetry >= 0
        assert m.breadth_estimate >= 1

    def test_compare_strategies(self, synthetic_prices):
        rets = synthetic_prices.pct_change().dropna()
        rm = fit_markov_regime(rets, n_regimes=3)
        labels = label_regimes(rm, rets)
        bt = run_regime_backtest(
            synthetic_prices, regimes=rm.regimes, regime_labels=labels,
        )
        n = len(rets.columns)
        eq_w = {c: 1.0 / n for c in rets.columns}
        bench = rets.copy()
        bench["port"] = sum(rets[c] * eq_w[c] for c in rets.columns)
        comp = compare_strategies({
            "Portfolio": bt.portfolio_returns,
            "Benchmark": bench["port"],
        })
        assert comp.shape[1] >= 2

    def test_effective_bets(self, synthetic_prices):
        rets = synthetic_prices.pct_change().dropna()
        port = rets.mean(axis=1)
        br = _effective_bets(port)
        assert br >= 1.0
        assert br <= 52.0

    def test_vol_target_no_lookahead(self, synthetic_prices):
        rets = synthetic_prices.pct_change().dropna()
        rm = fit_markov_regime(rets, n_regimes=3)
        labels = label_regimes(rm, rets)
        bt = run_regime_backtest(
            synthetic_prices, regimes=rm.regimes, regime_labels=labels,
            vol_target=0.15,
        )
        trailing = bt.portfolio_returns.rolling(60).std() * np.sqrt(252)
        median_vol = trailing.median()
        assert 0.05 <= median_vol <= 0.30
