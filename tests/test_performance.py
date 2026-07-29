import pytest
import pandas as pd
import numpy as np
from src.performance import compute_metrics, compare_strategies


class TestPerformance:
    def test_compute_metrics_positive_returns(self):
        returns = pd.Series(np.random.normal(0.0005, 0.01, 1000))
        metrics = compute_metrics(returns)
        assert metrics.sharpe_ratio is not None
        assert metrics.hit_ratio is not None
        assert metrics.asymmetry is not None

    def test_compute_metrics_negative_returns(self):
        returns = pd.Series(np.random.normal(-0.001, 0.01, 1000))
        metrics = compute_metrics(returns)
        assert isinstance(metrics.max_drawdown, float)
        assert metrics.max_drawdown <= 0

    def test_hit_ratio_fifty_fifty(self):
        returns = pd.Series([0.01, -0.01] * 500)
        metrics = compute_metrics(returns)
        assert metrics.hit_ratio == pytest.approx(0.5, abs=0.02)

    def test_high_sharpe_high_hit(self):
        returns = pd.Series([0.001] * 500 + [-0.01] * 50)
        metrics = compute_metrics(returns)
        assert metrics.hit_ratio > 0.8

    def test_compare_strategies(self):
        s1 = pd.Series(np.random.normal(0.001, 0.01, 500))
        s2 = pd.Series(np.random.normal(0.0005, 0.015, 500))
        result = compare_strategies({"Strategy A": s1, "Strategy B": s2})
        assert "Strategy A" in result.columns or "Strategy A" in result.index.values

    def test_empty_returns(self):
        empty = pd.Series(dtype=float)
        metrics = compute_metrics(empty)
        assert metrics.total_return == 0

    def test_short_returns(self):
        short = pd.Series([0.01, -0.005, 0.02])
        metrics = compute_metrics(short)
        assert isinstance(metrics.sharpe_ratio, float)
