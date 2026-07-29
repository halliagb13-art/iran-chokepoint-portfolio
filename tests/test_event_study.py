import pytest
import pandas as pd
import numpy as np
from src.event_study import run_event_study, summarize_results, estimate_normal_returns


class TestEventStudy:
    @pytest.fixture
    def sample_prices(self):
        dates = pd.date_range("2020-01-01", periods=200, freq="D")
        np.random.seed(42)
        data = {
            "oil_wti": 100 + np.cumsum(np.random.randn(200) * 0.5),
            "spy": 300 + np.cumsum(np.random.randn(200) * 0.3),
        }
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def sample_events(self):
        return ["2020-03-15", "2020-06-15"]

    def test_estimate_normal_returns(self, sample_prices):
        mu = estimate_normal_returns(
            sample_prices["oil_wti"],
            pd.Timestamp("2020-06-15"),
            (-30, -5),
        )
        assert isinstance(mu, float)

    def test_run_event_study_returns_dict(self, sample_prices, sample_events):
        results = run_event_study(sample_prices, sample_events)
        assert isinstance(results, dict)

    def test_event_study_has_required_attributes(self, sample_prices, sample_events):
        results = run_event_study(sample_prices, sample_events)
        for asset, res in results.items():
            assert hasattr(res, "aar")
            assert hasattr(res, "car")
            assert hasattr(res, "aar_t_stats")
            assert hasattr(res, "hit_rate")
            assert hasattr(res, "car_p_value")

    def test_aar_is_series(self, sample_prices, sample_events):
        results = run_event_study(sample_prices, sample_events)
        for res in results.values():
            assert isinstance(res.aar, pd.Series)

    def test_car_is_cumulative(self, sample_prices, sample_events):
        results = run_event_study(sample_prices, sample_events)
        for res in results.values():
            assert abs(res.car.iloc[-1]) >= abs(res.aar.iloc[-1])

    def test_summarize_results(self, sample_prices, sample_events):
        results = run_event_study(sample_prices, sample_events)
        summary = summarize_results(results)
        assert "asset" in summary.columns
        assert "car_event_window" in summary.columns
