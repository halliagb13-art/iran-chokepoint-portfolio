import pytest
import pandas as pd
import numpy as np
from src.regime_model import fit_markov_regime, label_regimes, _expanding_zscore


class TestRegimeModel:
    @pytest.fixture
    def sample_returns(self):
        np.random.seed(42)
        dates = pd.date_range("2018-01-01", periods=1000, freq="D")
        data = {
            "oil_wti": np.random.normal(0.0002, 0.02, 1000),
            "spy": np.random.normal(0.0005, 0.01, 1000),
            "gold": np.random.normal(0.0003, 0.008, 1000),
        }
        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def sample_gpr(self, sample_returns):
        return pd.Series(
            np.random.uniform(0, 100, len(sample_returns)),
            index=sample_returns.index,
            name="gpr",
        )

    def test_fit_returns_regimeresult(self, sample_returns):
        result = fit_markov_regime(sample_returns)
        assert hasattr(result, "regimes")
        assert hasattr(result, "regime_probs")
        assert hasattr(result, "transition_matrix")
        assert hasattr(result, "n_regimes")

    def test_three_regimes(self, sample_returns):
        result = fit_markov_regime(sample_returns, n_regimes=3)
        assert result.n_regimes == 3
        assert len(result.regimes.unique()) <= 3

    def test_with_gpr(self, sample_returns, sample_gpr):
        result = fit_markov_regime(sample_returns, gpr_series=sample_gpr)
        assert len(result.regimes) <= len(sample_returns)

    def test_regime_probs_sum_to_one(self, sample_returns):
        result = fit_markov_regime(sample_returns, n_regimes=3)
        probs_sum = result.regime_probs.sum(axis=1)
        assert probs_sum.min() == pytest.approx(1.0, abs=1e-4)

    def test_expanding_zscore_no_lookahead(self):
        data = pd.DataFrame({"x": range(1, 101)})
        z = _expanding_zscore(data)
        n = len(z)
        assert n >= 95
        for k in range(3, n):
            orig_idx = z.index[k]
            train = data.iloc[:orig_idx].iloc[:, 0]
            mean = train.mean()
            std = train.std()
            expected = (data.iloc[orig_idx].iloc[0] - mean) / std
            actual = z.iloc[k].iloc[0]
            assert abs(actual - expected) < 1e-4

    def test_label_regimes_three_states(self, sample_returns):
        result = fit_markov_regime(sample_returns, n_regimes=3)
        labels = label_regimes(result, sample_returns)
        assert set(labels.unique()) == {"Normal", "Elevated Risk", "Crisis"}
        assert len(labels) == len(result.regimes)

    def test_label_regimes_stable_across_runs(self, sample_returns):
        result1 = fit_markov_regime(sample_returns, n_regimes=3, random_state=42)
        result2 = fit_markov_regime(sample_returns, n_regimes=3, random_state=42)
        labels1 = label_regimes(result1, sample_returns)
        labels2 = label_regimes(result2, sample_returns)
        pd.testing.assert_series_equal(labels1, labels2)

    def test_expanding_drops_initial_nans(self, sample_returns):
        result = fit_markov_regime(sample_returns)
        assert len(result.regimes) <= len(sample_returns) - 1
