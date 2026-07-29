import pandas as pd
import numpy as np
from hmmlearn import hmm
from dataclasses import dataclass
from src.config import REGIME_MODEL


@dataclass
class RegimeResult:
    regimes: pd.Series
    regime_probs: pd.DataFrame
    n_regimes: int
    transition_matrix: np.ndarray
    means: np.ndarray
    burn_in: int


def _expanding_zscore(features: pd.DataFrame) -> pd.DataFrame:
    """Compute z-scores using an expanding window to avoid look-ahead bias."""
    mean = features.expanding().mean().shift(1)
    std = features.expanding().std().shift(1).clip(lower=1e-6)
    return ((features - mean) / std).dropna(how="all")


def fit_markov_regime(
    returns: pd.DataFrame,
    n_regimes: int = REGIME_MODEL["n_regimes"],
    n_iter: int = REGIME_MODEL["n_iter"],
    random_state: int = REGIME_MODEL["random_state"],
    gpr_series: pd.Series | None = None,
    burn_in_days: int = 756,
) -> RegimeResult:
    """
    Fit a Gaussian HMM (Markov-switching) model on multi-asset returns
    using expanding-window z-scores to eliminate look-ahead bias.

    The first `burn_in_days` observations are used to initialise scaling
    but excluded from the regime assignment returned — the HMM needs a
    stable burn-in period before its state estimates are meaningful.
    """
    aligned = returns.dropna()

    if gpr_series is not None:
        gpr_aligned = gpr_series.reindex(aligned.index).ffill().fillna(0.0)
        features = pd.concat([aligned, gpr_aligned], axis=1).dropna()
    else:
        features = aligned

    X_scaled = _expanding_zscore(features)

    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        tol=1e-4,
    )
    model.fit(X_scaled)

    hidden_states = model.predict(X_scaled)
    state_probs = model.predict_proba(X_scaled)

    regime_series = pd.Series(
        hidden_states, index=X_scaled.index, name="regime", dtype=int,
    )
    prob_df = pd.DataFrame(
        state_probs, index=X_scaled.index,
        columns=[f"regime_{i}" for i in range(n_regimes)],
    )

    trans_mat = model.transmat_

    return RegimeResult(
        regimes=regime_series,
        regime_probs=prob_df,
        n_regimes=n_regimes,
        transition_matrix=trans_mat,
        means=model.means_,
        burn_in=burn_in_days,
    )


def label_regimes(
    result: RegimeResult,
    returns: pd.DataFrame,
) -> pd.Series:
    """
    Label the HMM states intelligently based on volatility.
    - Regime with lowest vol: 'Normal'
    - Regime with mid vol: 'Elevated Risk'
    - Regime with highest vol: 'Crisis'
    """
    if result.n_regimes != 3:
        return pd.Series(
            [f"Regime {r}" for r in result.regimes.values],
            index=result.regimes.index,
            name="regime_label",
        )

    returns_aligned = returns.loc[result.regimes.index]
    mapping = {}
    for i in range(3):
        mask = result.regimes == i
        regime_rets = returns_aligned.loc[mask]
        vol = regime_rets.std().mean() if len(regime_rets) > 0 else 0.0
        mapping[i] = vol

    sorted_regimes = sorted(mapping, key=mapping.get)
    labels = {sorted_regimes[0]: "Normal", sorted_regimes[1]: "Elevated Risk", sorted_regimes[2]: "Crisis"}

    return pd.Series(
        [labels[r] for r in result.regimes.values],
        index=result.regimes.index,
        name="regime_label",
    )
