import logging
import pandas as pd
import numpy as np
from pathlib import Path
from src.events import get_events, EVENTS_DF
from src.market import fetch_all_assets, compute_returns, load_saved_assets
from src.event_study import run_event_study, summarize_results
from src.regime_model import fit_markov_regime, label_regimes
from src.portfolio import run_regime_backtest, benchmark_returns, BacktestResult
from src.performance import compute_metrics, compare_strategies
from src.gpr.factory import create_gpr_adapter
from src.config import PROCESSED_DIR

logger = logging.getLogger(__name__)


def run_full_pipeline(
    fetch_market: bool = True,
    gpr_method: str = "auto",
    start: str = "2000-01-01",
) -> dict:
    results = {}

    # 1. Events
    events = get_events()
    event_dates = events["date"].tolist()
    results["events"] = events
    results["event_dates"] = event_dates

    # 2. Market data
    if fetch_market:
        prices = fetch_all_assets(start=start)
    else:
        prices = load_saved_assets()
    returns = compute_returns(prices)
    results["prices"] = prices
    results["returns"] = returns

    if prices.empty:
        logger.error("No market data — returning partial results")
        return results

    # 3. Event study
    es_results = run_event_study(prices, event_dates)
    es_summary = summarize_results(es_results)
    results["event_study"] = es_summary
    results["event_study_detailed"] = es_results

    # 4. GPR index
    gpr_adapter = create_gpr_adapter(method=gpr_method)
    gpr_series = gpr_adapter.fetch_index(start=start)
    results["gpr"] = gpr_series
    results["gpr_name"] = gpr_adapter.name()

    # 5. Regime model
    regime_result = fit_markov_regime(returns, gpr_series=gpr_series)
    regime_labels = label_regimes(regime_result, returns)
    results["regime_model"] = regime_result
    results["regime_labels"] = regime_labels

    # 6. Portfolio backtest
    bt = run_regime_backtest(
        prices, regimes=regime_result.regimes, regime_labels=regime_labels
    )
    results["backtest"] = bt

    # 7. Benchmark
    n = len(returns.columns)
    equal_w = {c: 1.0 / n for c in returns.columns}
    bench = benchmark_returns(prices, weights=equal_w)
    results["benchmark"] = bench

    # 8. Performance comparison
    perf = compare_strategies({
        "MENA Risk-Aware Portfolio": bt.portfolio_returns,
        "Equal-Weight Benchmark": bench,
    })
    results["performance"] = perf

    return results


def save_processed_results(results: dict):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results["event_study"].to_csv(PROCESSED_DIR / "event_study_summary.csv")
    results["gpr"].to_csv(PROCESSED_DIR / "gpr_index.csv")
    results["regime_labels"].to_csv(PROCESSED_DIR / "regime_labels.csv")
    results["backtest"].portfolio_returns.to_frame().to_csv(
        PROCESSED_DIR / "portfolio_returns.csv"
    )
    results["backtest"].weights.to_csv(PROCESSED_DIR / "portfolio_weights.csv")
    results["backtest"].gross_returns.to_frame().to_csv(
        PROCESSED_DIR / "gross_returns.csv"
    )
    results["backtest"].transaction_costs.to_frame().to_csv(
        PROCESSED_DIR / "transaction_costs.csv"
    )
    results["benchmark"].to_frame().to_csv(PROCESSED_DIR / "benchmark_returns.csv")
    results["performance"].to_csv(PROCESSED_DIR / "performance_comparison.csv")

    logger.info("Saved processed results to %s", PROCESSED_DIR)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Running full pipeline")
    res = run_full_pipeline(fetch_market=True, gpr_method="zen")
    save_processed_results(res)
    logger.info("Done — see results in %s", PROCESSED_DIR)
