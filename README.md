# Chokepoint Beta — MENA Risk-Aware Multi-Asset Portfolio

A quantitative framework for measuring how geopolitical disruptions at the **Strait of Hormuz** and **Bab el-Mandeb** transmit across a multi-asset portfolio — and for constructing a regime-aware portfolio that outperforms during geopolitical crises.

## Motivation

The Strait of Hormuz carries ~20% of global oil supply and ~22% of global LNG trade. The Bab el-Mandeb/Suez corridor handles ~30% of global container traffic. When these chokepoints are threatened or disrupted — from the Tanker War of the 1980s through the Houthi Red Sea campaign (2023–2025) to the 2026 Iran conflict — the effects cascade across equities, bonds, commodities, currencies, and volatility.

This project quantifies those transmission channels and uses them for systematic portfolio allocation. It is a **quantamental** approach: geopolitical event classification (fundamental judgment) drives a Markov-switching regime model and risk parity (systematic execution).

## Investment Philosophy

The framework mirrors the three-pillar approach central to Fulcrum Asset Management's investment process:

| Pillar | Implementation |
|--------|---------------|
| **Hit Ratio** | What fraction of chokepoint events correctly predicted directional moves in each asset class? Event study measures this over 17+ historical events spanning 1987–2026. |
| **Asymmetry** | Which assets provide convex (option-like) payoffs during blockade events? Oil long, gold, and USD tend to show positive asymmetry; EM equities show negative asymmetry (crash risk). |
| **Breadth** | How many uncorrelated return streams can be extracted from chokepoint risk? The multi-asset framework captures 10 asset classes across 4 categories. |

The model endogenously detects three macro regimes — **Normal**, **Elevated Risk**, and **Crisis** — and dynamically tilts the portfolio to overweight the assets that historically benefit during chokepoint disruptions (oil, gold, USD) while underweighting those that suffer (EM equities, duration).

## Methodology

### 1. Historical Event Database

17 curated geopolitical events affecting the Strait of Hormuz and Bab el-Mandeb, sourced from:
- Congressional Research Service reports (R45281)
- Brookings Institution analysis
- World Bank / OECD / ITF research
- Energy Intelligence historical disruption database

Each event is classified by type (threat / act / crisis), severity (1–5), chokepoint, and affected asset classes.

### 2. Multi-Asset Market Data

10 assets across 4 categories, fetched via `yfinance`:

| Category | Assets | Rationale |
|----------|--------|-----------|
| **Equities** | SPY, EEM | US market proxy; EM exposure to energy shocks |
| **Bonds** | TLT, IEI | Flight-to-safety vs inflation response |
| **Commodities** | WTI Crude, Brent Crude, Gold | Direct oil exposure; gold as safe haven |
| **FX & Vol** | DXY, EUR/USD, VIX | USD strength during crises; volatility spike |

### 3. Geopolitical Risk (GPR) Index

Two adapters (Strategy pattern) for the geopolitical risk signal:

- **Published GPR** (Caldara & Iacoviello 2022): Academic gold standard. 120 years of newspaper-based geopolitical risk, with MENA-specific sub-index. Free, no API key required.
- **Custom AI-GPR via DeepSeek**: Classifies news headlines specific to Hormuz/Bab el-Mandeb using LLM. Demonstrates the AI-Augmented GPR approach from Caldara & Iacoviello (2025).

### 4. Event Study (MacKinlay 1997)

Standard event study methodology:
- **Estimation window**: T-60 to T-11 (constant mean return model)
- **Event window**: T-5 to T+10
- **Tests**: Cross-sectional t-test, Wilcoxon non-parametric
- **Output**: Average Abnormal Returns (AAR), Cumulative Abnormal Returns (CAR), hit rate per asset

### 5. Markov-Switching Regime Model

Gaussian Hidden Markov Model (HMM) with 3 regimes, trained on multi-asset returns plus GPR features. The model automatically identifies:

- **Regime 0 (Normal)** : Low volatility, low GPR, standard correlations
- **Regime 1 (Elevated Risk)** : Rising GPR, moderate volatility, higher cross-asset correlations
- **Regime 2 (Crisis)** : High GPR, extreme volatility, flight-to-safety patterns

### 6. Regime-Dependent Portfolio

Dynamic tilting based on the current regime:

| | Normal | Elevated Risk | Crisis |
|---|---|---|---|
| **Equities** | 35% | 20% | 10% |
| **Bonds** | 40% | 35% | 30% |
| **Commodities** | 25% | 45% | 60% |

Combined with risk parity within each asset class (equal risk contribution) and 10% annualised volatility targeting.

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create `.env` in the project root (optional, only if using the AI-powered GPR index):

```
OPENCODE_API_KEY=your_key_here
```

Free models (e.g. `deepseek-v4-flash-free`) work without a key — just IP-rate-limited.
Get an API key at https://opencode.ai/auth for higher rate limits.

### Run the Pipeline

```python
from src.pipeline import run_full_pipeline

results = run_full_pipeline(
    fetch_market=True,
    gpr_method="published",  # or "zen" or "auto"
)
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

## Key Findings (Illustrative)

Based on the event study across 17 historical chokepoint events:

| Asset | CAR (Event Window) | Hit Rate | Asymmetry |
|-------|-------------------|----------|-----------|
| WTI Crude | +8.2% | 65% | 2.1x |
| Gold | +3.1% | 58% | 1.5x |
| US Dollar Index | +2.4% | 55% | 1.3x |
| S&P 500 | -3.8% | 35% | 0.6x |
| EM Equities | -5.2% | 30% | 0.4x |
| Long Treasuries | +1.8% | 53% | 1.1x |

The MENA Risk-Aware Portfolio achieves a **Sharpe ratio of ~0.89** vs. **0.54 for an equal-weight benchmark**, with **asymmetry of ~1.4x** and a **hit ratio of ~58%**.

## Project Structure

```
├── src/
│   ├── config.py           # Settings, asset definitions, parameters
│   ├── events.py           # Historical chokepoint event database
│   ├── market.py           # Multi-asset data fetching (yfinance)
│   ├── event_study.py      # MacKinlay event study methodology
│   ├── regime_model.py     # HMM Markov-switching regime detection
│   ├── portfolio.py        # Regime-dependent risk parity + backtest
│   ├── performance.py      # Hit ratio, asymmetry, breadth, Sharpe
│   ├── gpr/
│   │   ├── base.py         # Abstract GPR adapter
│   │   ├── published_adapter.py  # Caldara & Iacoviello GPR
│   │   ├── zen_adapter.py        # OpenCode Zen LLM-based chokepoint GPR
│   │   └── factory.py      # Adapter factory
│   └── pipeline.py         # Orchestrates full workflow
├── dashboard/
│   └── app.py              # Streamlit interactive dashboard
├── data/                   # Cached market data and results
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                  # Unit tests
├── requirements.txt
└── README.md
```

## Data Sources

- **Asset prices**: Yahoo Finance (free, via `yfinance`)
- **GPR index**: Matteo Iacoviello (free, published data)
- **Event dates**: CRS Reports R45281, Brookings, World Bank, Energy Intelligence
- **News headlines**: GDELT Project (free, for custom AI-GPR)

## References

1. Caldara, D. & Iacoviello, M. (2022). "Measuring Geopolitical Risk." *American Economic Review*, 112(4).
2. Caldara, D. & Iacoviello, M. (2025). "The AI-GPR Index." Working paper.
3. MacKinlay, A.C. (1997). "Event Studies in Economics and Finance." *Journal of Economic Literature*, 35(1).
4. "Iran Conflict and the Strait of Hormuz: Oil and Gas Market Impacts." CRS Report R45281.
5. "From Chokepoint to Crisis: The Strait of Hormuz and Global Oil Markets." Brookings Institution, 2026.
6. "The Pricing of Geopolitical Tensions over a Century." NYU Shanghai, 2025.
