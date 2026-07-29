import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVENTS_DIR = DATA_DIR / "events"
MARKET_DIR = DATA_DIR / "market"
GPR_DIR = DATA_DIR / "gpr"
PROCESSED_DIR = DATA_DIR / "processed"

OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY", "")
ZEN_MODEL = os.getenv("ZEN_MODEL", "deepseek-v4-flash-free")
ZEN_BASE_URL = "https://opencode.ai/zen/v1"

ASSETS = {
    "equities_us": {"ticker": "IVV",  "name": "S&P 500 ETF",       "asset_class": "Equities"},
    "equities_em": {"ticker": "EEM",  "name": "Emerging Markets",  "asset_class": "Equities"},
    "bonds_long":  {"ticker": "TLT",  "name": "US 20Y+ Treasury",  "asset_class": "Bonds"},
    "bonds_int":   {"ticker": "IEI",  "name": "US 3-7Y Treasury",  "asset_class": "Bonds"},
    "gold":        {"ticker": "GLD",  "name": "Gold ETF",          "asset_class": "Commodities"},
    "oil_wti":     {"ticker": "CL=F", "name": "WTI Crude Oil",     "asset_class": "Commodities"},
    "oil_brent":   {"ticker": "BZ=F", "name": "Brent Crude Oil",   "asset_class": "Commodities"},
    "fx_usd_idx":  {"ticker": "DX-Y.NYB", "name": "US Dollar Index","asset_class": "FX"},
    "fx_eurusd":   {"ticker": "EURUSD=X", "name": "EUR/USD",       "asset_class": "FX"},
    "vix":         {"ticker": "^VIX", "name": "VIX Volatility",    "asset_class": "Volatility"},
}

ASSET_CATEGORIES = {
    "risky": ["equities_us", "equities_em"],
    "safe":  ["bonds_long", "bonds_int", "gold", "fx_usd_idx"],
    "commodities": ["oil_wti", "oil_brent", "gold"],
    "all": list(ASSETS.keys()),
}

EVENT_WINDOW = (-5, 10)
ESTIMATION_WINDOW = (-60, -11)

REGIME_MODEL = {
    "n_regimes": 3,
    "n_iter": 1000,
    "random_state": 42,
}

PORTFOLIO_CONFIG = {
    "rebalance_freq": "ME",
    "vol_target": 0.10,
    "max_leverage": 1.5,
    "transaction_cost": 0.001,
}

GPR_PUBLISHED_URL = (
    "https://www.matteoiacoviello.com/gpr_files/gpr_data.csv"
)
