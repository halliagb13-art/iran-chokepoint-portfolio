import logging
import pandas as pd
from pathlib import Path
from src.config import EVENTS_DIR

logger = logging.getLogger(__name__)

HISTORICAL_EVENTS = [
    {
        "date": "1987-07-24",
        "name": "Bridgeton hits mine (Operation Earnest Will)",
        "chokepoint": "Hormuz",
        "category": "act",
        "severity": 3,
        "description": (
            "Kuwaiti tanker Bridgeton, re-flagged under US protection, hits an Iranian mine "
            "in the Persian Gulf. Part of Operation Earnest Will protecting Gulf shipping."
        ),
        "oil_impact_pct": -1.6,
        "affected_assets": ["oil_wti", "oil_brent"],
        "source": "CRS Report R45281",
    },
    {
        "date": "1988-04-18",
        "name": "Operation Praying Mantis",
        "chokepoint": "Hormuz",
        "category": "act",
        "severity": 4,
        "description": (
            "US Navy destroys ~40% of Iran's navy in retaliation for mining. "
            "Paradoxically reduces risk of oil cutoff; oil prices drop after initial spike."
        ),
        "oil_impact_pct": 10.7,
        "affected_assets": ["oil_wti", "oil_brent", "equities_us"],
        "source": "CRS Report R45281",
    },
    {
        "date": "1995-03-28",
        "name": "Iran arms Strait of Hormuz",
        "chokepoint": "Hormuz",
        "category": "threat",
        "severity": 3,
        "description": (
            "Pentagon announces Iran installing missiles in the Strait and "
            "fortifying two islands also claimed by UAE."
        ),
        "oil_impact_pct": 2.5,
        "affected_assets": ["oil_wti", "oil_brent"],
        "source": "CRS Report R45281",
    },
    {
        "date": "2008-07-01",
        "name": "Iran threatens Strait closure",
        "chokepoint": "Hormuz",
        "category": "threat",
        "severity": 2,
        "description": (
            "Iranian officials threaten to close the Strait of Hormuz "
            "in response to international pressure over nuclear program."
        ),
        "oil_impact_pct": None,
        "affected_assets": ["oil_wti", "oil_brent"],
        "source": "Various",
    },
    {
        "date": "2011-12-28",
        "name": "Iran VP threatens closure",
        "chokepoint": "Hormuz",
        "category": "threat",
        "severity": 3,
        "description": (
            "Iran's first VP Mohammad Reza Rahimi threatens to close the Strait "
            "if oil sanctions are imposed. Oil rises ~4% over following week."
        ),
        "oil_impact_pct": 1.0,
        "affected_assets": ["oil_wti", "oil_brent", "equities_us"],
        "source": "CRS Report R45281",
    },
    {
        "date": "2012-01-04",
        "name": "Peak of Hormuz threat premium",
        "chokepoint": "Hormuz",
        "category": "threat",
        "severity": 3,
        "description": (
            "Oil peaks ~4% higher after Iran's December threats. "
            "Tensions remain elevated through early 2012."
        ),
        "oil_impact_pct": 4.0,
        "affected_assets": ["oil_wti", "oil_brent"],
        "source": "CRS Report R45281",
    },
    {
        "date": "2019-05-12",
        "name": "Gulf of Oman tanker sabotage",
        "chokepoint": "Hormuz",
        "category": "act",
        "severity": 3,
        "description": (
            "Four commercial ships damaged by explosive charges near Fujairah port, "
            "UAE. Oil prices rise ~3%."
        ),
        "oil_impact_pct": 3.2,
        "affected_assets": ["oil_wti", "oil_brent", "equities_us", "fx_usd_idx"],
        "source": "CRS Report R45281",
    },
    {
        "date": "2019-06-13",
        "name": "Gulf of Oman tanker attacks",
        "chokepoint": "Hormuz",
        "category": "act",
        "severity": 4,
        "description": (
            "Two oil tankers (Kokuka Courageous, Front Altair) attacked with limpet mines "
            "near the Strait. Brent initially surges 4%."
        ),
        "oil_impact_pct": 4.1,
        "affected_assets": ["oil_wti", "oil_brent", "equities_us", "vix"],
        "source": "CRS Report R45281",
    },
    {
        "date": "2019-07-19",
        "name": "Stena Impero seizure",
        "chokepoint": "Hormuz",
        "category": "act",
        "severity": 3,
        "description": (
            "Iranian IRGC seizes British oil tanker Stena Impero in the Strait. "
            "Brent rises ~2%."
        ),
        "oil_impact_pct": 2.1,
        "affected_assets": ["oil_wti", "oil_brent", "fx_usd_idx"],
        "source": "CRS Report R45281",
    },
    {
        "date": "2019-09-14",
        "name": "Abqaiq-Khurais attack",
        "chokepoint": "Hormuz",
        "category": "act",
        "severity": 5,
        "description": (
            "Drone/missile attack on Saudi Aramco facilities knocks out 5.7mbd (5% of global supply). "
            "Largest single supply disruption in history at the time. "
            "Brent spikes 15% in a single day."
        ),
        "oil_impact_pct": 14.6,
        "affected_assets": ["oil_wti", "oil_brent", "equities_us", "gold", "vix"],
        "source": "CRS Report R45281",
    },
    {
        "date": "2020-01-03",
        "name": "Soleimani assassination",
        "chokepoint": "Hormuz",
        "category": "act",
        "severity": 5,
        "description": (
            "US kills Qasem Soleimani, head of Iran's Quds Force. "
            "Iran vows revenge. Oil spikes 12% on fears of Strait closure. "
            "But prices reverse within a month (-20%) as no supply disruption materializes."
        ),
        "oil_impact_pct": 12.2,
        "affected_assets": ["oil_wti", "oil_brent", "equities_us", "gold", "vix", "fx_usd_idx"],
        "source": "CRS Report R45281",
    },
    {
        "date": "2022-05-27",
        "name": "Iran seizes Greek tankers",
        "chokepoint": "Hormuz",
        "category": "act",
        "severity": 2,
        "description": (
            "Iran seizes two Greek-flagged tankers in retaliation for EU sanctions. "
            "Shipping insurance costs rise."
        ),
        "oil_impact_pct": 3.5,
        "affected_assets": ["oil_wti", "oil_brent"],
        "source": "CRS Report R45281",
    },
    {
        "date": "2023-11-19",
        "name": "Houthi Red Sea attacks begin",
        "chokepoint": "Bab el-Mandeb",
        "category": "act",
        "severity": 4,
        "description": (
            "Houthis seize Galaxy Leader and begin sustained attacks on commercial shipping "
            "in the Red Sea. Major shipping lines suspend Suez transits. "
            "Container freight rates surge 130% over following months."
        ),
        "oil_impact_pct": None,
        "affected_assets": ["oil_wti", "oil_brent", "equities_us", "equities_em"],
        "source": "World Bank / EIA / ITF-OECD",
    },
    {
        "date": "2024-04-13",
        "name": "MSC Aries seizure",
        "chokepoint": "Hormuz",
        "category": "act",
        "severity": 3,
        "description": (
            "Iran seizes Portuguese-flagged container ship MSC Aries near the Strait. "
            "War risk premiums on shipping continue rising."
        ),
        "oil_impact_pct": 2.1,
        "affected_assets": ["oil_wti", "oil_brent"],
        "source": "CRS Report R45281",
    },
    {
        "date": "2025-06-13",
        "name": "Israel bombs Iran",
        "chokepoint": "Hormuz",
        "category": "act",
        "severity": 4,
        "description": (
            "Israel launches strikes on Iran. Brent jumps 8.5% ($71 → $77). "
            "Iran parliament votes to close Strait, pending Supreme Leader approval."
        ),
        "oil_impact_pct": 8.5,
        "affected_assets": ["oil_wti", "oil_brent", "equities_us", "gold", "vix", "fx_usd_idx"],
        "source": "CRS Report R45281",
    },
    {
        "date": "2026-02-28",
        "name": "US-Israel attack Iran / Strait effectively closed",
        "chokepoint": "Hormuz",
        "category": "crisis",
        "severity": 5,
        "description": (
            "US and Israel begin military operations against Iran. Iran retaliates with drones, "
            "missiles, and attacks on vessels in the Strait. Insurance becomes unavailable. "
            "Largest oil supply disruption in history: ~14-17mbd offline. "
            "Brent surges 60%+ to $137 at peak. "
            "Brokerage: supplies through Saudi/UAE bypass pipelines at max capacity (5.5mbd)."
        ),
        "oil_impact_pct": 60.0,
        "affected_assets": [
            "oil_wti", "oil_brent", "equities_us", "equities_em",
            "bonds_long", "gold", "vix", "fx_usd_idx", "fx_eurusd",
        ],
        "source": "Brookings / CRS Report R45281 / Seeer AI",
    },
    {
        "date": "2026-04-13",
        "name": "US blocks Iranian oil exports",
        "chokepoint": "Hormuz",
        "category": "act",
        "severity": 4,
        "description": (
            "US begins blockade of Iranian oil exports. "
            "Iran's remaining ~2mbd exports stopped. "
            "Oil markets remain highly elevated."
        ),
        "oil_impact_pct": None,
        "affected_assets": ["oil_wti", "oil_brent", "equities_us", "equities_em"],
        "source": "Brookings",
    },
]

EVENTS_DF = pd.DataFrame(HISTORICAL_EVENTS)


def get_events(
    chokepoint: str | None = None,
    min_severity: int = 1,
    category: str | None = None,
) -> pd.DataFrame:
    df = EVENTS_DF.copy()
    if chokepoint:
        df = df[df["chokepoint"] == chokepoint]
    if min_severity > 1:
        df = df[df["severity"] >= min_severity]
    if category:
        df = df[df["category"] == category]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def save_events_csv():
    path = EVENTS_DIR / "historical_events.csv"
    EVENTS_DF.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    path = save_events_csv()
    logger.info("Saved %d events to %s", len(EVENTS_DF), path)
