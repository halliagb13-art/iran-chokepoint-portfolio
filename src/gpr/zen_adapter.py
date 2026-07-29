import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal
from src.config import OPENCODE_API_KEY, ZEN_MODEL, ZEN_BASE_URL, GPR_DIR
from src.gpr.base import GPRAdapter

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


_SAMPLE_HEADLINES = [
    ("2026-02-28", "Iran closes Strait of Hormuz after US-Israel strikes; oil surges 15%"),
    ("2026-03-01", "IRGC mines shipping lanes in Persian Gulf, insurers halt coverage"),
    ("2026-03-05", "Saudi Arabia activates East-West pipeline at full capacity to bypass Hormuz"),
    ("2026-03-10", "IEA releases 400 million barrels from strategic reserves to calm markets"),
    ("2026-03-15", "China urges restraint as Hormuz crisis enters third week"),
    ("2026-04-01", "Iran begins tiered toll system for Strait passage"),
    ("2026-04-13", "US imposes blockade on Iranian oil exports"),
    ("2026-05-01", "Oil prices stabilize near $65 as strategic reserves and bypass routes hold"),
    ("2026-05-15", "UAE departs OPEC amid fracturing oil alliance"),
    ("2026-06-01", "Hormuz transit remains at near-standstill; negotiations stalled"),
    ("2025-06-13", "Israel strikes Iranian nuclear sites; Iran parliament votes to close Strait"),
    ("2025-06-14", "Brent crude jumps 8% on fears of Hormuz disruption"),
    ("2025-06-20", "US bombs Iranian nuclear facilities in coordinated strike"),
    ("2024-04-13", "Iran seizes Portuguese-flagged MSC Aries near Strait of Hormuz"),
    ("2023-11-19", "Houthis seize Galaxy Leader in Red Sea; shipping lines suspend Suez routes"),
    ("2023-12-01", "Container freight rates surge 50% as Red Sea crisis deepens"),
    ("2023-12-15", "Major oil companies pause Red Sea transits; insurance costs spike"),
    ("2020-01-03", "US kills Qasem Soleimani in Baghdad; Iran vows revenge"),
    ("2020-01-04", "Oil surges 12% on fears of Iranian retaliation in the Gulf"),
    ("2019-09-14", "Drone attack knocks out 5.7mbd Saudi production at Abqaiq-Khurais"),
    ("2019-06-13", "Two oil tankers attacked with limpet mines in Gulf of Oman"),
    ("2019-05-12", "Four commercial ships damaged by explosive charges off Fujairah"),
    ("2019-07-19", "Iran seizes British tanker Stena Impero in Strait of Hormuz"),
    ("2012-01-04", "Iran nuclear tensions peak; oil at elevated Hormuz risk premium"),
    ("2011-12-28", "Iran threatens to close Strait of Hormuz over oil sanctions"),
    ("2008-07-01", "Iranian officials threaten Strait closure amid nuclear standoff"),
]


class ZenGPRAdapter(GPRAdapter):
    """
    Builds a custom chokepoint-specific GPR index by classifying news
    headlines via the OpenCode Zen API (OpenAI-compatible, free models available).
    Falls back to hand-labelled data if no API key is configured.
    """

    def __init__(self, model: str | None = None):
        self._model = model or ZEN_MODEL
        self._client = None
        self._cached: pd.Series | None = None

        if OpenAI is not None:
            kwargs = {"base_url": f"{ZEN_BASE_URL}/"}
            if OPENCODE_API_KEY:
                kwargs["api_key"] = OPENCODE_API_KEY
            else:
                kwargs["api_key"] = "not-required"
            self._client = OpenAI(**kwargs)

    def name(self) -> str:
        if self._client:
            return f"OpenCode Zen GPR ({self._model})"
        return "Zen GPR (Fallback — using pre-labelled headlines)"

    def _classify_headline(self, headline: str) -> dict:
        prompt = (
            "You are a geopolitical risk analyst. Classify the following news headline "
            "for relevance to the Strait of Hormuz or Bab el-Mandeb maritime chokepoints "
            "and the Middle East geopolitical risk it represents.\n\n"
            "Return ONLY valid JSON with these fields:\n"
            "- 'relevance': 0.0 to 1.0 (how directly this affects Hormuz/Bab el-Mandeb)\n"
            "- 'intensity': 1 to 5 (severity of geopolitical risk)\n"
            "- 'direction': 'threat' | 'act' | 'crisis' | 'resolution'\n"
            "- 'affected_assets': list of likely affected markets\n"
            "- 'reasoning': one sentence\n\n"
            f"Headline: {headline}"
        )

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            return {"relevance": 0.5, "intensity": 3, "direction": "threat",
                    "affected_assets": [], "reasoning": str(e)}

    def _fallback_labels(self) -> pd.Series:
        rows = []
        for date_str, headline in _SAMPLE_HEADLINES:
            intensity_map = {
                "closure": 5, "blocks": 5, "surges": 4, "crisis": 5,
                "strikes": 4, "withdraws": 3, "normalises": 2, "stabilises": 2,
                "talks": 2, "pressure": 3, "disruption": 4, "seize": 3,
                "threaten": 3, "attack": 4, "kills": 5, "jumps": 3,
                "spike": 4, "pause": 3,
            }
            headline_lower = headline.lower()
            intensity = 3
            for word, val in intensity_map.items():
                if word in headline_lower:
                    intensity = max(intensity, val)
            rows.append({"date": date_str, "headline": headline, "intensity": intensity})

        return pd.DataFrame(rows).set_index("date")["intensity"]

    def fetch_index(self, start: str = "2000-01-01", end: str | None = None) -> pd.Series:
        if self._cached is not None:
            return self._cached

        if self._client:
            results = {}
            for date_str, headline in _SAMPLE_HEADLINES:
                if date_str < start or (end and date_str > end):
                    continue
                cls = self._classify_headline(headline)
                intensity = cls.get("intensity", 3) * cls.get("relevance", 0.5)
                results[date_str] = intensity
                time.sleep(0.3)

            series = pd.Series(results, name="gpr_custom").sort_index()
            series.index = pd.to_datetime(series.index)
        else:
            series = self._fallback_labels()
            series.index = pd.to_datetime(series.index)
            series = series[series.index >= start]
            if end:
                series = series[series.index <= end]
            series = series.sort_index()
            series.name = "gpr_custom"

        self._cached = series
        return series
