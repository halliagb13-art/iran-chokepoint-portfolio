import pytest
import pandas as pd
from src.events import get_events, EVENTS_DF, HISTORICAL_EVENTS


class TestEvents:
    def test_events_loaded(self):
        assert len(HISTORICAL_EVENTS) >= 15

    def test_event_fields(self):
        for evt in HISTORICAL_EVENTS:
            assert "date" in evt
            assert "name" in evt
            assert "chokepoint" in evt
            assert "category" in evt
            assert "severity" in evt
            assert evt["severity"] in [1, 2, 3, 4, 5]

    def test_get_events_filter_chokepoint(self):
        hormuz = get_events(chokepoint="Hormuz")
        assert all(hormuz["chokepoint"] == "Hormuz")

    def test_get_events_filter_severity(self):
        severe = get_events(min_severity=4)
        assert all(severe["severity"] >= 4)

    def test_get_events_filter_category(self):
        threats = get_events(category="threat")
        assert all(threats["category"] == "threat")

    def test_dataframe_has_required_columns(self):
        required = ["date", "name", "chokepoint", "category", "severity"]
        for col in required:
            assert col in EVENTS_DF.columns

    def test_dates_parsed_correctly(self):
        dates = pd.to_datetime(EVENTS_DF["date"])
        assert dates.is_monotonic_increasing
