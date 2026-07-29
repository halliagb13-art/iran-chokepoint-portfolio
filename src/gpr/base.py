from abc import ABC, abstractmethod
import pandas as pd


class GPRAdapter(ABC):
    @abstractmethod
    def fetch_index(self, start: str = "2000-01-01", end: str | None = None) -> pd.Series:
        pass

    @abstractmethod
    def name(self) -> str:
        pass
