from __future__ import annotations

from typing import Iterable, Dict, Optional, Protocol
import pandas as pd


class SourceConnector(Protocol):
    """Protocol for data source connectors.

    Implementations provide basic capabilities used by the agent:
    - enumerate available tables
    - fetch table schema
    - sample rows with different strategies
    """
    def list_tables(self) -> Iterable[str]: ...
    def get_schema(self, table: str) -> Dict[str, str]: ...
    def sample(
        self,
        table: str,
        limit: int,
        method: str = "random",
        stratify_by: Optional[str] = None,
    ) -> pd.DataFrame: ...


