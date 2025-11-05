from __future__ import annotations

from pathlib import Path


def resolve_output_dir(base: str, source_name: str) -> Path:
    path = Path(base).expanduser().resolve() / source_name
    path.mkdir(parents=True, exist_ok=True)
    return path







