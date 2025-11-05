from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


def _registry_path(base_dir: Path | None = None) -> Path:
    """Return path to local registry file creating directories as needed."""
    root = base_dir or Path.cwd()
    reg_dir = root / ".discovery"
    reg_dir.mkdir(parents=True, exist_ok=True)
    return reg_dir / "sources.json"


def load_registry(base_dir: Path | None = None) -> Dict[str, Any]:
    """Load source registry; return empty dict on first run or parse errors."""
    path = _registry_path(base_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_registry(registry: Dict[str, Any], base_dir: Path | None = None) -> None:
    """Persist the entire registry to disk as indented JSON."""
    path = _registry_path(base_dir)
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def upsert_source(alias: str, source: Dict[str, Any], base_dir: Path | None = None) -> None:
    """Insert or update a source definition under the given alias."""
    reg = load_registry(base_dir)
    reg[alias] = source
    save_registry(reg, base_dir)


def get_source(alias: str, base_dir: Path | None = None) -> Dict[str, Any] | None:
    """Fetch a source by alias; returns None if not registered."""
    reg = load_registry(base_dir)
    return reg.get(alias)


