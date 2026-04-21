"""Live-session position tracker (persisted to JSON)."""
from __future__ import annotations

import json
from pathlib import Path


class PositionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def load(self) -> dict[str, int]:
        return json.loads(self.path.read_text(encoding="utf-8") or "{}")

    def save(self, positions: dict[str, int]) -> None:
        self.path.write_text(json.dumps(positions, indent=2), encoding="utf-8")
