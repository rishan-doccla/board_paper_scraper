from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List


class ResultsStore:
    """Light‑weight JSON wrapper for persisting crawler / analysis output."""

    def __init__(self, file_path: str | Path) -> None:
        self.path = Path(file_path)
        self._data: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------ public
    @property
    def board_papers(self) -> List[Dict[str, Any]]:
        return self._data.setdefault("board_papers", [])

    def update(self, papers: List[Dict[str, Any]]) -> None:
        """Replace the stored papers and persist to disk."""
        self._data["last_run"] = _dt.datetime.now().isoformat()
        self._data["board_papers"] = papers
        self._save()

    # -------------------------------------------------------------- internals
    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            with self.path.open() as fh:
                return json.load(fh)
        return {"last_run": None, "board_papers": []}

    def _save(self) -> None:
        with self.path.open("w") as fh:
            json.dump(self._data, fh, indent=2)
