"""Log structuré (JSON Lines) des événements importants, pour analyser sans lire le code."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.petanque.models import TrackEvent

logger = logging.getLogger("petanque.events")


class EventLog:
    """Garde les événements en mémoire et, si `path` est donné, les écrit en JSONL."""

    def __init__(self, path: Path | None = None) -> None:
        self.events: list[TrackEvent] = []
        self._fh = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("w", encoding="utf-8")

    def emit(self, event: TrackEvent) -> TrackEvent:
        self.events.append(event)
        line = json.dumps(event.to_dict(), ensure_ascii=False)
        logger.debug(line)
        if self._fh is not None:
            self._fh.write(line + "\n")
        return event

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def of_type(self, *names: str) -> list[TrackEvent]:
        return [e for e in self.events if e.event in names]
