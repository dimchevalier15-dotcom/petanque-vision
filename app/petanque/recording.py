"""Enregistrement / relecture des observations (YOLO + ByteTrack) en JSON Lines.

YOLO sur des vidéos 4K est lent ; la logique TrackManager/GameState est rapide. On enregistre
donc une fois les observations, puis on rejoue la logique autant de fois que nécessaire
(itération rapide, résultats reproductibles, comparaison avant/après sur les mêmes données).

Format : 1re ligne `{"type": "meta", ...}`, puis une ligne par frame :
  {"frame": 12, "obs": [{"bbox": [x1,y1,x2,y2], "conf": 0.87, "cls": "Boule", "bt": 5}, ...]}
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from app.petanque.models import ObjectType, Observation

_CLASS_TO_TYPE = {"boule": ObjectType.BALL, "cochonnet": ObjectType.JACK}
_TYPE_TO_CLASS = {ObjectType.BALL: "Boule", ObjectType.JACK: "Cochonnet"}


def object_type_from_class_name(name: str) -> ObjectType | None:
    """None = classe non suivie (ex. « Cercle de jeu »)."""
    return _CLASS_TO_TYPE.get(name.strip().lower())


class ObservationWriter:
    def __init__(self, path: Path, meta: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")
        self._fh.write(json.dumps({"type": "meta", **meta}) + "\n")

    def write_frame(self, frame: int, observations: Iterable[Observation]) -> None:
        obs = [
            {
                "bbox": [round(v, 1) for v in o.bbox],
                "conf": round(o.confidence, 3),
                "cls": _TYPE_TO_CLASS[o.object_type],
                "bt": o.bt_id,
                **({"gt": o.gt_id} if o.gt_id is not None else {}),
            }
            for o in observations
        ]
        self._fh.write(json.dumps({"frame": frame, "obs": obs}) + "\n")

    def close(self) -> None:
        self._fh.close()


def write_observations(path: Path, meta: dict[str, Any], frames: list[list[Observation]]) -> None:
    w = ObservationWriter(path, meta)
    try:
        for i, obs in enumerate(frames):
            w.write_frame(i, obs)
    finally:
        w.close()


def read_observations(path: Path) -> tuple[dict[str, Any], list[list[Observation]]]:
    meta: dict[str, Any] = {}
    frames: dict[int, list[Observation]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("type") == "meta":
                meta = row
                continue
            obs = []
            for i, o in enumerate(row["obs"]):
                x1, y1, x2, y2 = o["bbox"]
                obs.append(
                    Observation(
                        row["frame"], x1, y1, x2, y2, o["conf"], ObjectType(o.get("type") or _CLASS_TO_TYPE[o["cls"].lower()]),
                        o.get("bt"), i, o.get("gt"),
                    )
                )
            frames[row["frame"]] = obs
    n = max(frames) + 1 if frames else 0
    return meta, [frames.get(i, []) for i in range(n)]
