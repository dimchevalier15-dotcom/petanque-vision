"""Abstraction de la zone de jeu : `field.contains(position)`.

Aujourd'hui : polygone en pixels (ou plein cadre) ou polygone en mètres via homographie.
Demain : n'importe quelle géométrie terrain réelle, sans toucher au TrackManager/GameState
(ils ne connaissent que `contains`, `to_world` et `outline`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

import numpy as np

from app.petanque.config import FieldConfig
from app.petanque.models import Point


class Field(Protocol):
    def contains(self, position: Point) -> bool: ...

    def to_world(self, position: Point) -> Point | None:
        """Position en mètres si une calibration existe, sinon None."""
        ...

    def outline(self) -> list[Point] | None:
        """Contour en pixels pour le debug visuel."""
        ...


def _signed_distance_to_polygon(p: Point, poly: list[Point]) -> float:
    """> 0 à l'intérieur, < 0 à l'extérieur (distance au bord le plus proche)."""
    x, y = p
    inside = False
    best = math.inf
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
        dx, dy = x2 - x1, y2 - y1
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / seg2))
        best = min(best, math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)))
    return best if inside else -best


class PolygonField:
    """Polygone en pixels. `margin` > 0 tolère un peu d'extérieur (bruit de bbox)."""

    def __init__(self, polygon: list[Point], margin: float = 0.0) -> None:
        if len(polygon) < 3:
            raise ValueError("Un polygone de terrain demande au moins 3 points")
        self.polygon = [(float(x), float(y)) for x, y in polygon]
        self.margin = margin

    @classmethod
    def full_frame(cls, width: float, height: float, margin: float = 0.0) -> PolygonField:
        return cls([(0, 0), (width, 0), (width, height), (0, height)], margin)

    def signed_distance(self, position: Point) -> float:
        return _signed_distance_to_polygon(position, self.polygon)

    def contains(self, position: Point) -> bool:
        return self.signed_distance(position) >= -self.margin

    def to_world(self, position: Point) -> Point | None:
        return None

    def outline(self) -> list[Point] | None:
        return self.polygon


class HomographyField:
    """Zone de jeu définie en mètres (repère de scripts.calibrate_court) + homographie."""

    def __init__(self, h_mat: np.ndarray, world_polygon: list[Point], margin_m: float = 0.0) -> None:
        self.h = np.asarray(h_mat, dtype=np.float64)
        self.h_inv = np.linalg.inv(self.h)
        self.world_polygon = [(float(x), float(y)) for x, y in world_polygon]
        self.margin_m = margin_m

    @staticmethod
    def _apply(m: np.ndarray, p: Point) -> Point | None:
        v = m @ np.array([p[0], p[1], 1.0])
        if abs(v[2]) < 1e-12:
            return None
        return (float(v[0] / v[2]), float(v[1] / v[2]))

    def to_world(self, position: Point) -> Point | None:
        return self._apply(self.h, position)

    def contains(self, position: Point) -> bool:
        w = self.to_world(position)
        if w is None:
            return False
        return _signed_distance_to_polygon(w, self.world_polygon) >= -self.margin_m

    def outline(self) -> list[Point] | None:
        pts = [self._apply(self.h_inv, p) for p in self.world_polygon]
        return [p for p in pts if p is not None] or None


def build_field(cfg: FieldConfig, frame_size: tuple[int, int], repo_root: Path | None = None) -> Field:
    """Construit la zone de jeu. frame_size = (width, height)."""
    width, height = frame_size
    if cfg.calibration:
        path = Path(cfg.calibration)
        if not path.is_absolute() and repo_root is not None:
            path = repo_root / path
        h_mat = np.array(json.loads(path.read_text(encoding="utf-8"))["homography"], dtype=np.float64)
        if not cfg.world_polygon:
            raise ValueError("field.calibration demande aussi field.world_polygon (mètres)")
        return HomographyField(h_mat, [(p[0], p[1]) for p in cfg.world_polygon], cfg.margin)
    if cfg.polygon:
        if cfg.polygon_space == "normalized":
            pts = [(p[0] * width, p[1] * height) for p in cfg.polygon]
            margin = cfg.margin * math.hypot(width, height)
        elif cfg.polygon_space == "pixel":
            pts = [(p[0], p[1]) for p in cfg.polygon]
            margin = cfg.margin
        else:
            raise ValueError(f"polygon_space inconnu : {cfg.polygon_space}")
        return PolygonField(pts, margin)
    return PolygonField.full_frame(width, height, cfg.margin * math.hypot(width, height))
