"""Scénarios synthétiques : objets avec vérité terrain (gt_id) + simulation d'un ByteTrack imparfait.

Sert à tester la *logique* (le comportement réel est validé sur vidéos). ByteTrack simulé :
un objet garde son ID tant qu'il n'est pas manquant plus de `bt_patience` frames ; au-delà,
il réapparaît avec un NOUVEL ID (le défaut observé sur les vraies vidéos).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from app.petanque.config import PipelineConfig
from app.petanque.field import PolygonField
from app.petanque.models import ObjectType, Observation, Point
from app.petanque.track_manager import FrameResult, TrackManager

BALL = 60.0  # diamètre apparent (px)


def static(x: float, y: float) -> Callable[[int], Point]:
    return lambda f: (x, y)


def linear(p0: Point, v: Point, start: int = 0) -> Callable[[int], Point]:
    """Position constante avant `start`, puis mouvement uniforme."""
    return lambda f: (p0[0] + v[0] * max(0, f - start), p0[1] + v[1] * max(0, f - start))


def rolling(p0: Point, v0: Point, start: int, decay: float = 0.93) -> Callable[[int], Point]:
    """Immobile jusqu'à `start`, lancé à v0 (px/frame) puis freiné (roule puis s'arrête)."""

    def pos(f: int) -> Point:
        k = max(0, f - start)
        if k == 0:
            return p0
        s = decay * (1.0 - decay**k) / (1.0 - decay)
        return (p0[0] + v0[0] * s, p0[1] + v0[1] * s)

    return pos


def piecewise(*segments: tuple[int, Callable[[int], Point]]) -> Callable[[int], Point]:
    """(frame_début, fonction) : applique la dernière fonction dont frame_début <= f."""

    def pos(f: int) -> Point:
        fn = segments[0][1]
        for start, g in segments:
            if f >= start:
                fn = g
        return fn(f)

    return pos


@dataclass
class SimObject:
    gt_id: int
    path: Callable[[int], Point]
    first_frame: int = 0
    last_frame: int = 10**9
    hidden: list[tuple[int, int]] = field(default_factory=list)  # [début, fin] inclus : non détecté
    size: float = BALL
    object_type: ObjectType = ObjectType.BALL

    def present(self, f: int) -> bool:
        if not (self.first_frame <= f <= self.last_frame):
            return False
        return not any(a <= f <= b for a, b in self.hidden)


def simulate(
    objects: list[SimObject],
    n_frames: int,
    noise: float = 0.6,
    seed: int = 0,
    bt_patience: int = 3,
    confidence: float = 0.9,
) -> list[list[Observation]]:
    rng = random.Random(seed)
    next_bt = 1
    bt_of: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    frames: list[list[Observation]] = []
    for f in range(n_frames):
        obs: list[Observation] = []
        for o in objects:
            if not o.present(f):
                continue
            if o.gt_id not in bt_of or f - last_seen[o.gt_id] > bt_patience + 1:
                bt_of[o.gt_id] = next_bt
                next_bt += 1
            last_seen[o.gt_id] = f
            x, y = o.path(f)
            x += rng.gauss(0, noise)
            y += rng.gauss(0, noise)
            s = o.size + rng.gauss(0, noise * 0.5)
            obs.append(
                Observation(
                    frame=f, x1=x - s / 2, y1=y - s / 2, x2=x + s / 2, y2=y + s / 2,
                    confidence=confidence, object_type=o.object_type,
                    bt_id=bt_of[o.gt_id], det_id=len(obs), gt_id=o.gt_id,
                )
            )
        rng.shuffle(obs)
        obs = [Observation(**{**o.__dict__, "det_id": i}) for i, o in enumerate(obs)]
        frames.append(obs)
    return frames


def default_field(width: int = 1920, height: int = 1080) -> PolygonField:
    return PolygonField.full_frame(width, height)


def run(frames: list[list[Observation]], cfg: PipelineConfig | None = None,
        field_=None) -> tuple[TrackManager, list[FrameResult]]:
    tm = TrackManager(cfg or PipelineConfig(), field_ or default_field())
    results = [tm.update(f, obs) for f, obs in enumerate(frames)]
    return tm, results


def gt_assignments(results: list[FrameResult]) -> list[tuple[int, int, int]]:
    """(frame, logical_track_id, gt_id) pour chaque observation vue par le TrackManager."""
    out = []
    for r in results:
        for tid, o in r.visible.items():
            if o.gt_id is not None:
                out.append((r.frame, tid, o.gt_id))
    return out


def raw_assignments(frames: list[list[Observation]]) -> list[tuple[int, int, int]]:
    return [(f, o.bt_id, o.gt_id) for f, obs in enumerate(frames) for o in obs if o.bt_id is not None and o.gt_id is not None]


def distance_px(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
