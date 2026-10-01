"""Numérotation des boules lancées : « boule n° k » = k-ième boule jouée de la mène.

Une boule n'est numérotée que si un ThrowEvent de type THROW l'a désignée et qu'elle est restée en jeu.
Les mouvements UNKNOWN (à revoir), les boules simplement présentes, le cochonnet, les boules sorties du terrain :
pas de numéro. Aucune identité de joueur ici.

Positions : celles du ThrowEvent pendant le lancer, puis celles du track logique tant qu'il est observé,
sinon la dernière position connue (une boule arrêtée ne bouge pas toute seule).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.petanque.models import ObjectType, Observation, Point, distance
from app.petanque.throw_detector import ThrowEventDetector
from app.petanque.throws import MovementClass, ThrowEvent, ThrowResult, ThrowType
from app.petanque.track_manager import TrackManager

RING_FLIGHT = (0, 165, 255)  # BGR orange : en vol
RING_REST = (80, 220, 80)  # vert : arrêtée


@dataclass
class NumberedBall:
    number: int
    throw: ThrowEvent
    frames: list[int] = field(default_factory=list)  # positions connues (triées)
    points: list[Point] = field(default_factory=list)
    position_source: str = "tracked"  # tracked | last_seen_in_contact | relocated_from_detections

    def position_at(self, frame: int) -> Point | None:
        if frame < self.throw.start_frame or not self.frames:
            return None
        i = bisect.bisect_right(self.frames, frame) - 1
        return self.points[max(0, i)]

    def to_dict(self) -> dict[str, Any]:
        t = self.throw
        return {
            "ball_number": self.number,
            "throw_id": t.throw_id,
            "start_frame": t.start_frame,
            "end_frame": t.end_frame,
            "start_time_s": round(t.start_timestamp, 2),
            "final_position": [round(self.points[-1][0], 1), round(self.points[-1][1], 1)] if self.points
            else [round(t.final_position[0], 1), round(t.final_position[1], 1)],
            "position_source": self.position_source,
            "final_state": t.final_state.value,
            "confidence": round(t.confidence, 3),
            "track_ids": t.track_id_history,
            "reasons": t.detection_reasons,
        }


def _follow_displacements(pos: dict[int, Point], ids: set[int], td: ThrowEventDetector, tm: TrackManager) -> None:
    """Une boule arrêtée qui disparaît puis repart sous un autre track (poussée) reste la même boule numérotée :
    on ajoute la suite de sa trajectoire. Les mouvements sont pris dans l'ordre chronologique (poussées en chaîne)."""
    for m in sorted(td.movements, key=lambda m: m.start_frame):
        if m.classification != MovementClass.DISPLACED or m.continues_track is None:
            continue
        if tm.resolve(m.continues_track) not in ids:
            continue
        nxt = tm.tracks.get(tm.resolve(m.track_id))
        if nxt is None:
            continue
        ids.add(nxt.logical_track_id)
        for p in nxt.trajectory:
            if p.frame >= m.start_frame:
                pos[p.frame] = p.pos


def number_balls(td: ThrowEventDetector, tm: TrackManager) -> list[NumberedBall]:
    """Numérote, dans l'ordre chronologique, les lancers confirmés dont la boule est en jeu."""
    kept = [t for t in td.throws if t.event_type == ThrowType.THROW and t.final_state != ThrowResult.OUT_OF_PLAY]
    balls: list[NumberedBall] = []
    for n, t in enumerate(sorted(kept, key=lambda t: (t.start_frame, t.throw_id)), start=1):
        pos: dict[int, Point] = {f: (x, y) for f, x, y in t.trajectory}
        track = tm.tracks.get(tm.resolve(t.ball_track_id))
        if track is not None:
            for p in track.trajectory:  # après l'arrêt : le track continue (boule poussée par un autre lancer)
                if p.frame > t.end_frame:
                    pos[p.frame] = p.pos
        pos.setdefault(t.end_frame, t.final_position)
        _follow_displacements(pos, {tm.resolve(i) for i in t.track_id_history} | {tm.resolve(t.ball_track_id)}, td, tm)
        frames = sorted(pos)
        src = "last_seen_in_contact" if any(r.startswith("lost_in_contact") for r in t.final_state_reasons) else "tracked"
        balls.append(NumberedBall(n, t, frames, [pos[f] for f in frames], position_source=src))
    return balls


def relocate_contact_ends(balls: list[NumberedBall], frames: list[list[Observation]], radius_diam: float = 4.0,
                          after: tuple[int, int] = (45, 135), min_share: float = 0.5, claim_diam: float = 0.8) -> None:
    """Une boule « perdue au contact » continue de rouler un peu après sa dernière détection. On la retrouve dans les
    détections brutes : amas stable (boule immobile) proche de la dernière position, et pas déjà occupé par une
    autre boule numérotée. Aucun amas convaincant -> on garde la dernière position vue (source « last_seen_in_contact »)."""
    for b in balls:
        if b.position_source != "last_seen_in_contact":
            continue
        t = b.throw
        size = float(t.metrics.get("size_px", 50.0))
        f0, f1 = t.end_frame + after[0], min(len(frames) - 1, t.end_frame + after[1])
        if f1 <= f0:
            continue
        last = t.final_position
        clusters: list[list[float]] = []  # [x, y, n]
        for f in range(f0, f1 + 1):
            taken: set[int] = set()
            for o in frames[f]:
                if o.object_type != ObjectType.BALL:
                    continue
                c = ((o.x1 + o.x2) / 2.0, (o.y1 + o.y2) / 2.0)
                if distance(c, last) > radius_diam * size:
                    continue
                for i, k in enumerate(clusters):
                    if i not in taken and distance((k[0], k[1]), c) <= 0.6 * size:
                        k[0], k[1], k[2] = (k[0] * k[2] + c[0]) / (k[2] + 1), (k[1] * k[2] + c[1]) / (k[2] + 1), k[2] + 1
                        taken.add(i)
                        break
                else:
                    clusters.append([c[0], c[1], 1.0])
                    taken.add(len(clusters) - 1)
        span = f1 - f0 + 1
        mid = (f0 + f1) // 2  # où sont les autres boules numérotées à ce moment-là (elles ont pu être poussées depuis)
        others = [q for q in (o.position_at(mid) for o in balls if o is not b) if q is not None]
        ok = [k for k in clusters if k[2] / span >= min_share
              and all(distance((k[0], k[1]), q) > claim_diam * size for q in others)]
        if not ok:
            continue
        best = min(ok, key=lambda k: distance((k[0], k[1]), last))
        b.frames.append(t.end_frame + after[0] // 3)
        b.points.append((best[0], best[1]))
        b.position_source = "relocated_from_detections"


RING_JACK = (255, 0, 255)  # magenta : le but, distinct des boules


class BallLabelRenderer:
    """Dessine uniquement les boules numérotées (anneau + numéro) et un compteur « boules en jeu ».

    Anneau orange = en vol, vert = arrêtée, pointillés = boule arrêtée au contact d'une autre dont la position
    exacte n'a pas pu être retrouvée (position approchée)."""

    def __init__(self, balls: list[NumberedBall], scale: float = 1.0) -> None:
        self.balls = balls
        self.scale = scale

    def draw(self, img: np.ndarray, frame: int, fps: float, jack: Any = None) -> np.ndarray:
        import cv2

        s = self.scale
        out = img
        live = 0
        for b in self.balls:
            p = b.position_at(frame)
            if p is None:
                continue
            live += 1
            t = b.throw
            size = float(t.metrics.get("size_px", 50.0)) * s
            in_flight = t.start_frame <= frame <= t.end_frame
            color = RING_FLIGHT if in_flight else RING_REST
            c = (int(p[0] * s), int(p[1] * s))
            r = max(6, int(size * 0.85))
            approx = b.position_source == "last_seen_in_contact" and frame > t.end_frame  # position seulement approchée
            if approx:  # anneau en pointillés : le numéro est sûr, l'emplacement exact non
                for a in range(0, 360, 40):
                    cv2.ellipse(out, c, (r, r), 0, a, a + 22, (0, 0, 0), max(3, int(size * 0.16)), cv2.LINE_AA)
                    cv2.ellipse(out, c, (r, r), 0, a, a + 22, color, max(2, int(size * 0.10)), cv2.LINE_AA)
            else:
                cv2.circle(out, c, r, (0, 0, 0), max(3, int(size * 0.16)), cv2.LINE_AA)
                cv2.circle(out, c, r, color, max(2, int(size * 0.10)), cv2.LINE_AA)
            label = str(b.number)
            scale = max(0.6, size / 28.0)
            thick = max(2, int(scale * 2))
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            # pastille numérotée en haut à droite de l'anneau
            ox, oy = c[0] + int(r * 0.85), c[1] - int(r * 0.85)
            pad = max(3, int(size * 0.12))
            p0, p1 = (ox - pad, oy - th - pad), (ox + tw + pad, oy + base + pad)
            cv2.rectangle(out, p0, p1, color, -1)
            cv2.rectangle(out, p0, p1, (0, 0, 0), 1)
            cv2.putText(out, label, (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
        if jack is not None:
            self._draw_jack(out, jack)
        hud = f"Boules en jeu : {live}   t={frame / fps:5.1f}s"
        hs = max(0.7, out.shape[1] / 1100.0)
        cv2.putText(out, hud, (int(14 * hs), int(40 * hs)), cv2.FONT_HERSHEY_SIMPLEX, hs, (0, 0, 0), int(hs * 4), cv2.LINE_AA)
        cv2.putText(out, hud, (int(14 * hs), int(40 * hs)), cv2.FONT_HERSHEY_SIMPLEX, hs, (255, 255, 255), int(hs * 2), cv2.LINE_AA)
        return out

    def _draw_jack(self, out: np.ndarray, jack: Any) -> None:
        """Le but : anneau magenta + croix + « BUT » ; pointillés si la position n'est pas observée (cachée)."""
        import cv2

        s = self.scale
        c = (int(jack.pos[0] * s), int(jack.pos[1] * s))
        r = max(8, int(40 * s))
        thick = max(2, int(5 * s))
        if jack.status == "seen":
            cv2.circle(out, c, r, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.circle(out, c, r, RING_JACK, thick, cv2.LINE_AA)
        else:
            for a in range(0, 360, 40):
                cv2.ellipse(out, c, (r, r), 0, a, a + 22, (0, 0, 0), thick + 2, cv2.LINE_AA)
                cv2.ellipse(out, c, (r, r), 0, a, a + 22, RING_JACK, thick, cv2.LINE_AA)
        k = int(r * 0.45)
        for d in ((k, 0), (0, k)):
            cv2.line(out, (c[0] - d[0], c[1] - d[1]), (c[0] + d[0], c[1] + d[1]), (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.line(out, (c[0] - d[0], c[1] - d[1]), (c[0] + d[0], c[1] + d[1]), RING_JACK, thick - 1 if thick > 2 else 2, cv2.LINE_AA)
        scale = max(0.6, 38 * s / 28.0)
        tx = int(scale * 2)
        (tw, th), base = cv2.getTextSize("BUT", cv2.FONT_HERSHEY_SIMPLEX, scale, tx)
        ox, oy = c[0] - tw // 2, c[1] + r + th + 8
        cv2.rectangle(out, (ox - 4, oy - th - 4), (ox + tw + 4, oy + base + 4), RING_JACK, -1)
        cv2.putText(out, "BUT", (ox, oy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), tx, cv2.LINE_AA)
