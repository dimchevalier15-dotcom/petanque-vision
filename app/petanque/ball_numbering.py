"""Numérotation des boules lancées : « boule n° k » = k-ième boule jouée de la mène.

Une boule n'est numérotée que si un ThrowEvent de type THROW l'a désignée et qu'elle est restée en jeu.
Les mouvements UNKNOWN (à revoir), les boules simplement présentes, le cochonnet, les boules sorties du terrain :
pas de numéro. Aucune identité de joueur ici.

Positions : trajectoire du lancer, puis repos figé (sans dérive du track YOLO). Après une collision
détectée, tireur et cible sont suivies via les détections YOLO assignées selon la normale de choc
(victime poussée le long de la normale, tireur surtout le long de la tangente).
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.petanque.models import ObjectType, Observation, Point, distance
from app.petanque.throw_detector import ThrowEventDetector
from app.petanque.throws import CollisionCandidate, MovementClass, ThrowEvent, ThrowResult, ThrowType
from app.petanque.track_manager import TrackManager

_COLLISION_MIN_CONF = 0.55
_COLLISION_TRACK_FRAMES = 150
_COLLISION_SEARCH_DIAM = 5.0

RING_FLIGHT = (0, 165, 255)  # BGR orange : en vol
RING_REST = (80, 220, 80)  # vert : arrêtée


@dataclass
class NumberedBall:
    number: int
    throw: ThrowEvent
    frames: list[int] = field(default_factory=list)  # positions connues (triées)
    points: list[Point] = field(default_factory=list)
    position_source: str = "tracked"  # tracked | last_seen_in_contact | relocated_from_detections
    retired_after: int | None = None  # carreau / remplacement : ne plus afficher après cette frame

    def position_at(self, frame: int) -> Point | None:
        if frame < self.throw.start_frame or not self.frames:
            return None
        if self.retired_after is not None and frame > self.retired_after:
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
            "retired_after": self.retired_after,
        }


def _last_observed_frame(tm: TrackManager, t: ThrowEvent) -> int:
    last = t.end_frame
    for tr in tm.tracks.values():
        if tr.trajectory:
            last = max(last, tr.trajectory[-1].frame)
    return last


def _hold_at_rest(pos: dict[int, Point], from_frame: int, to_frame: int, rest: Point) -> None:
    for f in range(from_frame, to_frame + 1):
        pos.setdefault(f, rest)


def _merge_positions(ball: NumberedBall, extra: dict[int, Point]) -> None:
    pos = {f: p for f, p in zip(ball.frames, ball.points)}
    pos.update(extra)
    ball.frames = sorted(pos)
    ball.points = [pos[f] for f in ball.frames]


def _merge_collision_segment(ball: NumberedBall, from_frame: int, segment: dict[int, Point]) -> None:
    """À partir du choc : remplace tout l'historique (repos figé, DISPLACED, relocate) par le suivi collision."""
    pos = {f: p for f, p in zip(ball.frames, ball.points) if f < from_frame}
    pos.update(segment)
    ball.frames = sorted(pos)
    ball.points = [pos[f] for f in ball.frames]


def _hold_segment_tail(segment: dict[int, Point], last_frame: int) -> None:
    if not segment:
        return
    end = max(segment)
    p = segment[end]
    for f in range(end + 1, last_frame + 1):
        segment[f] = p


def _ball_identity_ids(ball: NumberedBall, tm: TrackManager) -> set[int]:
    t = ball.throw
    return {tm.resolve(i) for i in t.track_id_history} | {tm.resolve(t.ball_track_id)}


def _ball_for_logical(balls: list[NumberedBall], tm: TrackManager, logical_id: int) -> NumberedBall | None:
    want = tm.resolve(logical_id)
    for b in balls:
        if want in _ball_identity_ids(b, tm):
            return b
    return None


def _unit(v: Point) -> Point:
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n) if n > 1e-9 else (0.0, 0.0)


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def _scale(v: Point, k: float) -> Point:
    return (v[0] * k, v[1] * k)


def _throw_pos_at(traj: list[tuple], frame: int) -> Point | None:
    last: Point | None = None
    for f, x, y in traj:
        if f == frame:
            return (x, y)
        if f <= frame:
            last = (x, y)
    return last


def _approach_dir(traj: list[tuple], frame: int, back: int = 8) -> Point | None:
    pts = [(f, x, y) for f, x, y in traj if frame - back <= f < frame]
    if len(pts) < 2:
        pts = [(f, x, y) for f, x, y in traj if f <= frame][-2:]
    if len(pts) < 2:
        return None
    return (pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2])


def _motion_align_cost(obs: Point, prev: Point, pred: Point, dir_hat: Point) -> float:
    cost = distance(obs, pred)
    step = (obs[0] - prev[0], obs[1] - prev[1])
    if math.hypot(step[0], step[1]) > 1e-6 and _dot(_unit(step), dir_hat) < 0:
        cost += 80.0
    return cost


def _assign_pair(
    cands: list[Point],
    pa_pred: Point,
    pv_pred: Point,
    pa_prev: Point,
    pv_prev: Point,
    v_t: Point,
    n: Point,
) -> tuple[Point | None, Point | None]:
    if not cands:
        return None, None
    if len(cands) == 1:
        c = cands[0]
        if _motion_align_cost(c, pa_prev, pa_pred, v_t) <= _motion_align_cost(c, pv_prev, pv_pred, n):
            return c, None
        return None, c
    cost0 = _motion_align_cost(cands[0], pa_prev, pa_pred, v_t) + _motion_align_cost(cands[1], pv_prev, pv_pred, n)
    cost1 = _motion_align_cost(cands[1], pa_prev, pa_pred, v_t) + _motion_align_cost(cands[0], pv_prev, pv_pred, n)
    if cost1 < cost0:
        return cands[1], cands[0]
    return cands[0], cands[1]


def _track_collision_pair(
    attacker: NumberedBall,
    victim: NumberedBall,
    collision: CollisionCandidate,
    frames: list[list[Observation]],
    size: float,
    all_balls: list[NumberedBall],
) -> None:
    """Suit les deux boules après le choc : normale = tireur → cible, victime le long de +n, tireur le long de la tangente."""
    cf = collision.frame
    f_end = min(len(frames) - 1, cf + _COLLISION_TRACK_FRAMES)
    traj = attacker.throw.trajectory
    src = _throw_pos_at(traj, cf) or collision.position
    vic = victim.position_at(cf - 1) or victim.position_at(cf) or collision.position
    n = _unit((vic[0] - src[0], vic[1] - src[1]))
    raw_v = _approach_dir(traj, cf) or n
    vn = _dot(raw_v, n)
    v_t = _scale(n, vn)
    v_t = (raw_v[0] - v_t[0], raw_v[1] - v_t[1])
    if math.hypot(v_t[0], v_t[1]) < 1e-6:
        v_t = (-n[1], n[0])
    else:
        v_t = _unit(v_t)

    step = max(size * 0.12, 4.0)
    max_jump = 2.2 * size
    extra_a: dict[int, Point] = {}
    extra_v: dict[int, Point] = {}
    pa, pv = src, vic
    extra_a[cf] = pa
    extra_v[cf] = pv
    still = 0
    radius = _COLLISION_SEARCH_DIAM * size

    for f in range(cf + 1, f_end + 1):
        pa_pred = _add(pa, _scale(v_t, step))
        pv_pred = _add(pv, _scale(n, step * 0.85))
        occupied: list[Point] = []
        for b in all_balls:
            if b is attacker or b is victim:
                continue
            p = b.position_at(f)
            if p is not None:
                occupied.append(p)
        cands: list[Point] = []
        for o in frames[f]:
            if o.object_type != ObjectType.BALL:
                continue
            c = o.center
            if any(distance(c, o) < 0.85 * size for o in occupied):
                continue
            if distance(c, pa_pred) <= radius or distance(c, pv_pred) <= radius:
                cands.append(c)
        if len(cands) > 2:
            mid = ((pa_pred[0] + pv_pred[0]) / 2, (pa_pred[1] + pv_pred[1]) / 2)
            cands = sorted(cands, key=lambda c: distance(c, mid))[:2]
        oa, ov = _assign_pair(cands, pa_pred, pv_pred, pa, pv, v_t, n)
        if oa is not None and distance(oa, pa) > max_jump and distance(oa, pa_pred) > max_jump:
            oa = None
        if ov is not None and distance(ov, pv) > max_jump and distance(ov, pv_pred) > max_jump:
            ov = None
        moved = 0.0
        if oa is not None:
            moved += distance(oa, pa)
            pa = oa
            extra_a[f] = pa
        if ov is not None:
            moved += distance(ov, pv)
            pv = ov
            extra_v[f] = pv
        if oa is None and ov is None:
            still += 1
        elif moved < size * 0.06:
            still += 1
        else:
            still = 0
        if still >= 10:
            break

    clip_end = len(frames) - 1
    _hold_segment_tail(extra_a, clip_end)
    _hold_segment_tail(extra_v, clip_end)
    _merge_collision_segment(attacker, cf, extra_a)
    _merge_collision_segment(victim, cf, extra_v)
    attacker.position_source = "tracked"
    victim.position_source = "tracked"


def apply_collision_display(
    balls: list[NumberedBall],
    td: ThrowEventDetector,
    tm: TrackManager,
    frames: list[list[Observation]],
) -> None:
    """Ré-ancre les anneaux après choc sur les bonnes bbox (géométrie de la trajectoire d'approche)."""
    for c in td.collisions:
        if c.target_type != "BALL" or c.confidence < _COLLISION_MIN_CONF:
            continue
        if c.kind not in ("impact_on_static", "moving_moving"):
            continue
        attacker = _ball_for_logical(balls, tm, c.source_ball)
        victim = _ball_for_logical(balls, tm, c.target_ball)
        if attacker is None or victim is None or attacker is victim:
            continue
        size = float(attacker.throw.metrics.get("size_px", 50.0))
        _track_collision_pair(attacker, victim, c, frames, size, balls)


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
        pos.setdefault(t.end_frame, t.final_position)
        identity_ids = {tm.resolve(i) for i in t.track_id_history} | {tm.resolve(t.ball_track_id)}
        last_frame = _last_observed_frame(tm, t)
        _hold_at_rest(pos, t.end_frame + 1, last_frame, t.final_position)
        _follow_displacements(pos, identity_ids, td, tm)
        frames = sorted(pos)
        src = "last_seen_in_contact" if any(r.startswith("lost_in_contact") for r in t.final_state_reasons) else "tracked"
        balls.append(NumberedBall(n, t, frames, [pos[f] for f in frames], position_source=src))
    _apply_carreau_supersession(balls, td, tm)
    return balls


def _apply_carreau_supersession(balls: list[NumberedBall], td: ThrowEventDetector, tm: TrackManager) -> None:
    """Une nouvelle boule jouée tout près d'une boule dont le track vient de disparaître = carreau :
    l'ancienne (souvent sortie / remplacée) ne reste plus affichée à cet endroit."""
    tc = td.tc
    fps = td.fps
    win = max(1, round(tc.rest_origin_window * fps))
    for later in sorted(td.throws, key=lambda t: t.start_frame):
        if later.event_type not in (ThrowType.THROW, ThrowType.UNKNOWN):
            continue
        if later.metrics.get("reach_diam", 0) < tc.min_travel:
            continue
        size = float(later.metrics.get("size_px", 50.0))
        for b in balls:
            if b.retired_after is not None or later.start_frame <= b.throw.end_frame:
                continue
            track = tm.tracks.get(tm.resolve(b.throw.ball_track_id))
            if track is None:
                continue
            anchor = b.points[-1] if b.points else b.throw.final_position
            last_seen = track.last_detection_frame
            if later.start_frame - last_seen > win:
                continue
            if distance(later.initial_position, anchor) > tc.rest_origin_distance * size:
                continue
            b.retired_after = last_seen


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
            if b.retired_after is not None and frame > b.retired_after:
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
