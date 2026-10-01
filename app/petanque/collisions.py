"""Collisions probables entre objets suivis (aucune simulation physique).

But unique : ne pas prendre une boule *poussée* pour une boule *lancée*.

Principe, par paire d'objets visibles :

    1. contact possible : centres à moins de `contact_distance` diamètres, l'un des deux en mouvement ;
    2. on retient l'instant de distance minimale (`f_min`) ;
    3. on attend `collision_settle_frames` pour voir la *réponse* : vitesse / direction avant et après ;
    4. collision probable si la cible réagit (elle part), ou si la source est déviée / stoppée net
       alors que le contact est net. Sinon : un simple passage, pas de candidat.

La proximité seule n'est jamais une collision : il faut une réponse cinématique observée.
Toutes les vitesses sont en diamètres / frame (indépendant de la perspective).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.petanque.config import ThrowConfig
from app.petanque.models import BallTrack, Point, TrackPoint, distance
from app.petanque.motion import estimate_velocity
from app.petanque.throws import CollisionCandidate
from app.petanque.track_manager import TrackManager

STATIC_SPEED = 0.08  # diam/frame : la cible était immobile avant le contact


@dataclass
class _Watch:
    opened: int
    f_min: int
    d_min: float


def _points(track: BallTrack, f0: int, f1: int) -> list[TrackPoint]:
    out: list[TrackPoint] = []
    for p in reversed(track.trajectory):
        if p.frame < f0:
            break
        if p.frame <= f1:
            out.append(p)
    out.reverse()
    return out


def _velocity(track: BallTrack, f0: int, f1: int) -> Point | None:
    pts = _points(track, f0, f1)
    return estimate_velocity(pts) if len(pts) >= 2 else None


def _position_at(track: BallTrack, frame: int, max_gap: int = 4) -> Point | None:
    """Position observée (ou interpolée sur un petit trou) ; None sinon. Jamais une prédiction."""
    before = after = None
    for p in reversed(track.trajectory):
        if p.frame == frame:
            return p.pos
        if p.frame > frame:
            after = p
        else:
            before = p
            break
    if before is None or after is None or after.frame - before.frame > max_gap:
        return None
    k = (frame - before.frame) / (after.frame - before.frame)
    return (before.x + (after.x - before.x) * k, before.y + (after.y - before.y) * k)


def _speed(v: Point | None, size: float) -> float | None:
    return None if v is None else math.hypot(*v) / max(size, 1e-6)


def _angle(u: Point, v: Point) -> float:
    nu, nv = math.hypot(*u), math.hypot(*v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    c = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv)))
    return math.degrees(math.acos(c))


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


class CollisionDetector:
    def __init__(self, cfg: ThrowConfig) -> None:
        self.cfg = cfg
        self._watch: dict[tuple[int, int], _Watch] = {}
        self._cooldown: dict[tuple[int, int], int] = {}
        self._next_id = 1

    # ------------------------------------------------------------------ API
    def update(self, frame: int, tm: TrackManager) -> list[CollisionCandidate]:
        """À appeler une fois par frame, après tm.update. Retourne les collisions évaluées à cette frame."""
        cfg = self.cfg
        vis = sorted((t for t in tm.live_tracks() if t.frames_missing(frame) == 0), key=lambda t: t.logical_track_id)
        for i, a in enumerate(vis):
            for b in vis[i + 1:]:
                key = (a.logical_track_id, b.logical_track_id)
                d = distance(a.current_position, b.current_position) / ((a.size + b.size) / 2.0)
                if d > cfg.contact_distance:
                    continue
                w = self._watch.get(key)
                if w is None:
                    if frame < self._cooldown.get(key, -1):
                        continue
                    if max(a.speed_diam, b.speed_diam) < cfg.collision_min_source_speed:
                        continue  # deux objets immobiles qui se touchent : pas une collision
                    self._watch[key] = _Watch(frame, frame, d)
                elif d < w.d_min - 0.05:
                    w.f_min, w.d_min = frame, d
        due = [k for k, w in self._watch.items()
               if frame - w.f_min >= cfg.collision_settle_frames or frame - w.opened >= cfg.collision_settle_frames + 30]
        return self._evaluate_keys(due, frame, tm)

    def flush(self, frame: int, tm: TrackManager) -> list[CollisionCandidate]:
        """Fin de flux : évalue ce qui reste (réponse possiblement incomplète -> confiance pénalisée)."""
        return self._evaluate_keys(list(self._watch), frame, tm)

    def remap(self, gone: int, survivor: int) -> None:
        """Le TrackManager a fusionné `gone` dans `survivor`."""
        for store in (self._watch, self._cooldown):
            for key in [k for k in store if gone in k]:
                val = store.pop(key)
                new = tuple(sorted(survivor if x == gone else x for x in key))
                if new[0] != new[1]:
                    store.setdefault(new, val)  # type: ignore[arg-type]

    # ------------------------------------------------------------ évaluation
    def _evaluate_keys(self, keys: list[tuple[int, int]], frame: int, tm: TrackManager) -> list[CollisionCandidate]:
        out: list[CollisionCandidate] = []
        for key in keys:
            w = self._watch.pop(key)
            self._cooldown[key] = frame + self.cfg.collision_cooldown_frames
            a, b = tm.tracks.get(tm.resolve(key[0])), tm.tracks.get(tm.resolve(key[1]))
            if a is None or b is None or a is b:
                continue
            c = self.evaluate(a, b, w.f_min, w.d_min, frame)
            if c is not None:
                c.collision_id = self._next_id
                self._next_id += 1
                out.append(c)
        return out

    def evaluate(self, a: BallTrack, b: BallTrack, f_c: int, d_min: float, frame: int) -> CollisionCandidate | None:
        cfg = self.cfg
        win = cfg.collision_window_frames

        def kin(t: BallTrack) -> tuple[Point | None, Point | None]:
            return _velocity(t, f_c - win, f_c - 1), _velocity(t, f_c + 1, f_c + win)

        va_pre, va_post = kin(a)
        vb_pre, vb_post = kin(b)
        sa_pre, sb_pre = _speed(va_pre, a.size), _speed(vb_pre, b.size)
        # la source est celle qui arrive le plus vite
        if (sa_pre or 0.0) >= (sb_pre or 0.0):
            src, tgt, v_pre_s, v_post_s, v_pre_t, v_post_t = a, b, va_pre, va_post, vb_pre, vb_post
        else:
            src, tgt, v_pre_s, v_post_s, v_pre_t, v_post_t = b, a, vb_pre, vb_post, va_pre, va_post
        s_pre_s, s_post_s = _speed(v_pre_s, src.size), _speed(v_post_s, src.size)
        s_pre_t, s_post_t = _speed(v_pre_t, tgt.size), _speed(v_post_t, tgt.size)
        if s_pre_s is None or s_pre_s < cfg.collision_min_source_speed:
            return None  # personne n'arrive vraiment sur l'autre

        reasons: list[str] = []
        # --- approche : la source s'est rapprochée (sinon ils se touchaient déjà) ---
        # L'alignement se mesure au *début* de l'approche : au point de distance minimale, la ligne des
        # centres est perpendiculaire à la vitesse pour tout passage latéral (angle ~90 degrés par construction).
        pa, pb = _position_at(src, f_c - win - 2), _position_at(tgt, f_c - win - 2)
        approach = align = None
        if pa is not None and pb is not None:
            approach = distance(pa, pb) / ((src.size + tgt.size) / 2.0) - d_min
            if approach < cfg.collision_min_approach:
                return None
            if v_pre_s is not None:
                align = _angle(v_pre_s, (pb[0] - pa[0], pb[1] - pa[1]))
                if align > cfg.collision_max_align_angle:
                    return None

        # --- réponse ---
        dv_t = (math.hypot(v_post_t[0] - v_pre_t[0], v_post_t[1] - v_pre_t[1]) / tgt.size
                if v_pre_t is not None and v_post_t is not None else None)
        dir_s = (_angle(v_pre_s, v_post_s)
                 if v_post_s is not None and (s_post_s or 0.0) >= 0.05 else None)
        drop_s = (1.0 - s_post_s / s_pre_s) if s_post_s is not None else None
        target_moved = dv_t is not None and dv_t >= cfg.collision_min_delta_v
        deflected = dir_s is not None and dir_s >= cfg.collision_min_direction_change and d_min <= cfg.collision_max_contact_distance
        stopped = drop_s is not None and drop_s >= cfg.collision_min_speed_drop and d_min <= cfg.collision_max_contact_distance
        if not (target_moved or deflected or stopped):
            return None

        if target_moved:
            resp = _clamp(0.5 + 0.5 * dv_t / (2 * cfg.collision_min_delta_v))  # type: ignore[operator]
            reasons.append("target_started_moving" if (s_pre_t or 0.0) <= STATIC_SPEED else "target_velocity_changed")
        else:
            resp = 0.6 if deflected else 0.55
            reasons.append("no_target_response")
        if deflected:
            reasons.append("source_changed_direction")
        if stopped:
            reasons.append("source_stopped_abruptly")
        prox = _clamp((cfg.contact_distance - d_min) / (cfg.contact_distance - 1.0))
        app_score = 0.6 if approach is None else _clamp(approach / 1.0)
        align_score = 0.7 if align is None else _clamp(1.0 - align / cfg.collision_max_align_angle)
        conf = 0.25 * prox + 0.25 * app_score + 0.15 * align_score + 0.35 * resp
        reasons.append(f"min_distance_{d_min:.2f}_diam")
        if approach is None:
            reasons.append("approach_unverified")
        if not target_moved:
            conf = min(conf, 0.65)
        if dv_t is None:
            conf *= 0.85
            reasons.append("target_response_unobserved")
        if v_post_s is None:
            conf *= 0.9
            reasons.append("source_not_observed_after_contact")
        if conf < cfg.collision_min_confidence:
            return None

        pos_s = _position_at(src, f_c) or src.current_position
        pos_t = _position_at(tgt, f_c) or tgt.current_position
        tgt_static = (s_pre_t is not None and s_pre_t <= STATIC_SPEED)
        return CollisionCandidate(
            collision_id=0,
            frame=f_c,
            detected_frame=frame,
            source_ball=src.logical_track_id,
            target_ball=tgt.logical_track_id,
            target_type=tgt.object_type.value,
            kind="impact_on_static" if tgt_static else "moving_moving",
            confidence=min(1.0, conf),
            reasons=reasons,
            min_distance_diam=d_min,
            position=((pos_s[0] + pos_t[0]) / 2.0, (pos_s[1] + pos_t[1]) / 2.0),
            source_speed_before=s_pre_s,
            source_speed_after=s_post_s,
            target_speed_before=s_pre_t,
            target_speed_after=s_post_t,
            source_direction_change=dir_s,
            target_delta_v=dv_t,
        )
