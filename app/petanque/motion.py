"""Cinématique d'un track (vitesse, accélération, direction) et états de mouvement.

Machine à états avec hystérésis (compteurs de frames consécutives + déplacement minimal
depuis l'ancre immobile) pour éviter MOVING/STATIONARY/MOVING dus au bruit des bbox :

    UNKNOWN -> STATIONARY <-> MOVING -> SLOWING -> STATIONARY

Tout est en diamètres de boule (taille apparente du track), donc indépendant de la perspective.
Seules les frames *observées* font avancer la machine ; pendant une occultation elle est figée.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from app.petanque.config import MotionConfig
from app.petanque.models import BallState, BallTrack, MovementEpisode, Point, TrackPoint, distance


@dataclass
class MotionMemory:
    """Compteurs internes de l'hystérésis (un par track)."""

    moving_run: int = 0
    moving_run_start: int | None = None
    stationary_run: int = 0
    stationary_run_start: int | None = None
    slowing_run: int = 0
    velocity_history: list[tuple[int, Point]] = field(default_factory=list)
    direction_history: list[tuple[int, float | None]] = field(default_factory=list)


@dataclass
class MotionTransition:
    kind: str  # MOVE_STARTED | FAST | SLOWING | STOPPED
    frame: int
    data: dict[str, Any] = field(default_factory=dict)


def estimate_velocity(points: list[TrackPoint]) -> Point:
    """Vitesse (px/frame) = pente des moindres carrés sur les points (robuste au bruit)."""
    n = len(points)
    if n < 2:
        return (0.0, 0.0)
    t0 = points[0].frame
    ts = [p.frame - t0 for p in points]
    tm = sum(ts) / n
    var = sum((t - tm) ** 2 for t in ts)
    if var <= 0:
        return (0.0, 0.0)
    xm = sum(p.x for p in points) / n
    ym = sum(p.y for p in points) / n
    vx = sum((t - tm) * (p.x - xm) for t, p in zip(ts, points)) / var
    vy = sum((t - tm) * (p.y - ym) for t, p in zip(ts, points)) / var
    return (vx, vy)


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


def window_points(track: BallTrack, frame: int, window_frames: int) -> list[TrackPoint]:
    out: list[TrackPoint] = []
    for p in reversed(track.trajectory):
        if p.frame <= frame - window_frames:
            break
        out.append(p)
    out.reverse()
    return out


def mean_speed_diam(track: BallTrack) -> float:
    """Vitesse moyenne (diam/frame) pendant les frames de mouvement."""
    if track.moving_frames <= 0:
        return 0.0
    return track.distance_travelled / track.size / track.moving_frames


def update_motion(track: BallTrack, mem: MotionMemory, cfg: MotionConfig) -> list[MotionTransition]:
    """Met à jour cinématique + état de mouvement après ajout d'un point. Retourne les transitions."""
    pts = track.trajectory
    cur = pts[-1]
    frame = cur.frame
    size = max(track.size, 1e-6)
    transitions: list[MotionTransition] = []

    if len(pts) >= 2:
        step = distance(pts[-2].pos, cur.pos)
        if step >= cfg.noise_floor * size:
            track.distance_travelled += step

    win = window_points(track, frame, cfg.window_frames)
    if len(win) < 2:
        return transitions

    # --- cinématique ---
    v = estimate_velocity(win)
    speed = math.hypot(*v) / size
    track.velocity = v
    track.speed_diam = speed
    old = next(((f, ov) for f, ov in reversed(mem.velocity_history) if f <= frame - cfg.accel_lag_frames), None)
    if old is not None and frame > old[0]:
        track.acceleration = ((v[0] - old[1][0]) / (frame - old[0]), (v[1] - old[1][1]) / (frame - old[0]))
    mem.velocity_history.append((frame, v))
    del mem.velocity_history[:-60]

    direction = math.degrees(math.atan2(v[1], v[0])) if speed >= cfg.stationary_threshold else None
    old_dir = next((d for f, d in reversed(mem.direction_history) if f <= frame - cfg.direction_change_lag_frames), None)
    track.direction_change = _angle_diff(direction, old_dir) if direction is not None and old_dir is not None else 0.0
    track.direction = direction
    mem.direction_history.append((frame, direction))
    del mem.direction_history[:-60]

    # --- étiquette brute de la frame ---
    mx = median(p.x for p in win)
    my = median(p.y for p in win)
    extent = max(distance(p.pos, (mx, my)) for p in win) / size
    ph = track.motion
    moving_label = speed >= cfg.moving_threshold
    stationary_label = speed <= cfg.stationary_threshold and extent <= cfg.stationary_extent
    slowing_label = (
        ph in (BallState.MOVING, BallState.SLOWING)
        and track.peak_speed_diam >= cfg.moving_threshold
        and speed < cfg.slowing_ratio * track.peak_speed_diam
        and not moving_label
    )

    if moving_label:
        mem.moving_run += 1
        if mem.moving_run == 1:
            mem.moving_run_start = frame
    else:
        mem.moving_run = 0
    if stationary_label:
        mem.stationary_run += 1
        if mem.stationary_run == 1:
            mem.stationary_run_start = frame
    else:
        mem.stationary_run = 0
    mem.slowing_run = mem.slowing_run + 1 if slowing_label else 0

    if ph in (BallState.MOVING, BallState.SLOWING):
        track.peak_speed_diam = max(track.peak_speed_diam, speed)

    # --- transitions (hystérésis) ---
    if mem.stationary_run >= cfg.min_stationary_frames and ph != BallState.STATIONARY:
        start = mem.stationary_run_start if mem.stationary_run_start is not None else frame
        recent = [p for p in win if p.frame >= start] or win
        anchor = (median(p.x for p in recent), median(p.y for p in recent))
        _enter_state(track, BallState.STATIONARY, start)
        track.stationary_anchor = anchor
        ep = track.episode
        if ep is not None:
            track.last_episode = ep
            track.episode = None
            transitions.append(
                MotionTransition(
                    "STOPPED",
                    frame,
                    {
                        "since_frame": start,
                        "from_rest": ep.from_rest,
                        "start_frame": ep.start_frame,
                        "travel_diam": distance(ep.start_pos, anchor) / size,
                        "path_diam": (track.distance_travelled - ep.path_start) / size,
                        "max_speed_diam": ep.max_speed,
                        "fast": ep.fast,
                        "slowed": ep.slowed,
                        "duration_frames": frame - ep.start_frame,
                    },
                )
            )
    elif mem.moving_run >= cfg.min_motion_frames and ph in (BallState.UNKNOWN, BallState.STATIONARY, BallState.SLOWING):
        displaced = True
        if ph == BallState.STATIONARY and track.stationary_anchor is not None:
            displaced = distance(cur.pos, track.stationary_anchor) >= cfg.move_start_displacement * size
        if displaced:
            start = mem.moving_run_start if mem.moving_run_start is not None else frame
            if ph != BallState.SLOWING or track.episode is None:
                from_rest = ph == BallState.STATIONARY
                start_pos = track.stationary_anchor if (from_rest and track.stationary_anchor) else next(
                    (p.pos for p in win if p.frame >= start), cur.pos
                )
                track.episode = MovementEpisode(
                    start_frame=start,
                    from_rest=from_rest,
                    start_pos=start_pos,
                    path_start=track.distance_travelled,
                )
                transitions.append(
                    MotionTransition(
                        "MOVE_STARTED",
                        frame,
                        {"from_rest": from_rest, "start_frame": start, "speed_diam": speed, "previous": ph.value},
                    )
                )
            track.peak_speed_diam = speed
            _enter_state(track, BallState.MOVING, start)
    elif mem.slowing_run >= cfg.min_motion_frames and ph == BallState.MOVING:
        _enter_state(track, BallState.SLOWING, frame)
        if track.episode is not None:
            track.episode.slowed = True
        transitions.append(
            MotionTransition("SLOWING", frame, {"speed_diam": speed, "peak_speed_diam": track.peak_speed_diam})
        )

    ep = track.episode
    if ep is not None and track.motion in (BallState.MOVING, BallState.SLOWING):
        ep.max_speed = max(ep.max_speed, speed)
        if speed >= cfg.fast_threshold and not ep.fast:
            ep.fast = True
            transitions.append(MotionTransition("FAST", frame, {"speed_diam": speed}))

    if track.motion in (BallState.MOVING, BallState.SLOWING):
        track.moving_frames += 1
    elif track.motion == BallState.STATIONARY:
        track.stationary_frames += 1
    return transitions


def _enter_state(track: BallTrack, state: BallState, since_frame: int) -> None:
    track.motion = state
    track.state = state
    track.state_since_frame = since_frame
