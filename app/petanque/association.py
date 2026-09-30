"""Stratégie d'association track <-> détection, isolée et testable.

`score_association` répond à : « cette détection peut-elle être la continuation de ce track ? »
en s'appuyant sur l'historique (position prédite, vitesse, direction, taille, continuité
ByteTrack) et **pas** sur la simple proximité de bbox.

Retourne None si l'association est impossible (hors porte), sinon un AssociationResult
avec une confiance dans [0, 1] et les raisons (positives) qui l'ont produite.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from app.petanque.config import TrackingConfig
from app.petanque.models import AssociationResult, BallTrack, Observation, distance
from app.petanque.motion import estimate_velocity

_GOOD = 0.6  # un composant >= GOOD compte comme « raison » positive


def position_gate(speed_diam: float, dt: int, stationary: bool, cfg: TrackingConfig) -> float:
    """Porte de position (en diamètres) : grandit avec le temps et la vitesse, plafonnée."""
    gate = cfg.position_gate + cfg.gate_growth_per_frame * dt
    if not stationary:
        gate += cfg.velocity_uncertainty * speed_diam * dt
    return min(gate, cfg.stationary_max_gate if stationary else cfg.moving_max_gate)


def _weighted(components: dict[str, float], weights: Mapping[str, float]) -> float:
    num = sum(weights.get(k, 0.0) * v for k, v in components.items())
    den = sum(weights.get(k, 0.0) for k in components)
    return num / den if den > 0 else 0.0


def _gap_factor(dt: int, limit: int, cfg: TrackingConfig) -> float:
    return 1.0 - cfg.gap_penalty * min(1.0, max(0, dt - 1) / max(limit, 1))


def score_association(
    track: BallTrack,
    obs: Observation,
    frame: int,
    cfg: TrackingConfig,
    bt_owner: Mapping[int, int] | None = None,
    kind: str = "continuation",
) -> AssociationResult | None:
    if track.object_type != obs.object_type:
        return None
    dt = max(1, frame - track.last_detection_frame)
    size = max(track.size, 1e-6)
    stationary = track.is_static(cfg.static_speed_cap)
    speed = 0.0 if stationary else track.speed_diam

    # --- ByteTrack : indice (jamais une vérité) ---
    bt_score: float | None = None
    bt_reason = ""
    if obs.bt_id is not None:
        if obs.bt_id == track.current_bt_id:
            bt_score, bt_reason = 1.0, "bytetrack_id_continuity"
        elif obs.bt_id in track.bytetrack_ids:
            bt_score, bt_reason = 0.8, "bytetrack_id_previously_linked"
        elif bt_owner is not None and bt_owner.get(obs.bt_id, track.logical_track_id) != track.logical_track_id:
            bt_score, bt_reason = 0.0, "bytetrack_id_owned_by_other_track"

    # --- position prédite ---
    px, py = track.predict(frame, cfg.velocity_decay, cfg.static_speed_cap)
    ox, oy = obs.center
    d_diam = math.hypot(ox - px, oy - py) / size
    gate = position_gate(speed, dt, stationary, cfg)
    if bt_score is not None and bt_score >= 0.8:
        gate = min(gate * cfg.bytetrack_gate_multiplier, max(cfg.moving_max_gate, gate))
    if d_diam > gate:
        return None

    # --- taille ---
    ratio = obs.size / size
    if ratio > cfg.size_ratio_gate or ratio < 1.0 / cfg.size_ratio_gate:
        return None
    size_score = max(0.0, 1.0 - abs(math.log(ratio)) / math.log(cfg.size_ratio_gate))

    comps: dict[str, float] = {
        "position": max(0.0, 1.0 - d_diam / gate),
        "size": size_score,
    }
    metrics: dict[str, float] = {
        "dt": float(dt),
        "distance_to_prediction_diam": d_diam,
        "gate_diam": gate,
        "size_ratio": ratio,
        "track_speed_diam": track.speed_diam,
    }

    # --- vitesse / direction : comparaison du déplacement réel au déplacement attendu ---
    lx, ly = track.current_position
    expected = (px - lx, py - ly)
    actual = (ox - lx, oy - ly)
    e_len, a_len = math.hypot(*expected), math.hypot(*actual)
    if not stationary and track.hits >= 3:
        err = math.hypot(actual[0] - expected[0], actual[1] - expected[1])
        comps["velocity"] = max(0.0, 1.0 - err / (cfg.velocity_gate * (e_len + 0.5 * size)))
        if e_len >= 0.5 * size and a_len >= 0.5 * size:
            cos = (expected[0] * actual[0] + expected[1] * actual[1]) / (e_len * a_len)
            angle = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            metrics["direction_error_deg"] = angle
            if angle > cfg.direction_gate_deg:
                return None
            comps["direction"] = max(0.0, 1.0 - angle / cfg.direction_gate_deg)
    if bt_score is not None:
        comps["bytetrack"] = bt_score

    limit = cfg.max_stationary_occlusion_frames if stationary else cfg.max_occlusion_frames
    confidence = _weighted(comps, cfg.weights) * _gap_factor(dt, limit, cfg)

    reasons: list[str] = []
    if comps["position"] >= _GOOD:
        reasons.append("stationary_position_match" if stationary else "predicted_position_close")
    if comps.get("velocity", 0.0) >= _GOOD:
        reasons.append("compatible_velocity")
    if comps.get("direction", 0.0) >= _GOOD:
        reasons.append("compatible_direction")
    if size_score >= _GOOD:
        reasons.append("compatible_size")
    if bt_score is not None and bt_score >= 0.8:
        reasons.append(bt_reason)
    elif bt_score == 0.0:
        reasons.append("NEGATIVE:" + bt_reason)
    if dt > 1:
        reasons.append(f"gap_{dt}_frames")
    return AssociationResult(
        track_id=track.logical_track_id,
        detection_id=obs.det_id,
        confidence=max(0.0, min(1.0, confidence)),
        reasons=reasons,
        metrics=metrics,
        kind=kind,
    )


def score_merge(old: BallTrack, young: BallTrack, cfg: TrackingConfig) -> AssociationResult | None:
    """`young` est-il la suite de `old` ? Cohérence bidirectionnelle (avant + arrière).

    Conditions dures : même type, AUCUN chevauchement temporel (deux objets vus dans la même
    frame ne peuvent pas être le même objet), trou borné, et au moins un sens cohérent.
    """
    if old.object_type != young.object_type or not old.trajectory or not young.trajectory:
        return None
    last, first = old.trajectory[-1], young.trajectory[0]
    gap = first.frame - last.frame
    if gap < 1 or gap > cfg.merge_max_gap_frames:
        return None
    if len(young.trajectory) < cfg.merge_min_points:
        return None

    stationary = old.is_static(cfg.static_speed_cap)
    old_speed = 0.0 if stationary else old.speed_diam

    # avant : où `old` devrait être quand `young` apparaît
    fx, fy = old.predict(first.frame, cfg.velocity_decay, cfg.static_speed_cap)
    d_fwd = math.hypot(first.x - fx, first.y - fy) / max(old.size, 1e-6)
    gate_fwd = position_gate(old_speed, gap, stationary, cfg)

    # arrière : où `young` était quand `old` a disparu (extrapolation linéaire de ses premiers points)
    early = young.trajectory[: max(cfg.merge_min_points, 5)]
    vy_ = estimate_velocity(early)
    young_speed = math.hypot(*vy_) / max(young.size, 1e-6)
    bx, by = first.x - vy_[0] * gap, first.y - vy_[1] * gap
    d_bwd = math.hypot(last.x - bx, last.y - by) / max(young.size, 1e-6)
    gate_bwd = position_gate(young_speed, gap, young_speed < 0.05, cfg)

    if d_fwd > 2 * gate_fwd or d_bwd > 2 * gate_bwd or (d_fwd > gate_fwd and d_bwd > gate_bwd):
        return None

    ratio = young.size / max(old.size, 1e-6)
    if ratio > cfg.size_ratio_gate or ratio < 1.0 / cfg.size_ratio_gate:
        return None
    comps = {
        "position": 0.5 * (max(0.0, 1.0 - d_fwd / gate_fwd) + max(0.0, 1.0 - d_bwd / gate_bwd)),
        "size": max(0.0, 1.0 - abs(math.log(ratio)) / math.log(cfg.size_ratio_gate)),
    }
    metrics = {
        "gap": float(gap),
        "forward_error_diam": d_fwd,
        "backward_error_diam": d_bwd,
        "size_ratio": ratio,
    }
    if not stationary and old.hits >= 3 and young_speed >= 0.05:
        va = old.velocity
        diff = math.hypot(va[0] - vy_[0], va[1] - vy_[1])
        comps["velocity"] = max(0.0, 1.0 - diff / (cfg.velocity_gate * (math.hypot(*va) + 0.5 * old.size)))
    limit = cfg.max_stationary_occlusion_frames if stationary else cfg.max_occlusion_frames
    confidence = _weighted(comps, cfg.weights) * _gap_factor(gap, limit, cfg)

    reasons: list[str] = []
    if d_fwd <= gate_fwd:
        reasons.append("predicted_position")
    if d_bwd <= gate_bwd:
        reasons.append("trajectory_consistency")
    if comps.get("velocity", 0.0) >= _GOOD:
        reasons.append("velocity_consistency")
    if comps["size"] >= _GOOD:
        reasons.append("compatible_size")
    if stationary and d_fwd <= gate_fwd:
        reasons.append("stationary_position_match")
    reasons.append("no_temporal_overlap")
    return AssociationResult(
        track_id=old.logical_track_id,
        detection_id=None,
        confidence=max(0.0, min(1.0, confidence)),
        reasons=reasons,
        metrics=metrics,
        kind="merge",
    )


__all__ = ["position_gate", "score_association", "score_merge", "distance"]
