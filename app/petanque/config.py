"""Configuration des seuils (tout est modifiable via YAML, rien de caché dans le code).

Convention : les distances sont exprimées en **diamètres de boule** (taille de la bbox
du track) et les vitesses en **diamètres / frame**. Ça rend les seuils indépendants de la
résolution et de la perspective (une boule lointaine est plus petite en pixels).
Les durées de la section `game` sont en secondes (converties avec le FPS de la vidéo).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class TrackingConfig:
    # --- persistance / occultation (frames) ---
    max_occlusion_frames: int = 30  # track en mouvement : fenêtre de réidentification
    max_stationary_occlusion_frames: int = 900  # track immobile : position connue, on attend plus
    short_gap_frames: int = 2  # <= : continuité ; > : réidentification (seuil plus strict)
    min_hits_confirm: int = 3  # détections avant de considérer un track comme réel
    tentative_max_missing: int = 2  # un track non confirmé disparaît vite (flash YOLO)

    # --- portes (en diamètres) ---
    position_gate: float = 0.75  # écart max à la position prédite, base
    velocity_uncertainty: float = 0.35  # élargissement de la porte : * vitesse * dt
    gate_growth_per_frame: float = 0.03  # élargissement de la porte : * dt
    stationary_max_gate: float = 1.5  # plafond pour un track immobile (il ne peut pas « voler »)
    moving_max_gate: float = 14.0  # plafond pour un track rapide
    velocity_gate: float = 1.0  # tolérance relative de vitesse (score seulement)
    velocity_decay: float = 0.95  # frottement : v(t+1) = decay * v(t) pour la prédiction
    static_speed_cap: float = 0.08  # diam/frame : STATIONARY mais plus vite que ça = départ en cours
    direction_gate_deg: float = 120.0  # rejet dur au-delà (si déplacements significatifs)
    size_ratio_gate: float = 1.8  # rejet dur si la taille de bbox varie plus que ça
    bytetrack_gate_multiplier: float = 2.0  # porte élargie si ByteTrack confirme l'ID

    # --- scores / décisions ---
    match_threshold: float = 0.35  # continuité (dt <= short_gap_frames)
    reidentification_threshold: float = 0.55  # réapparition après occultation
    ambiguity_margin: float = 0.12  # 2 candidats à moins que ça => ambigu => UNKNOWN
    ambiguity_confidence_factor: float = 0.7  # pénalité si continuité ambiguë acceptée
    gap_penalty: float = 0.25  # pénalité de confiance max pour un long trou
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "position": 0.45,
            "velocity": 0.15,
            "direction": 0.10,
            "size": 0.10,
            "bytetrack": 0.20,
        }
    )

    # --- fusion / doublons ---
    duplicate_merge_threshold: float = 0.70  # fusion rétroactive de fragments
    merge_window_frames: int = 20  # un jeune track peut être fusionné pendant ce temps
    merge_min_points: int = 3
    merge_max_gap_frames: int = 60
    duplicate_iou_threshold: float = 0.5  # 2 tracks simultanés quasi superposés
    duplicate_min_frames: int = 10  # ... pendant au moins ça
    duplicate_release_iou: float = 0.3  # un ID ByteTrack doublon n'est ignoré que tant qu'il recouvre l'original

    # --- confiance d'identité ---
    identity_recovery_per_frame: float = 0.01
    detection_confidence_ema: float = 0.2
    size_ema: float = 0.2

    # --- sortie de zone ---
    exit_min_speed: float = 0.25  # diamètres/frame : un tir/roulé rapide
    exit_horizon_frames: int = 20  # extrapolation de trajectoire pour voir si on sort
    exit_decision_delay: int = 3  # frames manquantes avant de juger « sortie »
    out_of_play_confirm_frames: int = 90  # candidat non revu => OUT_OF_PLAY (révocable)
    out_of_play_max_confidence: float = 0.75  # jamais de certitude sur une disparition


@dataclass
class MotionConfig:
    window_frames: int = 8  # fenêtre (en frames) d'estimation de vitesse
    stationary_threshold: float = 0.04  # diam/frame
    stationary_extent: float = 0.25  # dispersion max (diam) sur la fenêtre
    moving_threshold: float = 0.12  # diam/frame
    fast_threshold: float = 0.8  # diam/frame : tir
    min_motion_frames: int = 3  # frames consécutives pour entrer en MOVING
    min_stationary_frames: int = 8  # frames consécutives pour entrer en STATIONARY
    move_start_displacement: float = 0.3  # diam depuis l'ancre immobile pour quitter STATIONARY
    slowing_ratio: float = 0.5  # vitesse < ratio * pic => SLOWING
    noise_floor: float = 0.03  # diam : pas minimal compté dans la distance parcourue
    accel_lag_frames: int = 4
    direction_change_lag_frames: int = 6


@dataclass
class GameConfig:
    jack_stabilization_time: float = 1.0  # s immobile => JACK_STABILIZED
    ball_stabilization_time: float = 1.0  # s sans aucun mouvement => BALL_STABILIZED
    ball_play_detection_window: float = 1.5  # s : recherche d'un impact avant un départ
    impact_radius: float = 3.0  # diam : une boule qui bouge près d'une autre en mouvement
    min_play_travel: float = 1.5  # diam : déplacement min pour compter un lancer
    max_play_duration: float = 8.0  # s : un « lancer » plus long est abandonné
    expected_ball_count: int = 12  # 0 = désactive END_OF_MENE automatique
    jack_loss_grace: float = 3.0  # s : jack perdu avant de choisir un autre candidat


@dataclass
class FieldConfig:
    # Une seule source de géométrie (priorité : calibration > polygon > plein cadre).
    polygon: list[list[float]] | None = None
    polygon_space: str = "normalized"  # "normalized" (0..1 du cadre) ou "pixel"
    calibration: str | None = None  # JSON d'homographie (scripts.calibrate_court)
    world_polygon: list[list[float]] | None = None  # zone de jeu en mètres (repère calibration)
    margin: float = 0.0  # unité du champ : px / fraction du cadre / mètres (calibration)


@dataclass
class PipelineConfig:
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    game: GameConfig = field(default_factory=GameConfig)
    field: FieldConfig = field(default_factory=FieldConfig)


def _build(cls: type, data: dict[str, Any] | None, path: str = "") -> Any:
    """dataclass depuis un dict ; erreur explicite sur clé inconnue (typo = bug silencieux)."""
    data = data or {}
    unknown = set(data) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Clés de config inconnues dans '{path or cls.__name__}': {sorted(unknown)}")
    return cls(**data)


def config_from_dict(data: dict[str, Any] | None) -> PipelineConfig:
    data = data or {}
    sections = {"tracking": TrackingConfig, "motion": MotionConfig, "game": GameConfig, "field": FieldConfig}
    unknown = set(data) - set(sections)
    if unknown:
        raise ValueError(f"Sections de config inconnues : {sorted(unknown)}")
    return PipelineConfig(**{name: _build(cls, data.get(name), name) for name, cls in sections.items()})


def load_config(path: Path | None) -> PipelineConfig:
    """Charge un YAML (ou retourne les défauts si path est None)."""
    if path is None:
        return PipelineConfig()
    import yaml  # dépendance déjà tirée par ultralytics

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return config_from_dict(data)
