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
class ThrowConfig:
    """Détection des lancers et des collisions (couche au-dessus du TrackManager).

    Distances en diamètres de boule, vitesses en diam/frame, durées en secondes (sauf `*_frames`).
    """

    # --- mouvement candidat : en dessous, ce n'est pas un lancer ---
    min_travel: float = 4.0  # diam : portée (plus grand écart au départ) ; en dessous = bruit / simple frémissement
    min_duration: float = 0.5  # s
    max_duration: float = 10.0  # s : un mouvement qui ne finit jamais est abandonné (UNKNOWN)
    min_peak_speed: float = 0.15  # diam/frame : en dessous = dérive / main qui tremble
    confident_travel: float = 8.0  # diam : une grande portée est un indice de lancer
    fast_speed: float = 0.25  # diam/frame : vitesse de pointe typique d'un lancer
    held_ball_max_age: float = 3.0  # s : boule « née » depuis moins longtemps = encore en main
    occlusion_gap_frames: int = 10  # trou de trajectoire à partir duquel on note une occultation
    min_points: int = 4  # observations minimales dans le mouvement

    # --- décision (score 0..1 -> THROW / UNKNOWN / ignoré) ---
    throw_min_confidence: float = 0.60  # >= : event_type THROW
    unknown_min_confidence: float = 0.35  # >= : event_type UNKNOWN (à revoir par un humain) ; < : ignoré
    high_confidence: float = 0.80  # seuil de « confiance élevée » dans les résumés
    concurrent_penalty: float = 0.15  # un autre mouvement de boule non expliqué en même temps
    ambiguous_identity_penalty: float = 0.20  # le TrackManager a refusé de ré-identifier ce track
    unexplained_old_ball_penalty: float = 0.25  # boule ancienne qui bouge sans cause visible
    jack_unstable_penalty: float = 0.40  # le cochonnet n'est pas encore stabilisé
    no_jack_penalty: float = 0.15  # aucun cochonnet vu
    unclean_end_cap: float = 0.55  # fin non observée (perdu, timeout) : plafond du score

    # --- mouvements fragmentés / fins au contact d'une autre boule ---
    stitch_gap: float = 0.5  # s : deux fragments « entrés en mouvement » aussi proches dans le temps = même lancer
    stitch_max_distance: float = 3.0  # diam : écart max entre la fin du 1er fragment et le début du 2e
    stitch_max_angle: float = 70.0  # degrés : le 2e fragment doit prolonger la direction du 1er
    contact_end_distance: float = 2.2  # diam : track perdu à cette distance d'une boule présente = arrêt au contact (0 = off)
    contact_end_max_speed: float = 0.15  # diam/frame : ... s'il ralentissait (ou s'il venait de la heurter)
    contact_end_cap: float = 0.70  # plafond du score pour une fin « perdue au contact » (> unclean_end_cap)

    # --- collisions ---
    contact_distance: float = 1.6  # diam : centres plus proches que ça = contact possible
    collision_min_source_speed: float = 0.15  # diam/frame
    collision_window_frames: int = 4  # fenêtre de vitesse avant / après le contact
    collision_settle_frames: int = 5  # attente après le contact avant d'évaluer la réponse
    collision_min_delta_v: float = 0.10  # diam/frame : réponse de la cible
    collision_min_direction_change: float = 25.0  # degrés : déviation de la source
    collision_min_speed_drop: float = 0.6  # fraction : arrêt brutal de la source
    collision_max_contact_distance: float = 1.35  # diam : déviation / arrêt sans cible => contact net requis
    collision_max_align_angle: float = 80.0  # degrés : la source doit viser la cible
    collision_min_approach: float = 0.4  # diam : la source s'est bien rapprochée
    collision_cooldown_frames: int = 30  # une même paire n'est pas réévaluée pendant ce temps
    collision_min_confidence: float = 0.50  # en dessous : pas de candidat
    collision_probable_confidence: float = 0.70
    displaced_window: float = 0.5  # s : le départ d'une boule est « expliqué » par un contact proche dans le temps
    displaced_min_confidence: float = 0.70  # cible d'une collision probable => déplacée, pas lancée
    rest_origin_distance: float = 3.0  # diam : un mouvement « entré en mouvement » qui démarre là où une boule au repos vient
    rest_origin_window: float = 0.5  # s : de disparaître (dans ce délai) est cette boule repartie, pas un nouveau lancer
    rest_origin_min_rest: float = 1.0  # s : durée minimale d'immobilité de la boule d'origine

    # --- zone de lancer (futur cercle ; optionnelle) ---
    throw_zone_center: list[float] | None = None  # [x, y] en pixels
    throw_zone_radius: float = 0.0  # pixels
    origin_max_distance: float = 8.0  # diam hors de la zone au-delà desquels le départ est suspect
    zone_bonus: float = 0.10
    zone_penalty: float = 0.25


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
    throws: ThrowConfig = field(default_factory=ThrowConfig)
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
    sections = {"tracking": TrackingConfig, "motion": MotionConfig, "game": GameConfig, "throws": ThrowConfig, "field": FieldConfig}
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
