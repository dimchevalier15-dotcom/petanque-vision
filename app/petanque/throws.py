"""Modèle de données des lancers : ThrowEvent, collisions, mouvements analysés, attribution.

Trois problèmes volontairement séparés :
  * suivi            -> TrackManager (tracks logiques)
  * détection lancer -> ThrowEventDetector (ThrowEvent, sans joueur)
  * attribution      -> champs `owner*` (aujourd'hui manuels ; demain règle / capteur / modèle)

Un ThrowEvent décrit *un mouvement de boule* (départ -> arrêt / sortie). `owner` reste
UNASSIGNED tant qu'aucune source ne l'a renseigné.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Protocol

from app.petanque.config import ThrowConfig
from app.petanque.models import Point


class ThrowResult(str, Enum):
    ON_FIELD = "ON_FIELD"
    OUT_OF_PLAY = "OUT_OF_PLAY"
    UNKNOWN = "UNKNOWN"


class ThrowType(str, Enum):
    THROW = "THROW"  # mouvement compatible avec une boule jouée, assez d'indices
    UNKNOWN = "UNKNOWN"  # plausible mais ambigu : à revoir par un humain, ne pas compter d'office


class Owner(str, Enum):
    PLAYER_A = "PLAYER_A"
    PLAYER_B = "PLAYER_B"
    UNKNOWN = "UNKNOWN"  # quelqu'un a regardé et ne sait pas
    UNASSIGNED = "UNASSIGNED"  # personne n'a encore décidé


class OwnerSource(str, Enum):
    MANUAL = "MANUAL"
    RULE = "RULE"
    SENSOR = "SENSOR"
    MODEL = "MODEL"


class MovementClass(str, Enum):
    """Verdict du détecteur pour *chaque* mouvement analysé (lancer ou non)."""

    THROW = "THROW"
    UNKNOWN = "UNKNOWN"  # plausible, ambigu (devient un ThrowEvent de type UNKNOWN)
    DISPLACED = "DISPLACED"  # boule déjà en place, poussée par une collision
    JACK_THROW = "JACK_THROW"  # mouvement du cochonnet avant stabilisation
    JACK_DISPLACED = "JACK_DISPLACED"
    IGNORED = "IGNORED"  # trop court, trop lent, bruit


@dataclass
class ThrowEvent:
    throw_id: int
    ball_track_id: int  # track logique (peut être fusionné plus tard : voir track_id_history)
    start_frame: int
    end_frame: int
    start_timestamp: float  # s
    end_timestamp: float
    initial_position: Point
    final_position: Point
    trajectory: list[tuple[int, float, float]]  # (frame, x, y), observations uniquement
    final_state: ThrowResult
    confidence: float
    detection_reasons: list[str]

    event_type: ThrowType = ThrowType.THROW
    final_state_confidence: float = 0.0
    final_state_reasons: list[str] = field(default_factory=list)
    kind: str = "entered_in_motion"  # entered_in_motion | moved_from_rest
    mene_id: int = 1
    metrics: dict[str, float] = field(default_factory=dict)
    bytetrack_ids: list[int] = field(default_factory=list)
    track_id_history: list[int] = field(default_factory=list)
    collisions_as_source: list[dict[str, Any]] = field(default_factory=list)
    collisions_as_target: list[dict[str, Any]] = field(default_factory=list)
    player_context: dict[str, Any] | None = None  # contexte joueur injecté (cercle, capteur...) — jamais déduit ici

    # attribution (ce module n'attribue rien : UNASSIGNED)
    owner: Owner = Owner.UNASSIGNED
    owner_source: OwnerSource | None = None
    owner_confidence: float | None = None
    # vérité terrain humaine sur l'événement lui-même
    human_is_throw: bool | None = None  # False = l'humain dit « ce n'était pas un lancer »
    note: str = ""

    @property
    def duration_s(self) -> float:
        return self.end_timestamp - self.start_timestamp

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("final_state", "event_type", "owner"):
            d[k] = getattr(self, k).value
        d["owner_source"] = self.owner_source.value if self.owner_source else None
        d["initial_position"] = [round(v, 1) for v in self.initial_position]
        d["final_position"] = [round(v, 1) for v in self.final_position]
        d["trajectory"] = [[f, round(x, 1), round(y, 1)] for f, x, y in self.trajectory]
        d["start_timestamp"] = round(self.start_timestamp, 3)
        d["end_timestamp"] = round(self.end_timestamp, 3)
        d["confidence"] = round(self.confidence, 3)
        d["final_state_confidence"] = round(self.final_state_confidence, 3)
        d["metrics"] = {k: round(v, 3) for k, v in self.metrics.items()}
        if self.owner_confidence is not None:
            d["owner_confidence"] = round(self.owner_confidence, 3)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ThrowEvent:
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in d.items() if k in known}  # tolère des champs ajoutés plus tard
        data["final_state"] = ThrowResult(data.get("final_state", "UNKNOWN"))
        data["event_type"] = ThrowType(data.get("event_type", "THROW"))
        data["owner"] = Owner(data.get("owner", "UNASSIGNED"))
        src = data.get("owner_source")
        data["owner_source"] = OwnerSource(src) if src else None
        data["initial_position"] = tuple(data["initial_position"])
        data["final_position"] = tuple(data["final_position"])
        data["trajectory"] = [tuple(p) for p in data.get("trajectory", [])]
        return cls(**data)


@dataclass
class CollisionCandidate:
    """Contact probable entre deux objets (pas de physique : juste de quoi ne pas compter un lancer en trop)."""

    collision_id: int
    frame: int  # frame du contact (distance minimale)
    detected_frame: int  # frame où la réponse a pu être évaluée
    source_ball: int  # track logique de l'objet qui arrive (le plus rapide avant contact)
    target_ball: int
    target_type: str  # BALL | JACK
    kind: str  # impact_on_static | moving_moving
    confidence: float
    reasons: list[str]
    min_distance_diam: float
    position: Point  # point de contact (milieu des deux centres)
    source_speed_before: float | None = None  # diam/frame
    source_speed_after: float | None = None
    target_speed_before: float | None = None
    target_speed_after: float | None = None
    source_direction_change: float | None = None  # degrés
    target_delta_v: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["position"] = [round(v, 1) for v in self.position]
        d["confidence"] = round(self.confidence, 3)
        for k in ("min_distance_diam", "source_speed_before", "source_speed_after", "target_speed_before",
                  "target_speed_after", "source_direction_change", "target_delta_v"):
            if d[k] is not None:
                d[k] = round(d[k], 3)
        return d


@dataclass
class MovementRecord:
    """Trace de chaque mouvement analysé, y compris ceux qui n'ont PAS donné de ThrowEvent."""

    track_id: int
    object_type: str
    start_frame: int
    end_frame: int
    classification: MovementClass
    confidence: float
    reasons: list[str]
    kind: str
    reach_diam: float = 0.0  # plus grand écart au point de départ (diamètres)
    end_kind: str = "STOPPED"  # STOPPED | OUT_OF_PLAY | LOST | TIMEOUT | END_OF_STREAM
    throw_id: int | None = None
    displaced_by: int | None = None
    continues_track: int | None = None  # boule au repos qui a disparu puis « reparti » sous un autre track

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["classification"] = self.classification.value
        d["confidence"] = round(self.confidence, 3)
        d["reach_diam"] = round(self.reach_diam, 2)
        return d


# ---------------------------------------------------------------- zone de lancer (cercle futur)
class ThrowZone(Protocol):
    def distance(self, position: Point) -> float:
        """Distance en pixels à la zone de lancer (0 si dedans)."""
        ...


@dataclass
class CircleZone:
    center: Point
    radius: float

    def distance(self, position: Point) -> float:
        return max(0.0, math.hypot(position[0] - self.center[0], position[1] - self.center[1]) - self.radius)


def zone_from_config(cfg: ThrowConfig) -> ThrowZone | None:
    """Aucune zone tant que le cercle n'est pas détecté/configuré : la règle n'est alors jamais appliquée."""
    if cfg.throw_zone_center is None or cfg.throw_zone_radius <= 0:
        return None
    return CircleZone((float(cfg.throw_zone_center[0]), float(cfg.throw_zone_center[1])), cfg.throw_zone_radius)


def format_time(seconds: float) -> str:
    """mm:ss.mmm"""
    m, s = divmod(max(0.0, seconds), 60.0)
    return f"{int(m):02d}:{s:06.3f}"
