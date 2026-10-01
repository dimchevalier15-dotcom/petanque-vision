"""JackLocator : la place du but (cochonnet), retrouvée par sa couleur, indépendamment de la classe YOLO.

Constat sur les vraies vidéos : YOLO confond parfois le cochonnet avec une boule (ou ne le voit plus du tout
quand une boule le touche), et un « cochonnet » peut n'être qu'une feuille. Or la place du but est la référence
de la mène. On s'appuie donc sur la seule chose fiable : le cochonnet est jaune clair, les boules sont grises
et le sol est brun/gris (teinte ~10 contre ~30 pour le jack).

    1. `find_jack_seed` : blobs jaunes persistants sur des images échantillonnées ; les tracks YOLO « cochonnet »
       servent d'indice (ils départagent une tache jaune fixe d'un vrai cochonnet), jamais de preuve.
    2. `JackLocator.update` : suivi local image par image (recherche autour de la dernière position). Le but peut
       être poussé par une boule : il est suivi. Plus de blob (caché, touché par une boule) : on garde la
       dernière position, marquée « held » (approchée). Jamais de saut lointain.
    3. Déplacements du but : position de référence + événement quand elle change durablement.

Aucune distance au but, aucun score : uniquement « où est le but, et a-t-il bougé ».
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from app.petanque.models import Point, distance

YELLOW_LO = (22, 25, 150)  # teinte OpenCV 0..180 : le jack ~30 (cœur surexposé -> saturation faible), sol ~10
YELLOW_HI = (38, 255, 255)
MIN_AREA_RATIO = 0.03  # aire d'une tache / diamètre de boule² : en dessous = bruit
NEIGHBOURHOOD = 6.0  # diamètres de boule : rayon dans lequel une autre tache jaune fixe est un « distracteur »
MAX_AREA_RATIO = 0.60  # au-dessus = trop grand pour un cochonnet (ex. segments jaunes du rond de lancer)


def area_limits(ball_size: float) -> tuple[int, int]:
    return max(12, int(MIN_AREA_RATIO * ball_size**2)), int(MAX_AREA_RATIO * ball_size**2)


@dataclass
class Blob:
    x: float
    y: float
    area: float

    @property
    def pos(self) -> Point:
        return (self.x, self.y)


@dataclass
class JackFix:
    frame: int
    pos: Point
    status: str  # seen | held
    area: float = 0.0


def yellow_blobs(img: np.ndarray, origin: tuple[int, int] = (0, 0), min_area: int = 30, max_area: int = 1500,
                 round_only: bool = False) -> list[Blob]:
    """Taches jaunes de `img` (centres en coordonnées de l'image complète via `origin`).
    `round_only` : rejette les formes allongées / creuses (arc du rond de lancer, ficelle, feuille)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LO, YELLOW_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if round_only:
            w, h = float(stats[i, cv2.CC_STAT_WIDTH]), float(stats[i, cv2.CC_STAT_HEIGHT])
            if max(w, h) > 2.2 * min(w, h) or area < 0.45 * w * h:
                continue
        if min_area <= area <= max_area:
            out.append(Blob(float(cents[i][0]) + origin[0], float(cents[i][1]) + origin[1], area))
    return out


def find_jack_seed(frames: list[np.ndarray], hints: list[Point], ball_size: float,
                   inside=None) -> tuple[Point | None, list[Point], dict[str, Any]]:
    """Où est le cochonnet ? `frames` : images échantillonnées ; `hints` : positions des tracks YOLO « cochonnet ».

    Retourne (position, distracteurs, info). Distracteurs = autres taches jaunes persistantes (feuille, marque...)
    à ne jamais confondre avec le but. position = None si aucune tache jaune persistante : on ne devine pas.
    """
    tol = 0.5 * ball_size
    lo, hi = area_limits(ball_size)
    clusters: list[dict[str, Any]] = []
    for fi, img in enumerate(frames):
        for b in yellow_blobs(img, min_area=lo, max_area=hi, round_only=True):
            if inside is not None and not inside(b.pos):
                continue
            for c in clusters:
                if distance(c["pos"], b.pos) <= tol:
                    c["hits"].add(fi)
                    c["areas"].append(b.area)
                    break
            else:
                clusters.append({"pos": b.pos, "hits": {fi}, "areas": [b.area]})
    need = max(2, len(frames) // 4)
    solid = [c for c in clusters if len(c["hits"]) >= need]
    if not solid:
        return None, [], {"reason": "no_persistent_yellow_blob", "clusters": len(clusters)}

    def score(c: dict[str, Any]) -> tuple[float, float]:
        hinted = any(distance(c["pos"], h) <= tol for h in hints)
        return (len(c["hits"]) * (2.0 if hinted else 1.0), statistics.median(c["areas"]))

    solid.sort(key=score, reverse=True)
    best = solid[0]
    info = {
        "persistence": len(best["hits"]) / len(frames),
        "hinted_by_yolo": any(distance(best["pos"], h) <= tol for h in hints),
        "area_px": round(statistics.median(best["areas"]), 1),
        "candidates": len(solid),
    }
    if len(solid) > 1 and score(solid[1])[0] >= 0.8 * score(best)[0] and not info["hinted_by_yolo"]:
        info["ambiguous"] = True  # deux taches aussi plausibles, aucun indice YOLO : à vérifier à l'œil
    near = [c["pos"] for c in solid[1:] if distance(c["pos"], best["pos"]) <= NEIGHBOURHOOD * ball_size]
    return best["pos"], near, info  # seules les taches voisines du but peuvent le faire confondre


@dataclass
class JackLocator:
    """Suivi de la place du but. La position affichée est la position STABLE du but : elle ne change que si une tache
    jaune ronde, de la taille d'un cochonnet, reste immobile au même endroit `move_min_frames` images d'affilée
    pendant que l'ancien emplacement est vide (but poussé par une boule). Un vêtement jaune, une feuille, une main
    qui passent ne traînent donc jamais la marque ; pendant l'attente ou l'occultation la marque est « held »."""

    seed: Point
    ball_size: float
    distractors: list[Point] = field(default_factory=list)
    move_min_diam: float = 0.4  # au-delà : ce n'est plus le même emplacement
    move_min_frames: int = 10  # images de stabilité exigées avant d'accepter un nouvel emplacement
    search_diam: float = 4.0  # rayon de recherche autour de l'emplacement stable

    def __post_init__(self) -> None:
        self.ref: Point = self.seed
        self.fixes: list[JackFix] = []
        self.moves: list[dict[str, Any]] = []
        self._cand: tuple[Point, int, int] | None = None  # (position, nb images stables, 1re frame)
        self.radius = self.search_diam * self.ball_size

    @property
    def last(self) -> Point:
        return self.ref

    def update(self, frame: int, img: np.ndarray) -> JackFix:
        h, w = img.shape[:2]
        r = int(self.radius)
        x0, y0 = max(0, int(self.ref[0]) - r), max(0, int(self.ref[1]) - r)
        x1, y1 = min(w, int(self.ref[0]) + r), min(h, int(self.ref[1]) + r)
        lo, hi = area_limits(self.ball_size)
        blobs = yellow_blobs(img[y0:y1, x0:x1], (x0, y0), min_area=lo, max_area=hi, round_only=True)
        tol = self.move_min_diam * self.ball_size
        ok = [b for b in blobs if distance(b.pos, self.ref) <= self.radius
              and all(distance(b.pos, d) > 0.35 * self.ball_size or distance(self.ref, d) <= 0.35 * self.ball_size
                      for d in self.distractors)]
        near = [b for b in ok if distance(b.pos, self.ref) <= tol]
        if near:
            best = min(near, key=lambda b: distance(b.pos, self.ref))
            self._cand = None
            fix = JackFix(frame, best.pos, "seen", best.area)
        else:
            fix = JackFix(frame, self.ref, "held")
            far = [b for b in ok if distance(b.pos, self.ref) > tol]  # l'ancien emplacement est vide
            if far:
                b = min(far, key=lambda b: distance(b.pos, self.ref))
                if self._cand is not None and distance(self._cand[0], b.pos) <= tol:
                    self._cand = (self._cand[0], self._cand[1] + 1, self._cand[2])  # ancre fixe : pas de dérive par petits pas
                else:
                    self._cand = (b.pos, 1, frame)
                if self._cand[1] >= self.move_min_frames:
                    self.moves.append({
                        "frame": self._cand[2], "from": [round(self.ref[0], 1), round(self.ref[1], 1)],
                        "to": [round(b.pos[0], 1), round(b.pos[1], 1)],
                        "shift_diam": round(distance(self.ref, b.pos) / self.ball_size, 2),
                    })
                    self.ref, self._cand = b.pos, None
                    fix = JackFix(frame, self.ref, "seen", b.area)
            else:
                self._cand = None
        self.fixes.append(fix)
        return fix

    def summary(self) -> dict[str, Any]:
        seen = [f for f in self.fixes if f.status == "seen"]
        return {
            "seed": [round(self.seed[0], 1), round(self.seed[1], 1)],
            "final_position": [round(self.ref[0], 1), round(self.ref[1], 1)],
            "frames": len(self.fixes),
            "seen_share": round(len(seen) / max(1, len(self.fixes)), 3),
            "moves": self.moves,
            "distractors": [[round(d[0], 1), round(d[1], 1)] for d in self.distractors],
        }


def yolo_jack_hints(tm: Any) -> list[Point]:
    """Positions moyennes des tracks YOLO « cochonnet » (indices seulement, jamais une preuve)."""
    from app.petanque.models import ObjectType

    hints = []
    for t in tm.tracks.values():
        if t.object_type == ObjectType.JACK and len(t.trajectory) >= 30:
            hints.append((statistics.median(p.x for p in t.trajectory), statistics.median(p.y for p in t.trajectory)))
    return hints


def median_ball_size(frames: list[list[Any]]) -> float:
    from app.petanque.models import ObjectType

    sizes = [o.x2 - o.x1 for row in frames[:: max(1, len(frames) // 300)] for o in row if o.object_type == ObjectType.BALL]
    return float(statistics.median(sizes)) if sizes else 50.0


__all__ = ["area_limits", "Blob", "JackFix", "JackLocator", "find_jack_seed", "yellow_blobs", "yolo_jack_hints", "median_ball_size"]
