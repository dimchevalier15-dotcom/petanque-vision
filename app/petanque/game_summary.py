"""Résumé lisible d'une mène : équipe, distance au but, tenante / tirée.

Distances en cm : approximation via le diamètre apparent de la boule (7,05 cm).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.petanque.models import Point
from app.petanque.throws import Owner, ThrowResult

BOULE_DIAMETER_CM = 7.05


@dataclass
class MeneScore:
    points_a: int
    points_b: int
    scoring_team: Owner | None = None


@dataclass
class PlayedBallLine:
    ball_number: int
    throw_id: int
    owner: Owner
    final_state: ThrowResult
    distance_cm: float | None
    is_holder: bool = False  # plus proche du but en jeu
    is_tiree: bool = False  # tenante avant un OUT adverse, ou plus tenante après un lancer plus proche
    undetected: bool = False  # lancer ajouté à la main (pas de track)
    is_scoring: bool = False  # compte pour le score de la mène (règle FFPJP)

    def format(self) -> str:
        team = {
            Owner.PLAYER_A: "équipe A",
            Owner.PLAYER_B: "équipe B",
            Owner.UNKNOWN: "équipe ?",
            Owner.UNASSIGNED: "équipe ?",
        }[self.owner]
        if self.undetected:
            return f"boule {self.ball_number} — {team} — non détectée"
        if self.final_state == ThrowResult.OUT_OF_PLAY:
            return f"boule {self.ball_number} — {team} — OUT"
        if self.distance_cm is None:
            dist = "?"
        else:
            dist = f"{self.distance_cm:.0f} cm"
        if self.is_tiree:
            extra = " — tirée"
        elif self.is_holder:
            extra = " — tenante"
        else:
            extra = ""
        if self.is_scoring and not self.is_holder:
            extra += " — point"
        return f"boule {self.ball_number} — {team} — {dist}{extra}"


def distance_to_jack_cm(ball: Point, jack: Point, ball_size_px: float) -> float:
    if ball_size_px <= 1e-6:
        ball_size_px = 50.0
    dist_px = math.hypot(ball[0] - jack[0], ball[1] - jack[1])
    return (dist_px / ball_size_px) * BOULE_DIAMETER_CM


def _holder_in(lines: list[PlayedBallLine]) -> PlayedBallLine | None:
    in_play = [
        ln for ln in lines
        if not ln.undetected and ln.final_state != ThrowResult.OUT_OF_PLAY and ln.distance_cm is not None
    ]
    if not in_play:
        return None
    return min(in_play, key=lambda ln: ln.distance_cm)


def build_played_ball_lines(rows: list[dict], jack: Point) -> list[PlayedBallLine]:
    lines: list[PlayedBallLine] = []
    for r in rows:
        owner = r["owner"] if isinstance(r["owner"], Owner) else Owner(r["owner"])
        state = r["final_state"] if isinstance(r["final_state"], ThrowResult) else ThrowResult(r["final_state"])
        pos = r.get("position")
        size = float(r.get("size_px", 50.0))
        dist = None
        if not r.get("undetected") and state != ThrowResult.OUT_OF_PLAY and pos is not None:
            dist = distance_to_jack_cm(pos, jack, size)
        lines.append(
            PlayedBallLine(
                ball_number=int(r["ball_number"]),
                throw_id=int(r["throw_id"]),
                owner=owner,
                final_state=state,
                distance_cm=dist,
                undetected=bool(r.get("undetected")),
            )
        )
    _mark_holder_and_tiree(lines)
    return lines


def _mark_holder_and_tiree(lines: list[PlayedBallLine]) -> None:
    for ln in lines:
        ln.is_holder = False
        ln.is_tiree = False
    for i in range(1, len(lines)):
        curr = lines[i]
        if curr.undetected:
            continue
        h_before = _holder_in(lines[:i])
        if h_before is None:
            continue
        if (
            curr.final_state == ThrowResult.OUT_OF_PLAY
            and curr.owner in (Owner.PLAYER_A, Owner.PLAYER_B)
            and h_before.owner in (Owner.PLAYER_A, Owner.PLAYER_B)
            and curr.owner != h_before.owner
        ):
            h_before.is_tiree = True
        elif curr.final_state != ThrowResult.OUT_OF_PLAY:
            h_after = _holder_in(lines[: i + 1])
            if h_after is not None and h_after.throw_id != h_before.throw_id:
                h_before.is_tiree = True

    in_play = [
        ln for ln in lines
        if not ln.undetected and ln.final_state != ThrowResult.OUT_OF_PLAY and ln.distance_cm is not None
    ]
    candidates = [ln for ln in in_play if not ln.is_tiree]
    if candidates:
        min(candidates, key=lambda ln: ln.distance_cm).is_holder = True


def _scorable_lines(lines: list[PlayedBallLine]) -> list[PlayedBallLine]:
    return [
        ln for ln in lines
        if not ln.undetected
        and ln.final_state != ThrowResult.OUT_OF_PLAY
        and ln.distance_cm is not None
        and ln.owner in (Owner.PLAYER_A, Owner.PLAYER_B)
    ]


def score_mene(lines: list[PlayedBallLine]) -> MeneScore:
    """Points FFPJP : tenante + chaque boule suivante de la même équipe plus proche que la 1re adverse."""
    for ln in lines:
        ln.is_scoring = False
    ranked = sorted(_scorable_lines(lines), key=lambda ln: ln.distance_cm)
    if not ranked:
        return MeneScore(0, 0, None)
    team = ranked[0].owner
    points = 0
    for ln in ranked:
        if ln.owner != team:
            break
        points += 1
        ln.is_scoring = True
    if team == Owner.PLAYER_A:
        return MeneScore(points, 0, team)
    return MeneScore(0, points, team)


def format_mene_score(score: MeneScore) -> str:
    return f"équipe A {score.points_a} — équipe B {score.points_b}"
