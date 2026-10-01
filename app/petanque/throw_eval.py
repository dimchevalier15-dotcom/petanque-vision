"""Évaluation des ThrowEvents contre une vérité terrain : rendre les erreurs visibles et mesurables.

Catégories d'erreur :
  MISSED_THROW                    un lancer réel n'a pas été détecté
  FALSE_THROW                     un ThrowEvent (type THROW) ne correspond à aucun lancer réel
  WRONG_BALL                      bon moment, mais la boule arrive ailleurs que prévu
  COLLISION_AS_THROW              une boule poussée a été comptée comme lancée
  OCCLUSION_FAILURE               un même lancer coupé en plusieurs (occultation / fragment), ou raté à cause d'une occultation
  OUT_OF_PLAY_MISCLASSIFICATION   lancer trouvé, mais ON_FIELD / OUT_OF_PLAY faux

Les ThrowEvents de type UNKNOWN sont des *abstentions* : jamais comptés comme faux lancers, mais pas
comme détections confiantes non plus (rapportés à part).

Vérité terrain acceptée (JSON) :
    {"throws": [{"start_frame", "end_frame", "final_position": [x, y]?, "final_state"?, "occluded"?}],
     "not_throws": [{"start_frame", "end_frame", "cause": "collision" | "pickup" | "noise" | ...}]}
ou directement un `annotations.json` (owner A/B/U = vrai lancer, « pas un lancer » = not_throws,
`missed_throws` = lancers ratés ajoutés à la main).
"""

from __future__ import annotations

import math
from typing import Any

from app.petanque.throws import ThrowEvent, ThrowType

MISSED = "MISSED_THROW"
FALSE = "FALSE_THROW"
WRONG_BALL = "WRONG_BALL"
COLLISION = "COLLISION_AS_THROW"
OCCLUSION = "OCCLUSION_FAILURE"
OUT_MISCLASS = "OUT_OF_PLAY_MISCLASSIFICATION"
CATEGORIES = (MISSED, FALSE, WRONG_BALL, COLLISION, OCCLUSION, OUT_MISCLASS)


def expected_from_annotations(doc: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Vérité terrain dérivée d'un annotations.json (seuls les lancers que l'humain a traités comptent)."""
    real, fake = [], []
    for t in doc.get("throws", []):
        if t.get("human_is_throw") is False:
            fake.append({"start_frame": t["start_frame"], "end_frame": t["end_frame"], "cause": t.get("note") or "noise"})
        elif t.get("owner", "UNASSIGNED") != "UNASSIGNED":
            real.append({"start_frame": t["start_frame"], "end_frame": t["end_frame"],
                         "final_position": t.get("final_position")})
    for m in doc.get("missed_throws", []):
        real.append({"start_frame": m["start_frame"], "end_frame": m["end_frame"],
                     "final_state": m.get("final_state")})
    return {"throws": real, "not_throws": fake}


def _overlap(a0: int, a1: int, b0: int, b1: int, tol: int) -> bool:
    return a0 <= b1 + tol and b0 <= a1 + tol


def score_throws(throws: list[ThrowEvent], expected: dict[str, Any], tol_frames: int = 30,
                 position_tol_diam: float = 3.0) -> dict[str, Any]:
    exp = expected.get("throws", [])
    not_throws = expected.get("not_throws", [])
    confident = [t for t in throws if t.event_type == ThrowType.THROW]
    abst = [t for t in throws if t.event_type == ThrowType.UNKNOWN]
    findings: list[dict[str, Any]] = []

    def add(cat: str, detail: str, t: ThrowEvent | None = None, e: dict[str, Any] | None = None) -> None:
        findings.append({
            "category": cat, "detail": detail,
            "throw_id": t.throw_id if t else None,
            "frames": [t.start_frame, t.end_frame] if t else ([e["start_frame"], e["end_frame"]] if e else None),
        })

    # appariement 1-1 (plus petit écart de début d'abord)
    pairs = sorted(
        ((abs(t.start_frame - e["start_frame"]), i, j) for i, e in enumerate(exp) for j, t in enumerate(confident)
         if _overlap(t.start_frame, t.end_frame, e["start_frame"], e["end_frame"], tol_frames)),
        key=lambda x: x[0],
    )
    match: dict[int, int] = {}  # expected -> detected
    used: set[int] = set()
    for _, i, j in pairs:
        if i not in match and j not in used:
            match[i] = j
            used.add(j)

    correct = abstained_result = 0
    for i, e in enumerate(exp):
        if i not in match:
            if any(_overlap(t.start_frame, t.end_frame, e["start_frame"], e["end_frame"], tol_frames) for t in abst):
                continue  # traité plus bas (abstention)
            add(OCCLUSION if e.get("occluded") else MISSED,
                "expected throw not detected" + (" (ball occluded)" if e.get("occluded") else ""), e=e)
            continue
        t = confident[match[i]]
        ok = True
        fp = e.get("final_position")
        if fp is not None and math.hypot(fp[0] - t.final_position[0], fp[1] - t.final_position[1]) > \
                position_tol_diam * t.metrics.get("size_px", 60.0):
            add(WRONG_BALL, "ends far from the expected position", t, e)
            ok = False
        want = e.get("final_state")
        if want in ("ON_FIELD", "OUT_OF_PLAY"):
            if t.final_state.value == "UNKNOWN":
                abstained_result += 1
            elif t.final_state.value != want:
                add(OUT_MISCLASS, f"expected {want}, got {t.final_state.value}", t, e)
                ok = False
        correct += ok

    for j, t in enumerate(confident):
        if j in used:
            continue
        covered = next((e for i, e in enumerate(exp) if i in match and _overlap(
            t.start_frame, t.end_frame, e["start_frame"], e["end_frame"], tol_frames)), None)
        if covered is not None:
            add(OCCLUSION, "same real throw detected several times (fragmentation)", t, covered)
            continue
        cause = next((n.get("cause") for n in not_throws if _overlap(
            t.start_frame, t.end_frame, n["start_frame"], n["end_frame"], tol_frames)), None)
        if cause == "collision" or t.collisions_as_target:
            add(COLLISION, "ball was pushed by another ball but counted as thrown", t)
        else:
            add(FALSE, f"no real throw here ({cause or 'unlabelled'})", t)

    abstentions = [
        {"throw_id": t.throw_id, "frames": [t.start_frame, t.end_frame],
         "matches_real_throw": any(_overlap(t.start_frame, t.end_frame, e["start_frame"], e["end_frame"], tol_frames) for e in exp),
         "matches_not_throw": any(_overlap(t.start_frame, t.end_frame, n["start_frame"], n["end_frame"], tol_frames) for n in not_throws)}
        for t in abst
    ]
    counts = {c: sum(1 for f in findings if f["category"] == c) for c in CATEGORIES}
    tp = len(match)
    n_conf = len(confident)
    precision = tp / n_conf if n_conf else None
    recall = tp / len(exp) if exp else None
    return {
        "expected_throws": len(exp),
        "detected_confident": n_conf,
        "detected_unknown_type": len(abst),
        "matched": tp,
        "fully_correct": correct,
        "precision": precision,
        "recall": recall,
        "errors": counts,
        "result_abstained": abstained_result,
        "abstentions": abstentions,
        "findings": findings,
    }


def format_scores(s: dict[str, Any]) -> str:
    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{100 * v:.0f}%"

    lines = [
        f"Lancers réels: {s['expected_throws']}   détectés (THROW): {s['detected_confident']}   "
        f"UNKNOWN: {s['detected_unknown_type']}   appariés: {s['matched']}   tout juste: {s['fully_correct']}",
        f"Précision: {pct(s['precision'])}   Rappel: {pct(s['recall'])}   résultat laissé UNKNOWN: {s['result_abstained']}",
        "Erreurs : " + "  ".join(f"{k}={v}" for k, v in s["errors"].items()),
    ]
    for f in s["findings"]:
        lines.append(f"  - {f['category']:<30} frames {f['frames']}  throw={f['throw_id']}  {f['detail']}")
    for a in s["abstentions"]:
        kind = "vrai lancer" if a["matches_real_throw"] else ("pas un lancer" if a["matches_not_throw"] else "non étiqueté")
        lines.append(f"  ~ abstention UNKNOWN  throw #{a['throw_id']} frames {a['frames']}  ({kind})")
    return "\n".join(lines)
