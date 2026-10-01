"""Attribution manuelle des lancers (PLAYER_A / PLAYER_B / UNKNOWN) -> vérité terrain JSON.

Séparation volontaire : le détecteur écrit `throws.json` (jamais d'annotation humaine) ; l'humain
écrit `annotations.json` (le document complet + owner). Relancer la détection ne détruit donc
jamais le travail d'annotation : les annotations existantes sont ré-appariées aux lancers
(même boule + début proche, ou début très proche + même position d'arrivée). Une annotation qui
ne correspond plus à aucun lancer n'est pas perdue : elle est gardée dans `orphan_annotations`.

Le modèle prévoit déjà les futures sources (`owner_source` RULE / SENSOR / MODEL) ; ici : MANUAL.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.petanque.throw_export import throw_card
from app.petanque.throws import Owner, OwnerSource, ThrowEvent, ThrowType, format_time

HUMAN_CHOICES = (Owner.PLAYER_A, Owner.PLAYER_B, Owner.UNKNOWN)
MATCH_START_TOLERANCE = 15  # frames
_CARRIED = ("owner", "owner_source", "owner_confidence", "human_is_throw", "note")


def load_document(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def throws_from_document(doc: dict[str, Any]) -> list[ThrowEvent]:
    return [ThrowEvent.from_dict(d) for d in doc.get("throws", [])]


def _is_annotated(d: dict[str, Any]) -> bool:
    return d.get("owner", "UNASSIGNED") != "UNASSIGNED" or d.get("human_is_throw") is not None or bool(d.get("note"))


def _same_throw(prev: dict[str, Any], t: ThrowEvent) -> float | None:
    """Distance d'appariement (None = pas le même lancer)."""
    ds = abs(prev["start_frame"] - t.start_frame)
    if ds <= MATCH_START_TOLERANCE and prev.get("ball_track_id") == t.ball_track_id:
        return float(ds)
    size = t.metrics.get("size_px", 60.0)
    fp = prev.get("final_position") or [0.0, 0.0]
    if ds <= 5 and math.hypot(fp[0] - t.final_position[0], fp[1] - t.final_position[1]) <= 1.5 * size:
        return float(ds) + 0.5
    return None


def merge_previous(throws: list[ThrowEvent], previous: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    """Reporte les annotations d'une ancienne analyse. Retourne (nb apparié, annotations orphelines)."""
    todo = [p for p in previous if _is_annotated(p)]
    pairs = sorted(
        ((d, i, j) for i, p in enumerate(todo) for j, t in enumerate(throws) if (d := _same_throw(p, t)) is not None),
        key=lambda x: x[0],
    )
    used_p: set[int] = set()
    used_t: set[int] = set()
    for _, i, j in pairs:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        p, t = todo[i], throws[j]
        t.owner = Owner(p.get("owner", "UNASSIGNED"))
        src = p.get("owner_source")
        t.owner_source = OwnerSource(src) if src else None
        t.owner_confidence = p.get("owner_confidence")
        t.human_is_throw = p.get("human_is_throw")
        t.note = p.get("note", "")
    return len(used_p), [p for i, p in enumerate(todo) if i not in used_p]


class AnnotationSession:
    """État d'une session d'annotation (sans E/S console : testable)."""

    def __init__(self, doc: dict[str, Any], path: Path) -> None:
        self.doc = doc
        self.path = Path(path)
        self.throws = throws_from_document(doc)
        self.missed: list[dict[str, Any]] = []  # lancers que le détecteur a ratés (ajoutés à la main)
        self.orphans: list[dict[str, Any]] = []
        self.carried = 0
        if self.path.exists():
            prev = load_document(self.path)
            self.carried, self.orphans = merge_previous(self.throws, prev.get("throws", []))
            self.orphans += prev.get("orphan_annotations", [])
            self.missed = list(prev.get("missed_throws", []))

    # ------------------------------------------------------------ actions
    def assign(self, index: int, owner: Owner, confidence: float | None = None) -> None:
        if owner not in HUMAN_CHOICES:
            raise ValueError(f"Attribution manuelle : {[o.value for o in HUMAN_CHOICES]} seulement")
        t = self.throws[index]
        t.owner = owner
        t.owner_source = OwnerSource.MANUAL
        # PLAYER_A/B : l'humain est sûr ; UNKNOWN : il ne sait pas -> pas de confiance à porter
        t.owner_confidence = confidence if confidence is not None else (None if owner == Owner.UNKNOWN else 1.0)
        if t.human_is_throw is False:
            t.human_is_throw = None  # on attribue => c'était bien un lancer

    def mark_not_throw(self, index: int, flag: bool = True) -> None:
        t = self.throws[index]
        t.human_is_throw = False if flag else None
        if flag:
            t.owner, t.owner_source, t.owner_confidence = Owner.UNASSIGNED, None, None

    def clear(self, index: int) -> None:
        t = self.throws[index]
        t.owner, t.owner_source, t.owner_confidence, t.human_is_throw = Owner.UNASSIGNED, None, None, None

    def add_missed(self, start_frame: int, end_frame: int, owner: Owner = Owner.UNASSIGNED,
                   final_state: str = "UNKNOWN", note: str = "") -> None:
        self.missed.append({"start_frame": start_frame, "end_frame": end_frame, "owner": owner.value,
                            "final_state": final_state, "note": note, "source": "MANUAL"})

    def is_done(self, index: int) -> bool:
        t = self.throws[index]
        return t.owner != Owner.UNASSIGNED or t.human_is_throw is False

    def pending(self) -> list[int]:
        return [i for i in range(len(self.throws)) if not self.is_done(i)]

    def progress(self) -> tuple[int, int]:
        return len(self.throws) - len(self.pending()), len(self.throws)

    # ---------------------------------------------------------------- IO
    def to_document(self) -> dict[str, Any]:
        doc = {k: v for k, v in self.doc.items() if k != "throws"}
        doc["annotations_updated_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        doc["throws"] = [t.to_dict() for t in self.throws]
        doc["missed_throws"] = self.missed
        doc["orphan_annotations"] = self.orphans
        return doc

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_document(), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)  # jamais de fichier d'annotations à moitié écrit
        return self.path


# ---------------------------------------------------------------------------- CLI
HELP = ("[A] PLAYER_A  [B] PLAYER_B  [U] UNKNOWN  [X] pas un lancer  [V] voir la vidéo  "
        "[M] lancer manqué  [C] effacer  [N]/Entrée suivant  [P] précédent  [G] aller au #  [Q] quitter")


def run_cli(
    session: AnnotationSession,
    clip_fn: Callable[[ThrowEvent], str | None] | None = None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    only_pending: bool = False,
) -> None:
    """Boucle d'annotation. Chaque attribution est sauvegardée immédiatement."""
    order = session.pending() if only_pending else list(range(len(session.throws)))
    if not order:
        print_fn("Aucun lancer à annoter.")
        return
    pos = 0
    while 0 <= pos < len(order):
        i = order[pos]
        t = session.throws[i]
        done, total = session.progress()
        print_fn("\n" + throw_card(t))
        print_fn(f"[{pos + 1}/{len(order)}]  annotés {done}/{total}")
        if t.event_type == ThrowType.UNKNOWN:
            print_fn("⚠ Le détecteur n'est pas sûr que ce soit un lancer : regarde la vidéo avant de répondre.")
        cmd = input_fn(HELP + "\n> ").strip().lower()
        if cmd in ("a", "b", "u"):
            session.assign(i, {"a": Owner.PLAYER_A, "b": Owner.PLAYER_B, "u": Owner.UNKNOWN}[cmd])
            session.save()
            pos += 1
        elif cmd == "x":
            session.mark_not_throw(i)
            session.save()
            pos += 1
        elif cmd == "c":
            session.clear(i)
            session.save()
        elif cmd in ("", "n"):
            pos += 1
        elif cmd == "p":
            pos = max(0, pos - 1)
        elif cmd == "v":
            if clip_fn is None:
                print_fn("Pas de vidéo source : relance avec --video.")
            else:
                print_fn(f"Clip : {clip_fn(t)}")
        elif cmd == "g":
            raw = input_fn("Aller au throw # : ").strip()
            target = next((k for k, idx in enumerate(order) if raw.isdigit() and session.throws[idx].throw_id == int(raw)), None)
            if target is None:
                print_fn("Numéro inconnu.")
            else:
                pos = target
        elif cmd == "m":
            try:
                a = int(input_fn("frame de départ du lancer manqué : "))
                b = int(input_fn("frame d'arrivée : "))
            except ValueError:
                print_fn("Nombre attendu.")
                continue
            who = input_fn("joueur [A/B/U, vide = non attribué] : ").strip().lower()
            owner = {"a": Owner.PLAYER_A, "b": Owner.PLAYER_B, "u": Owner.UNKNOWN}.get(who, Owner.UNASSIGNED)
            session.add_missed(a, b, owner)
            session.save()
            print_fn(f"Lancer manqué enregistré ({format_time(a / float(session.doc.get('fps') or 30.0))}).")
        elif cmd == "q":
            break
        else:
            print_fn("Commande inconnue.")
    session.save()
    done, total = session.progress()
    print_fn(f"\nSauvegardé : {session.path}  ({done}/{total} lancers traités)")
