"""Interface mène : à chaque lancer détecté, choisir équipe A/B et voir le résumé (distance au but).

Usage :
  python -m scripts.annotate_game \\
    --dir videos/output/20260922_180525_1_0m08-1m21_petanque \\
    --video videos/input/20260922_180525_1.mp4

  python -m scripts.annotate_game --recap \\
    --dir videos/output/20260922_180525_1_0m08-1m21_petanque

Sauvegarde dans `<dir>/annotations.json` (même format que annotate_throws).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.petanque.annotations import AnnotationSession, load_document  # noqa: E402
from app.petanque.ball_numbering import NumberedBall, display_position_at, prepare_numbered_balls  # noqa: E402
from app.petanque.config import load_config  # noqa: E402
from app.petanque.field import build_field  # noqa: E402
from app.petanque.game_summary import (  # noqa: E402
    PlayedBallLine,
    build_played_ball_lines,
    format_mene_score,
    score_mene,
)
from app.petanque.models import Point  # noqa: E402
from app.petanque.pipeline import run_pipeline  # noqa: E402
from app.petanque.recording import read_observations  # noqa: E402
from app.petanque.throws import Owner, ThrowEvent, ThrowResult, ThrowType  # noqa: E402

PANEL_W = 380
PANEL_BG = (32, 32, 32)
PANEL_FG = (235, 235, 235)
PANEL_ACCENT = (80, 200, 120)
HOLDER_COLOR = (100, 220, 255)
TIREE_COLOR = (180, 180, 255)
OUT_COLOR = (255, 120, 120)


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_jack(dir_path: Path) -> tuple[float, float] | None:
    p = dir_path / "jack.json"
    if not p.exists():
        return None
    j = json.loads(p.read_text(encoding="utf-8"))
    pos = j.get("final_position") or j.get("seed")
    if not pos:
        return None
    return (float(pos[0]), float(pos[1]))


def _ball_number_map(dir_path: Path) -> dict[int, int]:
    p = dir_path / "balls.json"
    if not p.exists():
        return {}
    doc = json.loads(p.read_text(encoding="utf-8"))
    return {int(b["throw_id"]): int(b["ball_number"]) for b in doc.get("balls", [])}


def _observations_path(dir_path: Path) -> Path:
    stem = dir_path.name
    if stem.endswith("_petanque"):
        stem = stem[: -len("_petanque")]
    return dir_path.parent / f"{stem}_observations.jsonl"


def _static_final_positions(dir_path: Path) -> dict[int, Point]:
    p = dir_path / "balls.json"
    if not p.exists():
        return {}
    doc = json.loads(p.read_text(encoding="utf-8"))
    out: dict[int, Point] = {}
    for b in doc.get("balls", []):
        fp = b.get("final_position")
        if fp and len(fp) >= 2:
            out[int(b["throw_id"])] = (float(fp[0]), float(fp[1]))
    return out


def _load_numbered_by_throw(dir_path: Path) -> dict[int, NumberedBall]:
    obs = _observations_path(dir_path)
    if not obs.exists():
        return {}
    meta, frames = read_observations(obs)
    cfg = load_config(REPO_ROOT / "data/config/petanque.yaml")
    field_ = build_field(cfg.field, (int(meta["width"]), int(meta["height"])), REPO_ROOT)
    run = run_pipeline(frames, cfg, field_, float(meta["fps"]))
    balls = prepare_numbered_balls(run.td, run.tm, frames)
    return {b.throw.throw_id: b for b in balls}


def _rows_for_summary(
    session: AnnotationSession,
    ball_nums: dict[int, int],
    jack: tuple[float, float],
    numbered: dict[int, NumberedBall],
    static_final: dict[int, Point],
    as_of_frame: int,
) -> list[dict]:
    throw_at = {t.throw_id: t for t in session.throws}
    rows: list[dict] = []
    for t in sorted(session.throws, key=lambda x: (x.start_frame, x.throw_id)):
        if t.event_type != ThrowType.THROW or t.human_is_throw is False:
            continue
        if t.owner == Owner.UNASSIGNED:
            continue
        if t.start_frame > as_of_frame:
            continue
        rows.append({
            "throw_id": t.throw_id,
            "owner": t.owner,
            "final_state": t.final_state,
            "position": display_position_at(numbered, static_final, t, as_of_frame),
            "size_px": t.metrics.get("size_px", 50.0),
            "start_frame": t.start_frame,
            "undetected": False,
        })
    for m in session.missed:
        if m.get("kind") not in ("undetected_throw", "undetected_out"):
            continue
        owner = Owner(m.get("owner", "UNASSIGNED"))
        if owner == Owner.UNASSIGNED:
            continue
        before_id = m.get("before_throw_id")
        ref = throw_at.get(int(before_id)) if before_id is not None else None
        if ref is not None:
            if ref.start_frame > as_of_frame:
                continue
            order_frame = ref.start_frame
        else:
            order_frame = int(m["start_frame"])
            if order_frame > as_of_frame:
                continue
        mid = int(m.get("missed_id") or 0)
        rows.append({
            "throw_id": -(mid or len(rows) + 1),
            "owner": owner,
            "final_state": ThrowResult.UNKNOWN,
            "position": None,
            "size_px": 50.0,
            "start_frame": order_frame,
            "before_throw_id": before_id,
            "undetected": True,
        })
    rows.sort(key=lambda r: (r["start_frame"], 0 if r.get("undetected") else 1, r["throw_id"]))
    for i, r in enumerate(rows, start=1):
        r["ball_number"] = i
    return rows


def _recap_ball_number(rows: list[dict], throw_id: int) -> int | None:
    for r in rows:
        if not r.get("undetected") and r["throw_id"] == throw_id:
            return int(r["ball_number"])
    return None


def _show_terminal_recap(
    lines: list[PlayedBallLine],
    ann_path: Path,
    mene_id: int | str | None = None,
) -> None:
    """Résumé lisible dans le terminal (reste ouvert jusqu'à Entrée)."""
    w = 58
    bar = "=" * w
    print("\n" + bar)
    print("  RÉSUMÉ DE LA MÈNE")
    if mene_id is not None:
        print(f"  mène {mene_id}")
    print(bar)
    if not lines:
        print("  (aucune boule annotée dans le récap)")
    mene_score = score_mene(lines)
    if lines:
        for ln in lines:
            print(f"  {ln.format()}")
    print("-" * w)
    print(f"  Score : {format_mene_score(mene_score)}")
    scoring = [ln.ball_number for ln in lines if ln.is_scoring]
    if scoring:
        print(f"  Boules qui marquent : {', '.join(str(n) for n in scoring)}")
    print("-" * w)
    n = len(lines)
    print(f"  Boules jouées : {n}")
    if n and n < 6:
        print("  (une mène complète fait en général 6 ou 12 boules)")
    holders = [ln for ln in lines if ln.is_holder]
    if holders:
        print(f"  Tenante : boule {holders[0].ball_number}")
    tirees = [ln for ln in lines if ln.is_tiree]
    if tirees:
        nums = ", ".join(str(ln.ball_number) for ln in tirees)
        print(f"  Tirée(s) : boule(s) {nums}")
    print("-" * w)
    print(f"  Sauvegardé : {ann_path}")
    print(bar)
    try:
        input("\n  Entrée pour quitter… ")
    except EOFError:
        pass


def _build_recap_lines(dir_path: Path) -> tuple[list[PlayedBallLine], Path, int | str | None]:
    throws_path = dir_path / "throws.json"
    if not throws_path.exists():
        raise SystemExit(f"throws.json introuvable : {throws_path}")
    doc = load_document(throws_path)
    ann_path = dir_path / "annotations.json"
    session = AnnotationSession(doc, ann_path)
    jack = _load_jack(dir_path)
    if jack is None:
        raise SystemExit("jack.json manquant ou sans position — lance render_balls avant.")
    ball_nums = _ball_number_map(dir_path)
    static_final = _static_final_positions(dir_path)
    numbered = _load_numbered_by_throw(dir_path)
    last_frame = max((t.end_frame for t in session.throws if t.event_type == ThrowType.THROW), default=0)
    rows = _rows_for_summary(session, ball_nums, jack, numbered, static_final, last_frame)
    return build_played_ball_lines(rows, jack), ann_path, session.doc.get("mene_id")


def run_recap(dir_path: Path) -> None:
    dir_path = _abs(dir_path)
    lines, ann_path, mene_id = _build_recap_lines(dir_path)
    if not ann_path.exists():
        print(f"Note : pas de {ann_path.name} — récap vide ou incomplet.")
    _show_terminal_recap(lines, ann_path, mene_id)


def _draw_panel(lines: list[PlayedBallLine], height: int, title: str) -> np.ndarray:
    panel = np.full((height, PANEL_W, 3), PANEL_BG, dtype=np.uint8)
    y = 28
    cv2.putText(panel, title, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, PANEL_ACCENT, 2, cv2.LINE_AA)
    y += 28
    cv2.putText(panel, "Boules jouees", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    y += 22
    for ln in lines:
        if ln.undetected:
            color = (160, 160, 160)
        elif ln.final_state == ThrowResult.OUT_OF_PLAY:
            color = OUT_COLOR
        elif ln.is_tiree:
            color = TIREE_COLOR
        elif ln.is_holder:
            color = HOLDER_COLOR
        else:
            color = PANEL_FG
        text = ln.format()
        if len(text) > 42:
            text = text[:41] + "…"
        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        y += 22
        if y > height - 40:
            cv2.putText(panel, "…", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, PANEL_FG, 1, cv2.LINE_AA)
            break
    cv2.putText(panel, "[A] A  [B] B  [O] OUT", (10, height - 86), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(panel, "[M]+A/B = boule non detectee (trou recap)", (10, height - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(panel, "[N] suivant  [P] prec.  [Q] quitter", (10, height - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    return panel


def _compose(frame: np.ndarray, panel: np.ndarray) -> np.ndarray:
    h = max(frame.shape[0], panel.shape[0])
    if frame.shape[0] < h:
        frame = cv2.copyMakeBorder(frame, 0, h - frame.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if panel.shape[0] < h:
        panel = cv2.copyMakeBorder(panel, 0, h - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=PANEL_BG)
    return np.hstack([frame, panel])


def _ask_team_tk(ball_label: str) -> tuple[Owner | None, bool]:
    """Retourne (équipe, marquer OUT)."""
    try:
        import tkinter as tk
    except ImportError:
        return None, False

    choice: Owner | None = None
    out_flag = False

    def set_a() -> None:
        nonlocal choice
        choice = Owner.PLAYER_A
        root.destroy()

    def set_b() -> None:
        nonlocal choice
        choice = Owner.PLAYER_B
        root.destroy()

    def set_out() -> None:
        nonlocal choice, out_flag
        out_flag = True
        root.destroy()

    root = tk.Tk()
    root.title("Équipe")
    root.attributes("-topmost", True)
    tk.Label(root, text=f"{ball_label}\nQuelle équipe a joué cette boule ?", font=("", 13), pady=12).pack()
    fr = tk.Frame(root, pady=8)
    fr.pack()
    tk.Button(fr, text="Équipe A", width=10, command=set_a, font=("", 12)).pack(side=tk.LEFT, padx=6)
    tk.Button(fr, text="Équipe B", width=10, command=set_b, font=("", 12)).pack(side=tk.LEFT, padx=6)
    tk.Button(fr, text="OUT", width=8, command=set_out, font=("", 12)).pack(side=tk.LEFT, padx=6)
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return choice, out_flag


def _ask_team_cli(ball_label: str) -> tuple[Owner | None, bool]:
    print(f"\n>>> {ball_label}")
    cmd = input("Équipe [A/B], [O]=OUT seul, Entrée = ignorer : ").strip().lower()
    if cmd == "o":
        return None, True
    if cmd == "a":
        return Owner.PLAYER_A, False
    if cmd == "b":
        return Owner.PLAYER_B, False
    return None, False


def _refresh_view(
    cap: cv2.VideoCapture,
    session: AnnotationSession,
    ball_nums: dict[int, int],
    jack: tuple[float, float],
    numbered: dict[int, NumberedBall],
    static_final: dict[int, Point],
    t: ThrowEvent,
    offset: int,
    scale: float,
    prompt: str,
) -> None:
    as_of = t.end_frame
    rows = _rows_for_summary(session, ball_nums, jack, numbered, static_final, as_of)
    lines = build_played_ball_lines(rows, jack)
    pos = display_position_at(numbered, static_final, t, as_of)
    _show_throw_frame(cap, t, offset, scale, lines, prompt, pos)
    cv2.waitKey(1)


def _show_throw_frame(
    cap: cv2.VideoCapture,
    t: ThrowEvent,
    offset: int,
    scale: float,
    lines: list[PlayedBallLine],
    prompt: str,
    ball_pos: Point,
) -> None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, offset + t.end_frame)
    ok, img = cap.read()
    if not ok or img is None:
        return
    if scale != 1.0:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    x, y = int(ball_pos[0] * scale), int(ball_pos[1] * scale)
    cv2.circle(img, (x, y), 18, (0, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(img, prompt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "M+A/B = boule non detectee avant celle-ci", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    panel = _draw_panel(lines, img.shape[0], "Petanque — mene")
    view = _compose(img, panel)
    cv2.imshow("annotate_game", view)


def main() -> None:
    p = argparse.ArgumentParser(description="Annotation équipe A/B + résumé distance au but")
    p.add_argument("--dir", type=Path, required=True, help="Dossier *_petanque (throws.json, balls.json, jack.json)")
    p.add_argument("--video", type=Path, default=None, help="Requis sauf avec --recap")
    p.add_argument("--recap", action="store_true", help="Récap terminal seulement (annotations déjà faites)")
    p.add_argument("--scale", type=float, default=0.35)
    p.add_argument("--pending", action="store_true", help="Seulement les lancers sans équipe")
    args = p.parse_args()

    dir_path = _abs(args.dir)
    if args.recap:
        run_recap(dir_path)
        return
    if args.video is None:
        raise SystemExit("--video est requis (ou utilise --recap)")

    throws_path = dir_path / "throws.json"
    if not throws_path.exists():
        raise SystemExit(f"throws.json introuvable : {throws_path}")

    doc = load_document(throws_path)
    ann_path = dir_path / "annotations.json"
    session = AnnotationSession(doc, ann_path)
    jack = _load_jack(dir_path)
    if jack is None:
        raise SystemExit("jack.json manquant ou sans position — lance render_balls avant.")

    ball_nums = _ball_number_map(dir_path)
    static_final = _static_final_positions(dir_path)
    numbered = _load_numbered_by_throw(dir_path)
    if numbered:
        print(f"Positions mises à jour depuis {_observations_path(dir_path).name}")
    elif static_final:
        print("Positions depuis balls.json (pas de observations.jsonl — rejouer render_balls si besoin)")
    video = _abs(args.video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Vidéo introuvable : {video}")
    offset = int(doc.get("frame_offset", 0) or 0)

    order = session.pending() if args.pending else [
        i for i, t in enumerate(session.throws) if t.event_type == ThrowType.THROW and t.human_is_throw is not False
    ]
    if not order:
        print("Aucun lancer à annoter.")
        return

    use_tk = sys.platform == "win32" or (
        sys.platform == "linux" and "DISPLAY" in __import__("os").environ
    )

    def ask(label: str) -> tuple[Owner | None, bool]:
        if use_tk:
            try:
                return _ask_team_tk(label)
            except Exception as exc:
                print(f"Popup indisponible ({exc}). Touches A / B / O dans la fenêtre vidéo.")
        print(f"\n>>> {label} — [A] équipe A  [B] équipe B  [O] OUT  (fenêtre vidéo)")
        return None, False
    pos = 0
    missed_armed = False
    cv2.namedWindow("annotate_game", cv2.WINDOW_NORMAL)

    while 0 <= pos < len(order):
        i = order[pos]
        t = session.throws[i]
        recap_rows = _rows_for_summary(
            session, ball_nums, jack, numbered, static_final, t.end_frame,
        )
        num = _recap_ball_number(recap_rows, t.throw_id) or ball_nums.get(t.throw_id, t.throw_id)
        state_hint = " (OUT)" if t.final_state == ThrowResult.OUT_OF_PLAY else ""
        played = len(recap_rows)
        label = f"Boule {num} — lancer #{t.throw_id}{state_hint}  ({played} jouée{'s' if played != 1 else ''})"
        if missed_armed:
            label += "  |  [M] : A/B = boule non detectee avant celle-ci"

        _refresh_view(cap, session, ball_nums, jack, numbered, static_final, t, offset, args.scale, label)

        if t.owner == Owner.UNASSIGNED:
            owner, want_out = ask(label)
            if want_out:
                session.mark_out(i, True)
                session.save()
            if owner is not None:
                session.assign(i, owner)
                session.save()
            if want_out or owner is not None:
                suffix = " OUT" if t.final_state == ThrowResult.OUT_OF_PLAY else " OK"
                _refresh_view(cap, session, ball_nums, jack, numbered, static_final, t, offset, args.scale, label + suffix)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if missed_armed and key in (ord("a"), ord("A"), ord("b"), ord("B")):
            owner = Owner.PLAYER_A if key in (ord("a"), ord("A")) else Owner.PLAYER_B
            session.add_undetected_ball(owner, t.throw_id)
            session.save()
            missed_armed = False
            print(f"Boule non détectée ({owner.value}) ajoutée au récap avant le lancer #{t.throw_id}")
            _refresh_view(cap, session, ball_nums, jack, numbered, static_final, t, offset, args.scale, label)
            continue
        if missed_armed and key not in (ord("m"), ord("M")):
            missed_armed = False
        if key in (ord("m"), ord("M")):
            missed_armed = True
            _refresh_view(cap, session, ball_nums, jack, numbered, static_final, t, offset, args.scale, label)
            continue
        if key in (ord("a"), ord("A")):
            session.assign(i, Owner.PLAYER_A)
            session.save()
            _refresh_view(cap, session, ball_nums, jack, numbered, static_final, t, offset, args.scale, label)
        elif key in (ord("b"), ord("B")):
            session.assign(i, Owner.PLAYER_B)
            session.save()
            _refresh_view(cap, session, ball_nums, jack, numbered, static_final, t, offset, args.scale, label)
        elif key in (ord("o"), ord("O")):
            session.mark_out(i, not any(r.startswith("manual_out") for r in t.final_state_reasons))
            session.save()
            _refresh_view(cap, session, ball_nums, jack, numbered, static_final, t, offset, args.scale, label)
        elif key in (ord("n"), 13):
            pos += 1
        elif key in (ord("p"),):
            pos = max(0, pos - 1)
        else:
            pos += 1

    cv2.destroyAllWindows()
    cap.release()
    session.save()
    lines, _, mene_id = _build_recap_lines(dir_path)
    _show_terminal_recap(lines, ann_path, mene_id)


if __name__ == "__main__":
    main()
