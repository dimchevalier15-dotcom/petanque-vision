"""Métriques : ByteTrack seul vs ByteTrack + TrackManager.

Deux familles, calculées par le *même* code sur les deux systèmes (comparaison équitable) :

1. Métriques **sans vérité terrain** (vidéos réelles) : nombre de tracks, durée, fragmentation
   et doublons mesurés sur les *sites d'immobilité* (un site = un emplacement physique où une
   boule reste ; idéalement 1 seule identité par site).
2. Métriques **avec vérité terrain** (scénarios synthétiques, ou observations annotées via
   `Observation.gt_id`) : ID switches, faux splits, faux merges, pureté d'identité.

Un petit scoreur d'événements (précision / rappel avec tolérance en frames) complète le tout.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from app.petanque.models import Observation, TrackEvent

TrackSeq = dict[int, list[tuple[int, float, float, float]]]  # key -> [(frame, x, y, size)]


# --------------------------------------------------------------- construction
def raw_bytetrack_sequences(obs_by_frame: Mapping[int, list[Observation]]) -> TrackSeq:
    """Tracks « ByteTrack seul » : un track par ID ByteTrack."""
    out: TrackSeq = defaultdict(list)
    for frame in sorted(obs_by_frame):
        for o in obs_by_frame[frame]:
            if o.bt_id is not None:
                out[o.bt_id].append((frame, o.center[0], o.center[1], o.size))
    return dict(out)


def logical_sequences(tracks: Iterable[Any]) -> TrackSeq:
    """Tracks « après TrackManager » : un track par track logique confirmé."""
    return {
        t.logical_track_id: [(p.frame, p.x, p.y, p.size) for p in t.trajectory]
        for t in tracks
        if t.confirmed and t.merged_into is None
    }


# ------------------------------------------------------------- sans vérité terrain
def _stationary_segments(
    pts: list[tuple[int, float, float, float]], radius_diam: float, min_frames: int
) -> list[tuple[int, int, float, float, float]]:
    """Segments (début, fin, x, y, size) où l'objet reste dans un rayon : détecteur indépendant."""
    segs: list[tuple[int, int, float, float, float]] = []
    i, n = 0, len(pts)
    while i < n:
        j = i
        sx = sy = 0.0
        while j < n:
            sx2, sy2 = sx + pts[j][1], sy + pts[j][2]
            cx, cy = sx2 / (j - i + 1), sy2 / (j - i + 1)
            size = pts[i][3]
            if any(((p[1] - cx) ** 2 + (p[2] - cy) ** 2) ** 0.5 > radius_diam * size for p in pts[i : j + 1]):
                break
            sx, sy = sx2, sy2
            j += 1
        if j - i >= min_frames:
            seg = pts[i:j]
            segs.append((
                seg[0][0], seg[-1][0],
                statistics.fmean(p[1] for p in seg), statistics.fmean(p[2] for p in seg),
                statistics.fmean(p[3] for p in seg),
            ))
            i = j
        else:
            i += 1
    return segs


def proxy_metrics(
    seqs: TrackSeq,
    n_frames: int,
    min_len: int = 5,
    short_len: int = 30,
    stationary_radius_diam: float = 0.3,
    stationary_min_frames: int = 15,
    site_radius_diam: float = 0.6,
    site_max_gap_frames: int = 300,
    max_jump_diam_per_frame: float = 8.0,
) -> dict[str, Any]:
    tracks = {k: v for k, v in seqs.items() if len(v) >= min_len}
    lengths = [len(v) for v in tracks.values()]
    per_frame: Counter[int] = Counter()
    for v in tracks.values():
        for p in v:
            per_frame[p[0]] += 1

    segments: list[tuple[int, int, int, float, float, float]] = []  # key, start, end, x, y, size
    for k, v in tracks.items():
        for s in _stationary_segments(v, stationary_radius_diam, stationary_min_frames):
            segments.append((k, *s))

    parent = list(range(len(segments)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(len(segments)):
        for b in range(a + 1, len(segments)):
            ka, sa, ea, xa, ya, za = segments[a]
            kb, sb, eb, xb, yb, zb = segments[b]
            gap = max(sa, sb) - min(ea, eb)
            near = ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5 <= site_radius_diam * (za + zb) / 2
            if near and gap <= site_max_gap_frames:
                parent[find(a)] = find(b)

    sites: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(segments)):
        sites[find(idx)].append(idx)
    fragmentation = 0
    duplicates = 0
    for members in sites.values():
        keys = {segments[m][0] for m in members}
        fragmentation += len(keys) - 1
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                sa, ea = segments[members[a]][1:3]
                sb, eb = segments[members[b]][1:3]
                if segments[members[a]][0] != segments[members[b]][0] and min(ea, eb) - max(sa, sb) >= 5:
                    duplicates += 1
    # sauts physiquement impossibles au sein d'un même track = proxy de « faux merge / swap d'ID »
    jumps = 0
    for v in tracks.values():
        for a, b in zip(v, v[1:]):
            gap = max(1, b[0] - a[0])
            if ((b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2) ** 0.5 / (min(a[3], b[3]) * gap) > max_jump_diam_per_frame:
                jumps += 1
    return {
        "tracks": len(tracks),
        "short_tracks": sum(1 for n in lengths if n < short_len),
        "mean_track_length_frames": round(statistics.fmean(lengths), 1) if lengths else 0.0,
        "median_track_length_frames": float(statistics.median(lengths)) if lengths else 0.0,
        "max_concurrent_tracks": max(per_frame.values(), default=0),
        "stationary_sites": len(sites),
        "identity_fragmentation": fragmentation,  # identités en trop sur les sites immobiles
        "duplicate_tracks_at_site": duplicates,  # 2 identités simultanées sur le même site
        "implausible_jumps": jumps,  # téléportations dans un track (faux merge / swap), proxy
        "frames": n_frames,
    }


# ------------------------------------------------------------- avec vérité terrain
def identity_metrics(assignments: Iterable[tuple[int, int, int]]) -> dict[str, Any]:
    """assignments = (frame, track_key, gt_id). Mesure la qualité d'identité face à la vérité terrain."""
    by_gt: dict[int, list[tuple[int, int]]] = defaultdict(list)
    by_key: dict[int, set[int]] = defaultdict(set)
    for frame, key, gt in assignments:
        by_gt[gt].append((frame, key))
        by_key[key].add(gt)
    id_switches = 0
    false_splits = 0
    purity: list[float] = []
    lengths: list[int] = []
    for gt, seq in by_gt.items():
        seq.sort()
        keys = [k for _, k in seq]
        id_switches += sum(1 for a, b in zip(keys, keys[1:]) if a != b)
        false_splits += len(set(keys)) - 1
        purity.append(Counter(keys).most_common(1)[0][1] / len(keys))
    false_merges = sum(len(g) - 1 for g in by_key.values())
    counts: Counter[int] = Counter(k for _, k, _ in assignments)
    lengths = list(counts.values())
    return {
        "tracks": len(by_key),
        "gt_objects": len(by_gt),
        "id_switches": id_switches,
        "false_splits": false_splits,  # un objet physique réparti sur plusieurs tracks
        "false_merges": false_merges,  # un track mélange plusieurs objets physiques
        "identity_purity": round(statistics.fmean(purity), 3) if purity else 1.0,
        "mean_track_length_frames": round(statistics.fmean(lengths), 1) if lengths else 0.0,
    }


def score_events(
    predicted: Iterable[TrackEvent], expected: Iterable[dict[str, Any]], tolerance_frames: int = 10
) -> dict[str, Any]:
    """Précision / rappel. expected = [{"event": "BALL_PLAYED", "frame": 120}, ...]."""
    preds = sorted(predicted, key=lambda e: e.frame)
    exp = sorted(expected, key=lambda e: e["frame"])
    used: set[int] = set()
    tp = 0
    for p in preds:
        best = None
        for i, e in enumerate(exp):
            if i in used or e["event"] != p.event:
                continue
            if abs(e["frame"] - p.frame) <= tolerance_frames and (
                best is None or abs(e["frame"] - p.frame) < abs(exp[best]["frame"] - p.frame)
            ):
                best = i
        if best is not None:
            used.add(best)
            tp += 1
    fp, fn = len(preds) - tp, len(exp) - tp
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(tp / (tp + fp), 3) if tp + fp else 1.0,
        "recall": round(tp / (tp + fn), 3) if tp + fn else 1.0,
    }


def count_events(events: Iterable[TrackEvent]) -> dict[str, int]:
    return dict(Counter(e.event for e in events))


def format_comparison(before: Mapping[str, Any], after: Mapping[str, Any], title: str = "") -> str:
    """Tableau « ByteTrack seul » vs « + TrackManager » (une ligne par métrique commune)."""
    keys = [k for k in before if k in after and isinstance(before[k], (int, float))]
    width = max((len(k) for k in keys), default=10)
    lines = [title] if title else []
    lines.append(f"{'metric':<{width}}  {'ByteTrack':>10}  {'+TrackManager':>14}")
    for k in keys:
        lines.append(f"{k:<{width}}  {before[k]:>10}  {after[k]:>14}")
    return "\n".join(lines)
