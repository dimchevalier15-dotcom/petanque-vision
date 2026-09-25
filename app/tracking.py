"""Annotation vidéo avec pistes ByteTrack (via Ultralytics)."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import cv2

from app.detection import BOULE_CLASS_ID, BallDetection, BallDetector, draw_detections
from app.video import VideoProcessingError, _create_writer, _progress_interval_frames, _read_metadata

logger = logging.getLogger(__name__)

_TRACK_PALETTE = (
    (255, 128, 0),
    (0, 200, 255),
    (255, 0, 200),
    (128, 255, 0),
    (200, 128, 255),
    (0, 255, 180),
    (180, 180, 0),
    (100, 100, 255),
)


def track_color(track_id: int) -> tuple[int, int, int]:
    return _TRACK_PALETTE[track_id % len(_TRACK_PALETTE)]


def draw_tracked_detections(
    frame: np.ndarray,
    detections: list[BallDetection],
    min_track_hits: int = 1,
    track_hits: dict[int, int] | None = None,
) -> np.ndarray:
    """Boxes + ID de piste (couleur par ID pour les boules)."""
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, (det.x1, det.y1, det.x2, det.y2))
        is_boule = det.class_name.strip().lower() == "boule"
        if det.track_id is not None and is_boule:
            hits = (track_hits or {}).get(det.track_id, 0)
            if hits < min_track_hits:
                continue
            color = track_color(det.track_id)
            label = f"#{det.track_id} {det.confidence:.2f}"
            thickness = 2
        else:
            annotated = draw_detections(annotated, [det])
            continue
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        cx, cy = map(int, det.center)
        cv2.circle(annotated, (cx, cy), 4, color, -1, cv2.LINE_AA)
    # Autres classes sans ID affiché via draw_detections groupé serait mieux — refaire une passe
    others = [d for d in detections if d.class_name.strip().lower() != "boule"]
    if others:
        annotated = draw_detections(annotated, others)
    return annotated


def track_video(
    input_path: Path,
    output_path: Path,
    detector: BallDetector,
    trail_length: int = 30,
    min_track_hits: int = 4,
) -> dict:
    """Écrit une vidéo avec pistes ByteTrack sur les boules."""
    if not input_path.exists():
        raise VideoProcessingError(f"Vidéo inexistante : {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise VideoProcessingError(f"Impossible d'ouvrir : {input_path}")

    metadata = _read_metadata(cap)
    writer = _create_writer(output_path, metadata)
    progress_interval = _progress_interval_frames(metadata.fps)

    trails: dict[int, list[tuple[int, int]]] = defaultdict(list)
    frames_written = 0
    max_track_id = 0
    unique_boule_ids: set[int] = set()
    active_ids: set[int] = set()
    track_hits: dict[int, int] = defaultdict(int)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame is None or frame.size == 0:
                raise VideoProcessingError(f"Frame invalide #{frames_written}")

            detections = detector.track(frame, class_ids=[BOULE_CLASS_ID])
            boule_dets = [d for d in detections if d.track_id is not None]

            active_ids = {d.track_id for d in boule_dets if d.track_id is not None}
            for tid in active_ids:
                if tid is not None:
                    track_hits[tid] += 1
            unique_boule_ids |= active_ids
            for d in boule_dets:
                if d.track_id is not None:
                    max_track_id = max(max_track_id, d.track_id)

            annotated = frame.copy()
            for det in boule_dets:
                tid = det.track_id
                if tid is None or track_hits[tid] < min_track_hits:
                    continue
                cx, cy = int(det.center[0]), int(det.center[1])
                trail = trails[tid]
                trail.append((cx, cy))
                if len(trail) > trail_length:
                    trail.pop(0)
                color = track_color(tid)
                for i in range(1, len(trail)):
                    cv2.line(annotated, trail[i - 1], trail[i], color, 2, cv2.LINE_AA)

            annotated = draw_tracked_detections(
                annotated,
                detections,
                min_track_hits=min_track_hits,
                track_hits=track_hits,
            )
            writer.write(annotated)
            frames_written += 1

            if frames_written % progress_interval == 0:
                logger.info(
                    "  %d frames | pistes actives : %d | IDs vus (max) : %d",
                    frames_written,
                    len(active_ids),
                    max_track_id,
                )
    finally:
        cap.release()
        writer.release()

    if frames_written == 0:
        raise VideoProcessingError("Aucune frame lue")

    stats = {
        "frames": frames_written,
        "max_track_id": max_track_id,
        "unique_boule_track_ids": len(unique_boule_ids),
        "confirmed_track_ids": sum(1 for h in track_hits.values() if h >= min_track_hits),
        "min_track_hits": min_track_hits,
    }
    logger.info(
        "Tracking terminé : %d frames → %s (%d ID boule distincts, max #%d)",
        frames_written,
        output_path.name,
        len(unique_boule_ids),
        max_track_id,
    )
    return stats
