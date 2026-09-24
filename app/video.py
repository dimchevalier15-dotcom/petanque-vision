"""Pipeline vidéo OpenCV pour Petanque Vision."""

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.detection import BallDetector, draw_detections

logger = logging.getLogger(__name__)

VIDEO_CODEC = "mp4v"
PROGRESS_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int | None
    duration_seconds: float | None


class VideoProcessingError(Exception):
    """Erreur lors du traitement d'une vidéo."""


MEDIA_EXTENSIONS = {".mp4", ".jpg", ".jpeg", ".png"}


def find_media_files(input_dir: Path) -> list[Path]:
    """Retourne les fichiers média supportés du répertoire d'entrée."""
    if not input_dir.exists():
        raise VideoProcessingError(f"Répertoire d'entrée inexistant : {input_dir}")
    if not input_dir.is_dir():
        raise VideoProcessingError(f"Le chemin d'entrée n'est pas un répertoire : {input_dir}")

    files = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    ]
    return sorted(files)


def find_mp4_files(input_dir: Path) -> list[Path]:
    """Retourne les fichiers MP4 du répertoire d'entrée, triés par nom."""
    return [path for path in find_media_files(input_dir) if path.suffix.lower() == ".mp4"]


def output_copy_path(input_path: Path, output_dir: Path) -> Path:
    """Construit le chemin de sortie pour une copie vidéo."""
    return output_dir / f"{input_path.stem}_copy.mp4"


def output_detected_path(input_path: Path, output_dir: Path) -> Path:
    """Construit le chemin de sortie pour un média annoté."""
    if input_path.suffix.lower() == ".mp4":
        return output_dir / f"{input_path.stem}_detected.mp4"
    return output_dir / f"{input_path.stem}_detected{input_path.suffix.lower()}"


def _progress_interval_frames(fps: float) -> int:
    """Nombre de frames entre deux logs de progression."""
    return max(1, int(fps * PROGRESS_INTERVAL_SECONDS))


def _read_metadata(cap: cv2.VideoCapture) -> VideoMetadata:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    raw_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0:
        raise VideoProcessingError("Dimensions vidéo invalides")
    if fps <= 0:
        raise VideoProcessingError(f"FPS invalide : {fps}")

    frame_count = raw_frame_count if raw_frame_count > 0 else None
    duration_seconds = frame_count / fps if frame_count is not None else None

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration_seconds,
    )


def _log_metadata(input_path: Path, metadata: VideoMetadata) -> None:
    duration = (
        f"{metadata.duration_seconds:.2f}s"
        if metadata.duration_seconds is not None
        else "inconnue"
    )
    frame_count = (
        str(metadata.frame_count)
        if metadata.frame_count is not None
        else "inconnu"
    )

    logger.info("Vidéo : %s", input_path.name)
    logger.info("  Résolution : %dx%d", metadata.width, metadata.height)
    logger.info("  FPS : %.2f", metadata.fps)
    logger.info("  Frames : %s", frame_count)
    logger.info("  Durée : %s", duration)


def _create_writer(
    output_path: Path,
    metadata: VideoMetadata,
) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        metadata.fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        raise VideoProcessingError(
            f"Impossible de créer le writer pour {output_path} (codec={VIDEO_CODEC})"
        )
    return writer


def copy_video(input_path: Path, output_path: Path) -> int:
    """Copie une vidéo frame par frame. Retourne le nombre de frames écrites."""
    if not input_path.exists():
        raise VideoProcessingError(f"Fichier vidéo inexistant : {input_path}")

    cap: cv2.VideoCapture | None = None
    writer: cv2.VideoWriter | None = None
    frames_written = 0

    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise VideoProcessingError(f"Impossible d'ouvrir la vidéo : {input_path}")

        metadata = _read_metadata(cap)
        _log_metadata(input_path, metadata)

        writer = _create_writer(output_path, metadata)
        progress_interval = _progress_interval_frames(metadata.fps)
        expected_frames = metadata.frame_count

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None or frame.size == 0:
                raise VideoProcessingError(
                    f"Frame invalide lue à l'index {frames_written} dans {input_path.name}"
                )

            writer.write(frame)
            frames_written += 1

            if frames_written % progress_interval == 0:
                logger.info(
                    "  Progression : %d frame(s) traitée(s)",
                    frames_written,
                )

        if frames_written == 0:
            raise VideoProcessingError(f"Aucune frame lue dans {input_path.name}")

        if expected_frames is not None and frames_written < expected_frames:
            logger.warning(
                "  Fin de vidéo prématurée : %d/%d frames lues",
                frames_written,
                expected_frames,
            )

        logger.info(
            "  Terminé : %d frame(s) écrite(s) → %s",
            frames_written,
            output_path.name,
        )
        return frames_written

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


def detect_video(
    input_path: Path,
    output_path: Path,
    detector: BallDetector,
) -> int:
    """Détecte les boules frame par frame et écrit une vidéo annotée."""
    if not input_path.exists():
        raise VideoProcessingError(f"Fichier vidéo inexistant : {input_path}")

    cap: cv2.VideoCapture | None = None
    writer: cv2.VideoWriter | None = None
    frames_written = 0
    total_detections = 0

    try:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise VideoProcessingError(f"Impossible d'ouvrir la vidéo : {input_path}")

        metadata = _read_metadata(cap)
        _log_metadata(input_path, metadata)

        writer = _create_writer(output_path, metadata)
        progress_interval = _progress_interval_frames(metadata.fps)
        expected_frames = metadata.frame_count

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame is None or frame.size == 0:
                raise VideoProcessingError(
                    f"Frame invalide lue à l'index {frames_written} dans {input_path.name}"
                )

            detections = detector.detect(frame)
            total_detections += len(detections)
            annotated = draw_detections(frame, detections)
            writer.write(annotated)
            frames_written += 1

            if frames_written % progress_interval == 0:
                logger.info(
                    "  Progression : %d frame(s), %d détection(s) cumulées",
                    frames_written,
                    total_detections,
                )

        if frames_written == 0:
            raise VideoProcessingError(f"Aucune frame lue dans {input_path.name}")

        if expected_frames is not None and frames_written < expected_frames:
            logger.warning(
                "  Fin de vidéo prématurée : %d/%d frames lues",
                frames_written,
                expected_frames,
            )

        logger.info(
            "  Terminé : %d frame(s), %d détection(s) → %s",
            frames_written,
            total_detections,
            output_path.name,
        )
        return frames_written

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()


def detect_image(
    input_path: Path,
    output_path: Path,
    detector: BallDetector,
) -> int:
    """Détecte les objets sur une image et écrit le résultat annoté."""
    if not input_path.exists():
        raise VideoProcessingError(f"Fichier image inexistant : {input_path}")

    frame = cv2.imread(str(input_path))
    if frame is None or frame.size == 0:
        raise VideoProcessingError(f"Impossible de lire l'image : {input_path}")

    detections = detector.detect(frame)
    annotated = draw_detections(frame, detections)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise VideoProcessingError(f"Impossible d'écrire l'image : {output_path}")

    logger.info("Image : %s", input_path.name)
    logger.info("  %d détection(s) → %s", len(detections), output_path.name)
    for detection in detections:
        center = detection.center
        logger.info(
            "    %s %.2f @ (%.0f, %.0f)",
            detection.class_name,
            detection.confidence,
            center[0],
            center[1],
        )
    return len(detections)


def process_media_with_detection(
    input_dir: Path,
    output_dir: Path,
    detector: BallDetector,
) -> None:
    """Détecte les objets sur tous les médias supportés du répertoire d'entrée."""
    if not output_dir.exists():
        raise VideoProcessingError(f"Répertoire de sortie inexistant : {output_dir}")
    if not output_dir.is_dir():
        raise VideoProcessingError(
            f"Le chemin de sortie n'est pas un répertoire : {output_dir}"
        )

    media_files = find_media_files(input_dir)
    if not media_files:
        logger.warning("Aucun média supporté trouvé dans %s", input_dir)
        return

    logger.info("%d fichier(s) à traiter", len(media_files))

    for media_path in media_files:
        output_path = output_detected_path(media_path, output_dir)
        if media_path.suffix.lower() == ".mp4":
            detect_video(media_path, output_path, detector)
        else:
            detect_image(media_path, output_path, detector)


def process_videos_with_detection(
    input_dir: Path,
    output_dir: Path,
    detector: BallDetector,
) -> None:
    """Alias conservé pour compatibilité."""
    process_media_with_detection(input_dir, output_dir, detector)


def process_videos(input_dir: Path, output_dir: Path) -> None:
    """Traite toutes les vidéos MP4 du répertoire d'entrée."""
    if not output_dir.exists():
        raise VideoProcessingError(f"Répertoire de sortie inexistant : {output_dir}")
    if not output_dir.is_dir():
        raise VideoProcessingError(
            f"Le chemin de sortie n'est pas un répertoire : {output_dir}"
        )

    videos = find_mp4_files(input_dir)
    if not videos:
        logger.warning("Aucun fichier MP4 trouvé dans %s", input_dir)
        return

    logger.info("%d vidéo(s) à traiter", len(videos))

    for video_path in videos:
        output_path = output_copy_path(video_path, output_dir)
        copy_video(video_path, output_path)
