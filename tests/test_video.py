"""Tests unitaires pour la logique vidéo (hors OpenCV)."""

import tempfile
import unittest
from pathlib import Path

from app.video import (
    VideoProcessingError,
    find_media_files,
    find_mp4_files,
    output_copy_path,
    output_detected_path,
    _progress_interval_frames,
)


class TestOutputCopyPath(unittest.TestCase):
    def test_generates_copy_filename(self) -> None:
        input_path = Path("/app/videos/input/match.mp4")
        output_dir = Path("/app/videos/output")
        self.assertEqual(
            output_copy_path(input_path, output_dir),
            Path("/app/videos/output/match_copy.mp4"),
        )


class TestFindMediaFiles(unittest.TestCase):
    def test_returns_supported_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            (input_dir / "a.mp4").write_bytes(b"")
            (input_dir / "b.jpg").write_bytes(b"")
            (input_dir / "notes.txt").write_bytes(b"")

            files = find_media_files(input_dir)

            self.assertEqual(files, [input_dir / "a.mp4", input_dir / "b.jpg"])


class TestOutputDetectedPath(unittest.TestCase):
    def test_video_and_image_outputs(self) -> None:
        output_dir = Path("/app/videos/output")
        self.assertEqual(
            output_detected_path(Path("/app/videos/input/match.mp4"), output_dir),
            output_dir / "match_detected.mp4",
        )
        self.assertEqual(
            output_detected_path(Path("/app/videos/input/frame.jpg"), output_dir),
            output_dir / "frame_detected.jpg",
        )


class TestFindMp4Files(unittest.TestCase):
    def test_returns_sorted_mp4_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = Path(tmp)
            (input_dir / "b.mp4").write_bytes(b"")
            (input_dir / "a.mp4").write_bytes(b"")
            (input_dir / "notes.txt").write_bytes(b"")

            files = find_mp4_files(input_dir)

            self.assertEqual(files, [input_dir / "a.mp4", input_dir / "b.mp4"])

    def test_raises_when_directory_missing(self) -> None:
        with self.assertRaises(VideoProcessingError):
            find_mp4_files(Path("/does/not/exist"))


class TestProgressInterval(unittest.TestCase):
    def test_at_least_one_frame(self) -> None:
        self.assertEqual(_progress_interval_frames(0.5), 1)
        self.assertEqual(_progress_interval_frames(25.0), 25)


if __name__ == "__main__":
    unittest.main()
