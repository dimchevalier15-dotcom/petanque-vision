"""Le rendu debug ne plante pas et dessine réellement quelque chose (sans YOLO)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.petanque.config import PipelineConfig
from app.petanque.clips import clip_window, export_clip
from app.petanque.pipeline import make_pipeline, make_throw_detector
from app.petanque.visualization import DebugRenderer
from tests.synthetic import SimObject, default_field, linear, simulate
from tests.test_throws import Y, hit_paths, jack_thrown


class TestDebugRenderer(unittest.TestCase):
    def test_draws_overlay_for_visible_occluded_and_events(self):
        field_ = default_field()
        frames = simulate([SimObject(1, linear((200, 500), (10, 0)), hidden=[(30, 36)])], 60, bt_patience=1)
        tm, gs = make_pipeline(PipelineConfig(), field_, 30.0)
        renderer = DebugRenderer(field_, 30.0, (1920, 1080), detail="full")
        blank = np.zeros((1080, 1920, 3), np.uint8)
        changed = []
        for f, obs in enumerate(frames):
            res = tm.update(f, obs)
            gev = gs.update(f, tm, res.events)
            out = renderer.draw(blank, f, tm, res, gs, gev)
            self.assertEqual(out.shape, blank.shape)
            changed.append(int((out != 0).sum()))
        self.assertTrue(all(c > 0 for c in changed))  # panneau + contour au minimum
        self.assertGreater(changed[20], changed[0])  # boîte + trajectoire + labels
        self.assertTrue(blank.sum() == 0)  # l'image d'entrée n'est pas modifiée


class TestThrowOverlays(unittest.TestCase):
    def test_throw_and_collision_labels_are_drawn(self):
        field_ = default_field()
        a_path, b_path, _ = hit_paths(150, (200, Y), (30, 0), (920, Y), 60, a_after=(0, 0), b_after=(24, 0))
        frames = simulate([jack_thrown(), SimObject(1, a_path, first_frame=150), SimObject(2, b_path)], 330)
        cfg = PipelineConfig()
        tm, gs = make_pipeline(cfg, field_, 30.0)
        td = make_throw_detector(cfg, field_, 30.0, tm.log)
        renderer = DebugRenderer(field_, 30.0, (1920, 1080))
        blank = np.zeros((1080, 1920, 3), np.uint8)
        seen_throw = seen_collision = False
        with_labels = without = 0
        for f, obs in enumerate(frames):
            res = tm.update(f, obs)
            gev = gs.update(f, tm, res.events) + td.update(f, tm, res.events, gs)
            out = renderer.draw(blank, f, tm, res, gs, gev, td)
            names = {e.event for e in gev}
            if "THROW_DETECTED" in names:
                seen_throw = True
                with_labels = int((out != 0).sum())
                without = int((DebugRenderer(field_, 30.0, (1920, 1080)).draw(blank, f, tm, res, gs, [], None) != 0).sum())
            seen_collision |= "COLLISION_CANDIDATE" in names
        self.assertTrue(seen_throw and seen_collision)
        self.assertGreater(with_labels, without)  # bloc « THROW #n » + trajectoire surimprimés


class TestClips(unittest.TestCase):
    def test_clip_window_is_clamped(self):
        from tests.test_annotations import make_throw

        t = make_throw(1, 30, 90)
        self.assertEqual(clip_window(t, 30.0, 3.0, 3.0, n_frames=150), (0, 149))
        self.assertEqual(clip_window(t, 30.0, 1.0, 2.0, n_frames=None), (0, 150))

    def test_export_clip_writes_a_readable_video_with_overlay(self):
        import cv2

        from tests.test_annotations import make_throw

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.avi"
            w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (320, 240))
            if not w.isOpened():
                self.skipTest("codec MJPG indisponible")
            for _ in range(120):
                w.write(np.zeros((240, 320, 3), np.uint8))
            w.release()
            t = make_throw(1, 40, 70)
            out = export_clip(src, t, [{"frame": 60, "position": [100.0, 100.0], "source_ball": 1, "target_ball": 2,
                                        "confidence": 0.9}], Path(tmp) / "clip.mp4", 30.0, before_s=0.5, after_s=0.5,
                              scale=1.0, n_frames=120)
            cap = cv2.VideoCapture(str(out))
            n = 0
            nonblack = 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                n += 1
                nonblack += int(img.sum() > 0)
            cap.release()
        a, b = clip_window(t, 30.0, 0.5, 0.5, 120)
        self.assertEqual(n, b - a + 1)
        self.assertGreater(nonblack, n // 2)  # bandeau + boule + collision dessinés sur un fond noir


if __name__ == "__main__":
    unittest.main()
