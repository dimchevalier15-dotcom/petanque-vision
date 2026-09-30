"""Le rendu debug ne plante pas et dessine réellement quelque chose (sans YOLO)."""

import unittest

import numpy as np

from app.petanque.config import PipelineConfig
from app.petanque.pipeline import make_pipeline
from app.petanque.visualization import DebugRenderer
from tests.synthetic import SimObject, default_field, linear, simulate


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


if __name__ == "__main__":
    unittest.main()
