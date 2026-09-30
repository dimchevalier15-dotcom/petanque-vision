"""Config, terrain, association isolée, métriques, enregistrement."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path


from app.petanque.association import score_association
from app.petanque.config import PipelineConfig, TrackingConfig, config_from_dict, load_config
from app.petanque.field import HomographyField, PolygonField, build_field
from app.petanque.config import FieldConfig
from app.petanque.metrics import identity_metrics, score_events
from app.petanque.models import BallTrack, ObjectType, Observation, TrackEvent, TrackPoint
from app.petanque.recording import read_observations, write_observations
from tests.synthetic import SimObject, linear, simulate, static

REPO = Path(__file__).resolve().parents[1]


def _obs(x, y, size=60.0, bt=1, frame=10):
    return Observation(frame, x - size / 2, y - size / 2, x + size / 2, y + size / 2, 0.9, ObjectType.BALL, bt, 0)


def _track(x, y, vx=0.0, vy=0.0, last_frame=5, hits=10, bt=1):
    t = BallTrack(1, ObjectType.BALL, 0, last_frame, size=60.0, hits=hits, confirmed=True, detection_confidence=0.9)
    t.trajectory.append(TrackPoint(last_frame, x, y, 60.0, 0.9, bt))
    t.bytetrack_ids.append(bt)
    t.velocity = (vx, vy)
    t.speed_diam = (vx * vx + vy * vy) ** 0.5 / 60.0
    return t


class TestConfig(unittest.TestCase):
    def test_yaml_matches_code_defaults(self):
        self.assertEqual(asdict(load_config(REPO / "data/config/petanque.yaml")), asdict(PipelineConfig()))

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ValueError):
            config_from_dict({"tracking": {"position_gat": 1.0}})
        with self.assertRaises(ValueError):
            config_from_dict({"trackin": {}})

    def test_override(self):
        cfg = config_from_dict({"tracking": {"max_occlusion_frames": 7}})
        self.assertEqual(cfg.tracking.max_occlusion_frames, 7)
        self.assertEqual(cfg.tracking.position_gate, TrackingConfig().position_gate)

    def test_thresholds_actually_drive_behaviour(self):
        from tests.synthetic import run

        obj = SimObject(1, linear((200, 500), (10, 0)), hidden=[(40, 49)])
        frames = simulate([obj], 100, bt_patience=2)
        strict = PipelineConfig()
        strict.tracking.max_occlusion_frames = 5  # occultation de 10 frames > limite
        strict.tracking.merge_max_gap_frames = 5
        tm, _ = run(frames, strict)
        self.assertEqual(len(tm.confirmed_tracks()), 2)
        tm, _ = run(frames)
        self.assertEqual(len(tm.confirmed_tracks()), 1)


class TestField(unittest.TestCase):
    def test_polygon_contains(self):
        f = PolygonField([(0, 0), (100, 0), (100, 100), (0, 100)])
        self.assertTrue(f.contains((50, 50)))
        self.assertFalse(f.contains((150, 50)))
        self.assertTrue(PolygonField([(0, 0), (100, 0), (100, 100), (0, 100)], margin=60).contains((150, 50)))

    def test_normalized_polygon_is_resolution_independent(self):
        cfg = FieldConfig(polygon=[[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        small, big = build_field(cfg, (100, 100)), build_field(cfg, (1000, 1000))
        self.assertTrue(small.contains((50, 50)) and big.contains((500, 500)))
        self.assertFalse(small.contains((5, 50)) or big.contains((50, 500)))

    def test_homography_field_uses_metres(self):
        import numpy as np

        h = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1.0]])  # 100 px = 1 m
        f = HomographyField(h, [(0, 0), (3, 0), (3, 10), (0, 10)])
        self.assertTrue(f.contains((150, 500)))
        self.assertFalse(f.contains((450, 500)))
        self.assertEqual(f.to_world((100, 200)), (1.0, 2.0))
        self.assertEqual(len(f.outline()), 4)


class TestAssociationIsolated(unittest.TestCase):
    cfg = TrackingConfig()

    def test_close_prediction_scores_high_with_reasons(self):
        t = _track(500, 500, vx=10, last_frame=5)
        r = score_association(t, _obs(500 + 10 * 5 * 0.85, 500, frame=10), 10, self.cfg)
        self.assertIsNotNone(r)
        self.assertGreater(r.confidence, 0.8)
        self.assertIn("predicted_position_close", r.reasons)
        self.assertIn("bytetrack_id_continuity", r.reasons)

    def test_velocity_direction_matter_not_only_distance(self):
        t = _track(500, 500, vx=12, last_frame=9, bt=1)
        ahead = score_association(t, _obs(512, 500, bt=2, frame=10), 10, self.cfg)
        behind = score_association(t, _obs(488, 500, bt=3, frame=10), 10, self.cfg)  # même distance, sens inverse
        self.assertGreater(ahead.confidence, behind.confidence if behind else 0.0)

    def test_stationary_track_has_tight_gate(self):
        t = _track(500, 500, last_frame=9)
        t.motion = __import__("app.petanque.models", fromlist=["BallState"]).BallState.STATIONARY
        self.assertIsNone(score_association(t, _obs(580, 500, bt=2), 10, self.cfg))  # 1.33 diam
        self.assertIsNotNone(score_association(t, _obs(510, 500, bt=2), 10, self.cfg))

    def test_size_mismatch_rejected(self):
        t = _track(500, 500, last_frame=9)
        self.assertIsNone(score_association(t, _obs(500, 500, size=150), 10, self.cfg))

    def test_different_type_rejected(self):
        t = _track(500, 500, last_frame=9)
        o = _obs(500, 500)
        o = Observation(**{**o.__dict__, "object_type": ObjectType.JACK})
        self.assertIsNone(score_association(t, o, 10, self.cfg))

    def test_bytetrack_id_owned_by_other_track_is_negative_evidence(self):
        t = _track(500, 500, last_frame=9, bt=1)
        free = score_association(t, _obs(505, 500, bt=7), 10, self.cfg)
        owned = score_association(t, _obs(505, 500, bt=7), 10, self.cfg, bt_owner={7: 99})
        self.assertLess(owned.confidence, free.confidence)
        self.assertTrue(any(r.startswith("NEGATIVE") for r in owned.reasons))


class TestMetrics(unittest.TestCase):
    def test_identity_metrics(self):
        # gt 1 : track 10 puis 11 (1 switch, 1 split) ; track 12 mélange gt 2 et gt 3 (1 merge)
        a = [(0, 10, 1), (1, 10, 1), (2, 11, 1), (0, 12, 2), (1, 12, 3)]
        m = identity_metrics(a)
        self.assertEqual((m["id_switches"], m["false_splits"], m["false_merges"]), (1, 1, 1))

    def test_score_events(self):
        pred = [TrackEvent(100, "BALL_PLAYED"), TrackEvent(300, "BALL_PLAYED"), TrackEvent(500, "BALL_STOPPED")]
        gt = [{"event": "BALL_PLAYED", "frame": 105}, {"event": "BALL_PLAYED", "frame": 200}]
        s = score_events(pred, gt, tolerance_frames=10)
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (1, 2, 1))


class TestRecording(unittest.TestCase):
    def test_roundtrip(self):
        frames = simulate([SimObject(1, static(300, 300)), SimObject(100, static(400, 300), object_type=ObjectType.JACK)], 5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "obs.jsonl"
            write_observations(path, {"fps": 30.0, "width": 1920, "height": 1080}, frames)
            meta, back = read_observations(path)
        self.assertEqual(meta["fps"], 30.0)
        self.assertEqual(len(back), 5)
        self.assertEqual(sorted((o.object_type.value, o.bt_id) for o in back[0]),
                         sorted((o.object_type.value, o.bt_id) for o in frames[0]))
        self.assertAlmostEqual(back[2][0].x1, frames[2][0].x1, places=1)


if __name__ == "__main__":
    unittest.main()
