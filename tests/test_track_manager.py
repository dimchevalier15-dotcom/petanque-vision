"""Scénarios TrackManager (cas 1-6, 8-10 du cahier des charges)."""

import unittest

from app.petanque.config import PipelineConfig
from app.petanque.field import PolygonField
from app.petanque.metrics import identity_metrics
from app.petanque.models import BallState, EventType
from tests.synthetic import (
    SimObject,
    gt_assignments,
    linear,
    piecewise,
    raw_assignments,
    rolling,
    run,
    simulate,
    static,
)


def events_of(tm, *names):
    return [e for e in tm.log.events if e.event in names]


class TestNormalTracks(unittest.TestCase):
    def test_case1_continuous_detection_single_logical_track(self):
        frames = simulate([SimObject(1, linear((200, 500), (10, 0)))], 100)
        tm, _ = run(frames)
        tracks = tm.confirmed_tracks()
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].hits, 100)
        self.assertEqual(tracks[0].motion, BallState.MOVING)
        self.assertGreater(tracks[0].distance_travelled, 900)
        self.assertEqual(len(events_of(tm, EventType.TRACK_CREATED)), 1)

    def test_case6_stationary_ball_is_stationary_without_flapping(self):
        frames = simulate([SimObject(1, static(600, 500))], 150, noise=1.5)
        tm, _ = run(frames)
        (t,) = tm.confirmed_tracks()
        self.assertEqual(t.state, BallState.STATIONARY)
        self.assertEqual(events_of(tm, EventType.BALL_MOVE_STARTED), [])  # bruit != mouvement
        self.assertEqual(events_of(tm, EventType.BALL_STOPPED), [])  # jamais bougé
        self.assertLess(t.speed_diam, 0.04)
        self.assertGreater(t.stationary_frames, 100)


class TestOcclusion(unittest.TestCase):
    def test_case2_short_occlusion_keeps_logical_track(self):
        for gap in (2, 3, 5):
            with self.subTest(gap=gap):
                obj = SimObject(1, linear((200, 500), (10, 0)), hidden=[(40, 40 + gap - 1)])
                frames = simulate([obj], 100, bt_patience=1)  # ByteTrack perd l'ID
                tm, results = run(frames)
                tracks = tm.confirmed_tracks()
                self.assertEqual(len(tracks), 1, [e.to_dict() for e in tm.log.events])
                self.assertGreaterEqual(len(tracks[0].bytetrack_chain()), 1)
                self.assertEqual(identity_metrics(gt_assignments(results))["false_splits"], 0)

    def test_case9_new_bytetrack_id_after_occlusion_same_logical_track(self):
        obj = SimObject(1, linear((200, 500), (10, 0)), hidden=[(40, 47)])
        frames = simulate([obj], 100, bt_patience=2)
        raw = {o.bt_id for f in frames for o in f}
        self.assertEqual(len(raw), 2)  # ByteTrack a bien changé d'ID
        tm, _ = run(frames)
        (t,) = tm.confirmed_tracks()
        self.assertEqual(t.bytetrack_chain(), sorted(raw))
        reid = events_of(tm, EventType.TRACK_REIDENTIFIED, EventType.TRACK_MERGED)
        self.assertTrue(reid)
        ev = reid[0].to_dict()
        self.assertEqual((ev["old_bytetrack_id"], ev["new_bytetrack_id"]), (1, 2))
        self.assertGreater(ev["confidence"], 0.5)
        self.assertTrue(ev["reasons"])

    def test_stationary_ball_survives_long_occlusion(self):
        obj = SimObject(1, static(600, 500), hidden=[(50, 249)])
        frames = simulate([obj], 300)
        tm, results = run(frames)
        self.assertEqual(len(tm.confirmed_tracks()), 1)
        self.assertEqual(identity_metrics(gt_assignments(results))["false_splits"], 0)

    def test_case3_long_occlusion_of_moving_ball_does_not_invent_identity(self):
        # La boule disparaît 120 frames en mouvement puis réapparaît ailleurs : pas de fusion forcée.
        obj = SimObject(1, piecewise((0, linear((200, 500), (10, 0))), (150, static(1500, 800))), hidden=[(30, 149)])
        frames = simulate([obj], 250)
        tm, _ = run(frames)
        tracks = tm.confirmed_tracks()
        self.assertEqual(len(tracks), 2)
        self.assertTrue(events_of(tm, EventType.TRACK_LOST))
        self.assertEqual(tm.tracks[tracks[0].logical_track_id].state, BallState.LOST)

    def test_case3_ambiguous_reappearance_is_not_reidentified(self):
        # A (immobile) disparaît ; deux détections symétriques réapparaissent : on ne choisit pas.
        objs = [
            SimObject(1, static(600, 500), hidden=[(60, 10**6)]),
            SimObject(2, static(565, 500), first_frame=100),
            SimObject(3, static(635, 500), first_frame=100),
        ]
        frames = simulate(objs, 200)
        tm, _ = run(frames)
        a = tm.tracks[1]
        self.assertEqual(a.last_detection_frame, 59)  # A n'a pas été « volé »
        self.assertEqual(len(tm.confirmed_tracks()), 3)
        amb = events_of(tm, EventType.IDENTITY_AMBIGUOUS)
        self.assertTrue(amb)
        self.assertTrue(all(1 in tm.tracks[t].ambiguous_with for t in (2, 3)))


class TestProximity(unittest.TestCase):
    def test_case4_two_touching_balls_stay_two_tracks(self):
        objs = [SimObject(1, static(500, 500)), SimObject(2, static(561, 500))]
        frames = simulate(objs, 200, noise=1.0)
        tm, results = run(frames)
        self.assertEqual(len(tm.confirmed_tracks()), 2)
        m = identity_metrics(gt_assignments(results))
        self.assertEqual((m["id_switches"], m["false_merges"]), (0, 0))
        # proximité != ambiguïté : deux boules qui se touchent ne déclenchent aucune alerte d'identité
        self.assertEqual(events_of(tm, EventType.IDENTITY_AMBIGUOUS), [])

    def test_case4_two_touching_balls_one_hidden_briefly(self):
        objs = [SimObject(1, static(500, 500)), SimObject(2, static(561, 500), hidden=[(60, 75)])]
        frames = simulate(objs, 200, noise=1.0, bt_patience=2)
        tm, results = run(frames)
        self.assertEqual(len(tm.confirmed_tracks()), 2)
        m = identity_metrics(gt_assignments(results))
        self.assertEqual((m["false_splits"], m["false_merges"]), (0, 0))

    def test_case5_temporary_occlusion_behind_neighbour(self):
        # B arrive au contact de A, est masquée 15 frames derrière A, puis repart.
        b_path = piecewise(
            (0, linear((900, 500), (-8, 0))),
            (50, static(540, 500)),
            (66, linear((540, 500), (8, 0), 66)),
        )
        objs = [SimObject(1, static(500, 500)), SimObject(2, b_path, hidden=[(54, 65)])]
        frames = simulate(objs, 200, noise=0.8, bt_patience=2)
        tm, results = run(frames)
        m = identity_metrics(gt_assignments(results))
        self.assertEqual(m["false_merges"], 0, [e.to_dict() for e in tm.log.events if e.event != "TRACK_CREATED"])
        self.assertEqual(m["false_splits"], 0, [e.to_dict() for e in tm.log.events if e.event != "TRACK_CREATED"])

    def test_case10_false_approach_not_merged(self):
        # B se rapproche de A jusqu'à 66 px (quasi contact) puis repart : toujours 2 boules.
        b_path = piecewise(
            (0, linear((900, 500), (-8, 0))),
            (42, linear((564, 500), (8, 0), 42)),
        )
        objs = [SimObject(1, static(500, 500)), SimObject(2, b_path)]
        frames = simulate(objs, 120, noise=0.8, bt_patience=2)
        tm, results = run(frames)
        m = identity_metrics(gt_assignments(results))
        self.assertEqual((m["false_merges"], m["false_splits"], m["id_switches"]), (0, 0, 0))
        self.assertEqual(len(tm.confirmed_tracks()), 2)


class TestDuplicates(unittest.TestCase):
    """Observé sur vidéo réelle : ByteTrack garde 2 IDs sur le même objet (IoU ~0.6) pendant des centaines de frames."""

    @staticmethod
    def frames(n=200, offset=10.0):
        from app.petanque.models import ObjectType, Observation

        out = []
        for f in range(n):
            a = Observation(f, 480, 480, 520, 520, 0.8, ObjectType.BALL, 1, 0)
            b = Observation(f, 480 + offset, 480, 520 + offset, 520, 0.5, ObjectType.BALL, 2, 1)
            out.append([a, b])
        return out

    def test_sustained_duplicate_is_merged_once_and_not_recreated(self):
        tm, results = run(self.frames())
        self.assertEqual(len(tm.confirmed_tracks()), 1)
        self.assertEqual(len(events_of(tm, EventType.DUPLICATE_SUSPECTED)), 1)  # pas de boucle créer/fusionner
        self.assertEqual(len(events_of(tm, EventType.TRACK_CREATED)), 2)
        self.assertTrue(all(r.suppressed for r in results[-50:]))  # doublon ignoré à chaque frame ensuite

    def test_duplicate_is_released_when_boxes_separate(self):
        from app.petanque.models import ObjectType, Observation

        frames = self.frames(60)
        for f in range(60, 160):  # le 2e ID devient un vrai objet distinct, loin du premier
            frames.append([
                Observation(f, 480, 480, 520, 520, 0.8, ObjectType.BALL, 1, 0),
                Observation(f, 880, 480, 920, 520, 0.8, ObjectType.BALL, 2, 1),
            ])
        tm, results = run(frames)
        self.assertEqual(len(tm.confirmed_tracks()), 2)  # relâché : redevient une boule à part entière
        self.assertEqual(results[-1].suppressed, [])

    def test_two_real_balls_partially_overlapping_are_not_duplicates(self):
        from app.petanque.models import ObjectType, Observation

        frames = [[
            Observation(f, 480, 480, 520, 520, 0.8, ObjectType.BALL, 1, 0),
            Observation(f, 510, 484, 550, 524, 0.8, ObjectType.BALL, 2, 1),  # IoU ~0.17 : occlusion partielle
        ] for f in range(200)]
        tm, _ = run(frames)
        self.assertEqual(len(tm.confirmed_tracks()), 2)
        self.assertEqual(events_of(tm, EventType.DUPLICATE_SUSPECTED), [])


class TestExit(unittest.TestCase):
    field = PolygonField([(100, 100), (1800, 100), (1800, 1000), (100, 1000)])

    def test_case8_fast_ball_leaving_area_is_out_candidate(self):
        obj = SimObject(1, linear((1400, 500), (28, 0)), last_frame=40)  # sort par x=1800 vers f=14
        frames = simulate([obj], 200)
        tm, _ = run(frames, field_=self.field)
        cand = events_of(tm, EventType.BALL_OUT_OF_PLAY_CANDIDATE)
        self.assertEqual(len(cand), 1)
        self.assertIn("trajectory_exits_area", cand[0].reasons)
        self.assertLess(cand[0].confidence, 0.9)  # jamais de certitude
        (t,) = tm.confirmed_tracks()
        out = events_of(tm, EventType.BALL_OUT_OF_PLAY)
        self.assertEqual(t.state, BallState.OUT_OF_PLAY)
        self.assertLessEqual(out[0].confidence, PipelineConfig().tracking.out_of_play_max_confidence)

    def test_hidden_ball_inside_area_is_occluded_not_out(self):
        obj = SimObject(1, linear((400, 500), (18, 0)), hidden=[(30, 10**6)])
        frames = simulate([obj], 60)
        tm, _ = run(frames, field_=self.field)
        self.assertEqual(events_of(tm, EventType.BALL_OUT_OF_PLAY_CANDIDATE), [])
        (t,) = tm.confirmed_tracks()
        self.assertEqual(t.state, BallState.OCCLUDED)

    def test_candidate_is_cancelled_when_ball_reappears(self):
        obj = SimObject(1, linear((1400, 500), (28, 0)), hidden=[(12, 40)], last_frame=20)
        # la boule sort d'un côté, puis on « revoit » une boule cohérente plus tard : pas d'OUT_OF_PLAY forcé
        frames = simulate([obj], 100)
        tm, _ = run(frames, field_=self.field)
        self.assertTrue(events_of(tm, EventType.BALL_OUT_OF_PLAY_CANDIDATE))
        self.assertEqual(events_of(tm, EventType.BALL_OUT_OF_PLAY), [])


class TestBenchmarkVsByteTrack(unittest.TestCase):
    def test_trackmanager_reduces_id_switches_and_splits(self):
        objs = [
            SimObject(1, linear((200, 500), (10, 0)), hidden=[(30, 36), (70, 77)]),
            SimObject(2, static(800, 600), hidden=[(20, 60)]),
            SimObject(3, rolling((300, 300), (20, 6), 10), hidden=[(14, 18)]),
            SimObject(4, static(860, 600)),
        ]
        frames = simulate(objs, 160, bt_patience=2)
        raw = identity_metrics(raw_assignments(frames))
        _, results = run(frames)
        managed = identity_metrics(gt_assignments(results))
        self.assertLess(managed["id_switches"], raw["id_switches"])
        self.assertLess(managed["false_splits"], raw["false_splits"])
        self.assertEqual(managed["false_merges"], 0)
        self.assertGreater(managed["identity_purity"], raw["identity_purity"])


if __name__ == "__main__":
    unittest.main()
