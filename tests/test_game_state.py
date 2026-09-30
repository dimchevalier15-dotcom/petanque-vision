"""Scénarios GameState (cas 7, 8 + mène complète synthétique)."""

import unittest

from app.petanque.config import PipelineConfig
from app.petanque.field import PolygonField
from app.petanque.game_state import MenePhase, PetanqueGameState, PlayerContext, PlayStatus
from app.petanque.models import BallState, EventType, ObjectType
from app.petanque.track_manager import TrackManager
from tests.synthetic import BALL, SimObject, linear, piecewise, rolling, simulate, static

FPS = 30.0
FIELD = PolygonField([(100, 100), (1800, 100), (1800, 1000), (100, 1000)])
JACK = dict(size=BALL / 2, object_type=ObjectType.JACK)


def play(objs, n_frames, cfg=None, seed=0, noise=0.6, player=None):
    cfg = cfg or PipelineConfig()
    frames = simulate(objs, n_frames, seed=seed, noise=noise)
    tm = TrackManager(cfg, FIELD)
    gs = PetanqueGameState(cfg, FPS, FIELD, event_log=tm.log)
    gs.active_player = player
    phases = []
    for f, obs in enumerate(frames):
        res = tm.update(f, obs)
        gs.update(f, tm, res.events)
        if not phases or phases[-1] != gs.phase:
            phases.append(gs.phase)
    return tm, gs, phases


def events(tm, *names):
    return [e for e in tm.log.events if e.event in names]


def jack_thrown():
    # cochonnet lancé, roule puis s'arrête (~frame 40), puis reste immobile
    return SimObject(100, rolling((900, 900), (0, -14), 0), **JACK)


class TestBallPlayed(unittest.TestCase):
    def test_case7_stationary_moving_stationary_is_ball_played(self):
        objs = [
            jack_thrown(),
            SimObject(1, rolling((600, 700), (0, -20), 200)),  # immobile puis poussée
        ]
        tm, gs, phases = play(objs, 400)
        played = events(tm, EventType.BALL_PLAYED)
        self.assertEqual(len(played), 1)
        ev = played[0].to_dict()
        self.assertEqual(ev["kind"], "moved_from_rest")
        self.assertEqual(ev["pattern"], "STATIONARY>MOVING>SLOWING>STATIONARY")
        self.assertLess(ev["confidence"], 0.8)  # départ depuis immobile = moins sûr qu'un lancer vu en vol
        self.assertEqual(gs.balls[0].status, PlayStatus.IN_PLAY)
        self.assertEqual(tm.tracks[gs.balls[0].logical_track_id].state, BallState.STATIONARY)
        self.assertIn(MenePhase.BALL_PLAY, phases)
        self.assertEqual(phases[-1], MenePhase.NEXT_BALL)

    def test_new_stationary_object_is_not_a_played_ball(self):
        objs = [jack_thrown(), SimObject(1, static(600, 700), first_frame=200)]  # apparaît immobile
        tm, gs, _ = play(objs, 400)
        self.assertEqual(events(tm, EventType.BALL_PLAYED), [])
        self.assertEqual(gs.balls, [])

    def test_ball_moving_before_jack_is_stable_is_ignored(self):
        objs = [jack_thrown(), SimObject(1, rolling((600, 700), (0, -20), 10))]
        tm, gs, _ = play(objs, 200)
        self.assertEqual(gs.balls, [])

    def test_tiny_jitter_is_not_a_play(self):
        objs = [jack_thrown(), SimObject(1, static(600, 700))]
        tm, gs, _ = play(objs, 400, noise=2.0)
        self.assertEqual(gs.balls, [])


class TestMene(unittest.TestCase):
    def build(self):
        return [
            jack_thrown(),
            # ball_1 / ball_2 : lancers vus en vol, s'arrêtent dans le terrain
            SimObject(1, rolling((500, 900), (8, -25), 100), first_frame=100),
            SimObject(2, rolling((1200, 950), (-6, -25), 200), first_frame=200),
            # ball_3 : tir très rapide qui sort de la zone
            SimObject(3, linear((1500, 600), (45, 0), 300), first_frame=300, last_frame=330),
            # ball_4 frappe ball_5 qui était immobile (déplacement, pas un lancer)
            SimObject(4, piecewise((0, linear((300, 500), (20, 0), 420)), (442, static(740, 500))), first_frame=420),
            SimObject(5, piecewise((0, static(800, 500)), (442, rolling((800, 500), (14, 0), 442))), first_frame=0),
        ]

    def test_full_mene_sequence(self):
        cfg = PipelineConfig()
        cfg.game.expected_ball_count = 4
        tm, gs, phases = play(self.build(), 640, cfg=cfg)
        # ordre de jeu : 3 lancers + ball_4 ; ball_5 est déplacée, pas jouée
        self.assertEqual([b.order_index for b in gs.balls], [1, 2, 3, 4])
        statuses = [b.status for b in gs.balls]
        self.assertEqual(statuses[0], PlayStatus.IN_PLAY)
        self.assertEqual(statuses[1], PlayStatus.IN_PLAY)
        self.assertIn(statuses[2], (PlayStatus.OUT_OF_PLAY_CANDIDATE, PlayStatus.OUT_OF_PLAY))
        self.assertLess(gs.balls[2].confidence, 0.6)  # jamais de certitude sur une sortie
        self.assertEqual(len(gs.displaced), 1)
        self.assertEqual(events(tm, EventType.BALL_DISPLACED)[0].logical_track_id, gs.displaced[0]["track"])
        self.assertNotIn(gs.displaced[0]["track"], [b.logical_track_id for b in gs.balls])
        self.assertEqual(gs.jack.state, BallState.STATIONARY.value)
        self.assertIsNotNone(gs.jack.stabilized_frame)
        self.assertEqual(
            phases[:6],
            [MenePhase.SETUP, MenePhase.JACK_THROW, MenePhase.JACK_STABILIZED, MenePhase.BALL_PLAY,
             MenePhase.BALL_STABILIZED, MenePhase.NEXT_BALL],
        )
        self.assertEqual(gs.phase, MenePhase.END_OF_MENE)
        report = gs.report(tm)
        self.assertIn("ball_1", report)
        self.assertIn("END_OF_MENE", report)

    def test_tolerates_missing_balls(self):
        # expected_ball_count=12 mais seulement 4 lancers détectés : la mène « continue », sans inventer.
        tm, gs, _ = play(self.build(), 640)
        self.assertEqual(len(gs.balls), 4)
        self.assertNotEqual(gs.phase, MenePhase.END_OF_MENE)
        self.assertIn("MENE CONTINUES", gs.report(tm))

    def test_player_context_is_injected_not_inferred(self):
        ctx = PlayerContext(player_id="PLAYER_A_IN_CIRCLE", team="A", source="test", confidence=1.0)
        tm, gs, _ = play(self.build(), 640, player=ctx)
        self.assertTrue(gs.balls)
        b = gs.balls[0]
        self.assertEqual((b.order_index, b.player_context.player_id, b.player_context.team), (1, "PLAYER_A_IN_CIRCLE", "A"))
        tm2, gs2, _ = play(self.build(), 640)
        self.assertIsNone(gs2.balls[0].player_context)  # sans contexte : pas de joueur inventé


class TestJack(unittest.TestCase):
    def test_jack_identity_survives_fragmented_trajectory(self):
        jack = SimObject(100, rolling((900, 900), (0, -14), 0), hidden=[(10, 14)], **JACK)
        frames = simulate([jack], 150, bt_patience=1)
        cfg = PipelineConfig()
        tm = TrackManager(cfg, FIELD)
        gs = PetanqueGameState(cfg, FPS, FIELD, event_log=tm.log)
        for f, obs in enumerate(frames):
            gs.update(f, tm, tm.update(f, obs).events)
        jacks = [t for t in tm.confirmed_tracks() if t.object_type == ObjectType.JACK]
        self.assertEqual(len(jacks), 1)
        self.assertEqual(gs.jack.logical_track_id, jacks[0].logical_track_id)
        self.assertEqual(gs.phase, MenePhase.JACK_STABILIZED)
        self.assertIsNotNone(gs.jack.position)


if __name__ == "__main__":
    unittest.main()
