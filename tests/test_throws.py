"""Scénarios ThrowEventDetector : lancers, collisions, boules déplacées, cochonnet, cas ambigus.

Règle vérifiée partout : une collision ne doit JAMAIS créer de ThrowEvent en plus.
"""

import json
import unittest

from app.petanque.collisions import CollisionDetector
from app.petanque.config import PipelineConfig
from app.petanque.field import PolygonField
from app.petanque.game_state import PetanqueGameState, PlayerContext
from app.petanque.models import EventType, ObjectType
from app.petanque.throw_detector import ThrowEventDetector
from app.petanque.throws import MovementClass, Owner, ThrowEvent, ThrowResult, ThrowType
from app.petanque.track_manager import TrackManager
from tests.synthetic import BALL, SimObject, linear, piecewise, rolling, simulate, static

FPS = 30.0
FIELD = PolygonField([(100, 100), (1800, 100), (1800, 1000), (100, 1000)])
JACK = dict(size=BALL / 2, object_type=ObjectType.JACK)
Y = 600.0  # ligne de jeu des scénarios


def play(objs, n_frames, cfg=None, seed=0, noise=0.6, bt_patience=3, finalize=True, td_setup=None):
    cfg = cfg or PipelineConfig()
    frames = simulate(objs, n_frames, seed=seed, noise=noise, bt_patience=bt_patience)
    tm = TrackManager(cfg, FIELD)
    gs = PetanqueGameState(cfg, FPS, FIELD, event_log=tm.log)
    td = ThrowEventDetector(cfg, FPS, FIELD, tm.log)
    if td_setup:
        td_setup(td)
    for f, obs in enumerate(frames):
        res = tm.update(f, obs)
        gs.update(f, tm, res.events)
        td.update(f, tm, res.events, gs)
    if finalize:
        td.finalize(n_frames - 1, tm)
    return tm, gs, td


def jack_thrown(gt=100):
    """Cochonnet lancé puis immobile : stabilisé vers la frame 40 (le contexte « but stabilisé »)."""
    return SimObject(gt, rolling((900, 900), (0, -14), 0), **JACK)


def thrown(gt, p0, v0, start, decay=0.96, **kw):
    """Boule qui *apparaît* en mouvement (cas réel : boule lancée entrant dans le cadre)."""
    return SimObject(gt, rolling(p0, v0, start, decay), first_frame=start, **kw)


def held_then_thrown(gt, p0, v0, appear, release, decay=0.96):
    """Boule visible immobile (en main) puis lancée : STATIONARY -> MOVING."""
    return SimObject(gt, rolling(p0, v0, release, decay), first_frame=appear)


def static_ball(gt, x, y=Y, **kw):
    return SimObject(gt, static(x, y), **kw)


def sure(td):
    return [t for t in td.throws if t.event_type == ThrowType.THROW]


def movements_of(td, cls):
    return [m for m in td.movements if m.classification == cls]


def hit_paths(a_start, a_p0, a_v, b_pos, contact_gap, a_after, b_after, decay_a=0.93, decay_b=0.95):
    """A arrive en ligne droite sur B (immobile) ; au contact A et B repartent avec (a_after, b_after)."""
    k = round((b_pos[0] - contact_gap - a_p0[0]) / a_v[0])
    tc = a_start + k
    pc = (a_p0[0] + a_v[0] * k, a_p0[1])
    a_path = piecewise((a_start, linear(a_p0, a_v, a_start)), (tc, rolling(pc, a_after, tc, decay_a)))
    b_path = piecewise((0, static(*b_pos)), (tc, rolling(b_pos, b_after, tc, decay_b)))
    return a_path, b_path, tc


class TestBasicThrows(unittest.TestCase):
    # 1. lancer normal
    def test_normal_throw(self):
        _, _, td = play([jack_thrown(), thrown(1, (400, Y), (30, 0), start=100)], 330)
        self.assertEqual(len(td.throws), 1)
        t = td.throws[0]
        self.assertEqual(t.event_type, ThrowType.THROW)
        self.assertEqual(t.final_state, ThrowResult.ON_FIELD)
        self.assertEqual(t.owner, Owner.UNASSIGNED)
        self.assertIsNone(t.owner_source)
        self.assertGreaterEqual(t.confidence, 0.8)
        self.assertAlmostEqual(t.start_frame, 100, delta=8)
        self.assertGreater(t.end_frame, t.start_frame + 40)
        self.assertEqual(t.kind, "entered_in_motion")
        self.assertGreater(t.metrics["travel_diam"], 8)
        self.assertGreater(len(t.trajectory), 40)
        self.assertEqual(t.trajectory[0][0], t.start_frame)
        self.assertLess(abs(t.initial_position[0] - 400), 40)  # départ observé près de la position d'entrée
        self.assertGreater(t.final_position[0], 900)
        self.assertIn("movement_ended_with_stop", t.detection_reasons)
        self.assertIn("after_jack_stabilization", t.detection_reasons)

    # 2. lancer + arrêt, depuis une boule d'abord immobile (en main)
    def test_throw_from_held_ball_and_stop(self):
        _, _, td = play([jack_thrown(), held_then_thrown(1, (400, Y), (30, 0), appear=80, release=120)], 340)
        t = sure(td)[0]
        self.assertEqual(len(td.throws), 1)
        self.assertEqual(t.kind, "moved_from_rest")
        self.assertIn("recently_appeared_ball_started_moving", t.detection_reasons)
        self.assertGreater(t.metrics["pre_stationary_s"], 0.5)
        self.assertGreater(t.metrics["post_stationary_s"], 0.2)  # durée d'immobilité après le mouvement
        self.assertEqual(t.final_state, ThrowResult.ON_FIELD)

    # 3. lancer + sortie du terrain
    def test_throw_leaving_the_field_is_out_of_play_only_when_confirmed(self):
        ball = thrown(1, (1400, Y), (45, 0), start=100, decay=0.97, last_frame=118)  # sort par la droite
        tm, _, td = play([jack_thrown(), ball], 130, finalize=False)
        self.assertEqual(td.throws, [])  # disparition récente : candidat de sortie, pas de conclusion
        self.assertTrue(any(ep.exit_candidate for ep in td.open_episodes))
        tm, _, td = play([jack_thrown(), ball], 260)
        self.assertEqual(len(td.throws), 1)
        t = td.throws[0]
        self.assertEqual(t.final_state, ThrowResult.OUT_OF_PLAY)
        self.assertLessEqual(t.final_state_confidence, 0.75)  # jamais de certitude sur une disparition
        self.assertIn("disappeared_towards_exit", t.final_state_reasons)
        self.assertEqual(t.end_frame, 118)

    def test_disappearing_mid_field_is_not_out_of_play(self):
        ball = thrown(1, (400, Y), (20, 0), start=100, decay=0.98, last_frame=125)  # disparaît en plein terrain
        _, _, td = play([jack_thrown(), ball], 300)
        self.assertEqual(len(td.throws), 1)
        t = td.throws[0]
        self.assertEqual(t.final_state, ThrowResult.UNKNOWN)
        self.assertEqual(t.event_type, ThrowType.UNKNOWN)  # fin non observée : pas de THROW confirmé
        self.assertLess(t.confidence, PipelineConfig().throws.throw_min_confidence)
        self.assertIn("track_lost_while_moving", t.final_state_reasons)

    # 4. lancer + occultation : même track logique, un seul lancer
    def test_throw_with_occlusion_is_one_throw(self):
        ball = thrown(1, (400, Y), (30, 0), start=100, hidden=[(125, 137)])
        tm, _, td = play([jack_thrown(), ball], 360, bt_patience=1)
        self.assertEqual(len(td.throws), 1)
        t = td.throws[0]
        self.assertEqual(t.event_type, ThrowType.THROW)
        self.assertGreaterEqual(t.metrics["occlusion_frames"], 10)
        self.assertTrue(any(r.startswith("occlusion_during_movement") for r in t.detection_reasons))
        self.assertGreaterEqual(len(t.bytetrack_ids), 2)  # ByteTrack a changé d'ID, le lancer est resté lié au track logique

    # 8. track fragmenté pendant un lancer (ByteTrack change d'ID à chaque trou)
    def test_fragmented_track_during_throw_is_one_throw(self):
        ball = thrown(1, (400, Y), (30, 0), start=100, hidden=[(120, 123), (140, 144)])
        _, _, td = play([jack_thrown(), ball], 360, bt_patience=1)
        self.assertEqual(len(td.throws), 1, [(t.ball_track_id, t.start_frame, t.end_frame) for t in td.throws])
        self.assertEqual(td.throws[0].event_type, ThrowType.THROW)
        self.assertEqual(len({m.track_id for m in td.movements if m.classification == MovementClass.THROW}), 1)

    # 11. plusieurs lancers successifs
    def test_successive_throws_are_chronological_and_distinct(self):
        balls = [
            thrown(1, (400, 500), (30, 0), start=100),
            thrown(2, (400, 700), (30, 0), start=300),
            held_then_thrown(3, (400, 900), (26, 0), appear=460, release=500),
            thrown(4, (400, 560), (28, 0), start=700),
        ]
        _, _, td = play([jack_thrown()] + balls, 950)
        ts = sure(td)
        self.assertEqual(len(ts), 4, [(t.ball_track_id, t.start_frame) for t in td.throws])
        self.assertEqual([t.throw_id for t in ts], [1, 2, 3, 4])
        self.assertEqual(sorted(t.start_frame for t in ts), [t.start_frame for t in ts])  # chronologique
        self.assertEqual(len({t.ball_track_id for t in ts}), 4)
        self.assertEqual(td.collisions, [])
        self.assertTrue(all(t.owner == Owner.UNASSIGNED for t in ts))

    def test_no_jack_lowers_confidence_but_keeps_the_throw(self):
        _, _, td = play([thrown(1, (400, Y), (30, 0), start=100)], 330)
        self.assertEqual(len(td.throws), 1)
        self.assertIn("no_jack_seen", td.throws[0].detection_reasons)

    def test_movement_before_jack_is_stable_is_unknown_not_throw(self):
        # boule qui roule alors que le cochonnet bouge encore : on ne confirme pas
        jack = SimObject(100, rolling((900, 900), (0, -14), 100), first_frame=0, **JACK)
        _, _, td = play([jack, thrown(1, (400, Y), (30, 0), start=100)], 330)
        self.assertEqual([t for t in sure(td) if t.start_frame < 140], [])


class TestNotThrows(unittest.TestCase):
    # 9. faux mouvements
    def test_isolated_detection_is_not_a_throw(self):
        flash = SimObject(1, linear((500, Y), (25, 0), 100), first_frame=100, last_frame=101)  # 2 frames
        _, _, td = play([jack_thrown(), flash], 250)
        self.assertEqual(td.throws, [])
        self.assertEqual(td.collisions, [])

    def test_slow_drift_is_not_a_throw(self):
        drift = SimObject(1, linear((500, Y), (1.0, 0), 100), first_frame=0)  # 0.02 diam/frame
        _, _, td = play([jack_thrown(), drift], 400)
        self.assertEqual(td.throws, [])

    def test_small_nudge_is_ignored(self):
        nudged = SimObject(1, piecewise((0, static(600, Y)), (150, rolling((600, Y), (25, 0), 150, 0.85))), first_frame=0)
        _, _, td = play([jack_thrown(), nudged], 400)
        self.assertEqual(td.throws, [])  # ~2.4 diamètres : sous le déplacement minimal
        self.assertTrue(any(m.classification == MovementClass.IGNORED for m in td.movements))

    # 7. boule déplacée sans être lancée (aucune cause visible)
    def test_old_ball_moving_without_cause_is_not_a_confirmed_throw(self):
        old = SimObject(1, piecewise((0, static(600, Y)), (200, rolling((600, Y), (30, 0), 200, 0.96))), first_frame=0)
        _, _, td = play([jack_thrown(), old], 480)
        self.assertEqual(sure(td), [])
        self.assertTrue(td.throws)  # gardé comme UNKNOWN pour revue humaine, pas effacé
        t = td.throws[0]
        self.assertEqual(t.event_type, ThrowType.UNKNOWN)
        self.assertIn("old_ball_moved_without_identified_cause", t.detection_reasons)


class TestCollisions(unittest.TestCase):
    def setUp(self):
        self.jack = jack_thrown()

    # 5. collision avec une boule immobile : un seul lancer, l'autre est déplacée
    def test_collision_with_static_ball_does_not_create_a_second_throw(self):
        a_path, b_path, tc = hit_paths(150, (200, Y), (30, 0), (920, Y), 60, a_after=(0, 0), b_after=(24, 0))
        a = SimObject(1, a_path, first_frame=150)
        b = SimObject(2, b_path, first_frame=0)
        tm, _, td = play([self.jack, a, b], 420)
        self.assertEqual(len(td.throws), 1, [(t.ball_track_id, t.start_frame, t.event_type) for t in td.throws])
        thrower = td.throws[0]
        self.assertEqual(thrower.event_type, ThrowType.THROW)
        self.assertEqual(len(td.collisions), 1)
        c = td.collisions[0]
        self.assertEqual(c.source_ball, thrower.ball_track_id)
        self.assertEqual(c.kind, "impact_on_static")
        self.assertAlmostEqual(c.frame, tc, delta=4)
        self.assertGreaterEqual(c.confidence, 0.7)
        self.assertIn("target_started_moving", c.reasons)
        self.assertEqual(thrower.collisions_as_source[0]["target_ball"], c.target_ball)
        displaced = movements_of(td, MovementClass.DISPLACED)
        self.assertEqual(len(displaced), 1)
        self.assertEqual(displaced[0].track_id, c.target_ball)
        self.assertEqual(displaced[0].displaced_by, thrower.ball_track_id)

    def test_young_ball_pushed_by_a_shot_is_not_a_throw_but_would_be_without_collision_logic(self):
        # B vient d'apparaître (en main) : sans l'analyse des collisions elle ressemble à un lancer (départ du repos,
        # track jeune). Le témoin ci-dessous prouve que ce scénario *dépend* bien de la détection de collision.
        a_path, b_path, _ = hit_paths(150, (200, Y), (30, 0), (920, Y), 60, a_after=(0, 0), b_after=(34, 0), decay_b=0.96)
        objs = lambda: [self.jack, SimObject(1, a_path, first_frame=150), SimObject(2, b_path, first_frame=100)]  # noqa: E731
        _, _, td = play(objs(), 420)
        self.assertEqual(len(sure(td)), 1)
        self.assertTrue(movements_of(td, MovementClass.DISPLACED))
        blind = PipelineConfig()
        blind.throws.collision_min_confidence = 2.0  # témoin : collisions jamais détectées
        _, _, td_blind = play(objs(), 420, cfg=blind)
        self.assertEqual(td_blind.collisions, [])
        self.assertEqual(len(sure(td_blind)), 2, "sans collision, la boule poussée devient un faux lancer")

    def test_shot_that_pushes_ball_out_of_the_field(self):
        a_path, b_path, _ = hit_paths(150, (900, Y), (45, 0), (1500, Y), 60, a_after=(0, 0), b_after=(40, 0), decay_b=0.97)
        b = SimObject(2, b_path, first_frame=0, last_frame=175)
        tm, _, td = play([self.jack, SimObject(1, a_path, first_frame=150), b], 330)
        self.assertEqual(len(sure(td)), 1)  # seul le tireur est un lancer
        self.assertEqual(len(td.collisions), 1)
        self.assertEqual(movements_of(td, MovementClass.THROW)[0].track_id, td.collisions[0].source_ball)

    # 6. collision entre deux boules en mouvement : deux lancers, pas un de plus
    def test_collision_between_two_moving_balls(self):
        tc = 89
        a_path = piecewise((70, linear((300, Y), (30, 0), 70)), (tc, rolling((870, Y), (8, 0), tc, 0.95)))
        b_path = piecewise((70, linear((1215, Y), (-15, 0), 70)), (tc, rolling((930, Y), (20, 0), tc, 0.95)))
        a, b = SimObject(1, a_path, first_frame=70), SimObject(2, b_path, first_frame=70)
        _, _, td = play([self.jack, a, b], 330)
        self.assertEqual(len(td.collisions), 1, [c.reasons for c in td.collisions])
        self.assertEqual(td.collisions[0].kind, "moving_moving")
        self.assertEqual(len(td.throws), 2, [(t.ball_track_id, t.start_frame) for t in td.throws])
        self.assertEqual({t.ball_track_id for t in td.throws}, {td.collisions[0].source_ball, td.collisions[0].target_ball})
        self.assertTrue(all(t.event_type == ThrowType.THROW for t in td.throws))
        for t in td.throws:  # chacun garde son propre lancer, relié à l'autre par la collision
            self.assertEqual(len(t.collisions_as_source) + len(t.collisions_as_target), 1)

    # 10. deux boules proches : pas de fausse collision, pas de faux lancer
    def test_two_close_balls_and_a_throw_passing_close_by(self):
        calls = []
        real_evaluate = CollisionDetector.evaluate

        def spy(detector, *args, **kw):
            result = real_evaluate(detector, *args, **kw)
            calls.append(result)
            return result

        CollisionDetector.evaluate = spy
        self.addCleanup(setattr, CollisionDetector, "evaluate", real_evaluate)
        p1, p2 = static_ball(1, 900), static_ball(2, 978)  # 1.3 diamètre l'une de l'autre
        passing = thrown(3, (650, Y + 80), (30, 0), start=150, decay=0.97)  # passe à 1.3 diamètre, encore vite
        tm, _, td = play([self.jack, p1, p2, passing], 420)
        # le passage est bel et bien évalué comme contact possible ... puis rejeté faute de réponse
        self.assertTrue(calls, "le test doit réellement solliciter l'évaluation de contact")
        self.assertTrue(all(r is None for r in calls))
        self.assertEqual(len(td.throws), 1)
        self.assertEqual(td.collisions, [])
        self.assertEqual([m for m in td.movements if m.object_type == "BALL" and m.start_frame < 140], [])  # p1 / p2 n'ont pas bougé

    def test_ball_rolling_to_a_stop_next_to_another_is_not_a_collision(self):
        p1 = static_ball(1, 900)
        slow = thrown(2, (300, Y), (22, 0), start=150, decay=0.93)  # ~ 315 px : s'arrête à ~1.1 diamètre de p1
        _, _, td = play([self.jack, p1, slow], 420)
        self.assertEqual(len(sure(td)), 1)
        self.assertEqual(td.collisions, [], [c.reasons for c in td.collisions])

    def test_glancing_contact_with_no_response_is_low_confidence(self):
        # A (déviée d'environ 40 degrés) frôle B qui ne bouge pas : candidat faible, B n'est pas lancée
        # (même profil que le contact réel de la frame 907 de la vidéo 174531)
        a_path = piecewise((150, linear((700, 400), (20, 15), 150)), (168, rolling((1060, 670), (20, -1), 168, 0.96)))
        tm, _, td = play([self.jack, static_ball(2, 1093, 626), SimObject(1, a_path, first_frame=150)], 400)
        self.assertEqual(len(td.collisions), 1)
        c = td.collisions[0]
        self.assertLessEqual(c.confidence, 0.65)
        self.assertIn("no_target_response", c.reasons)
        self.assertIn("source_changed_direction", c.reasons)
        self.assertEqual(len(td.throws), 1)  # seule A est lancée
        self.assertEqual(td.throws[0].ball_track_id, c.source_ball)
        self.assertNotEqual(td.throws[0].ball_track_id, c.target_ball)

    # 12. cochonnet séparé des boules
    def test_jack_is_never_a_throw(self):
        _, gs, td = play([jack_thrown()], 200)
        self.assertEqual(td.throws, [])
        self.assertTrue(movements_of(td, MovementClass.JACK_THROW))
        self.assertTrue(all(m.object_type == "JACK" for m in td.movements))

    def test_ball_hitting_the_jack_moves_the_jack_without_extra_throw(self):
        jack_path = piecewise((0, rolling((900, 900), (0, -14), 0)), (0, rolling((900, 900), (0, -14), 0)))
        jack_stop = (900, 900 - round(14 * 0.93 / 0.07))  # ~ position d'arrêt du cochonnet
        a_path, j_path, _ = hit_paths(200, (200, jack_stop[1]), (30, 0), (jack_stop[0], jack_stop[1]), 45,
                                      a_after=(0, 0), b_after=(22, 0))
        jack = SimObject(100, piecewise((0, jack_path), (150, j_path)), **JACK)
        _, _, td = play([jack, SimObject(1, a_path, first_frame=200)], 480)
        self.assertEqual(len(sure(td)), 1)
        self.assertEqual(td.throws[0].ball_track_id, td.collisions[0].source_ball)
        self.assertEqual(td.collisions[0].target_type, "JACK")
        moved = movements_of(td, MovementClass.JACK_DISPLACED)
        self.assertTrue(moved)
        self.assertEqual(moved[-1].displaced_by, td.throws[0].ball_track_id)


class TestRestingBallPushedAwayUnderNewTrack(unittest.TestCase):
    """Réel (180525, frame ~1370) : une boule au repos disparaît, une « nouvelle » boule repart à côté : c'est la même."""

    def _scene(self, vanish_at=400, restart_at=406, offset=(0, -45), restart_v=(0, -25)):
        a = thrown(1, (200, Y), (30, 0), start=100, decay=0.95, last_frame=vanish_at)
        first = play([jack_thrown(), a], 330)[2].throws[0].final_position
        start = (first[0] + offset[0], first[1] + offset[1])
        c = thrown(3, start, restart_v, start=restart_at, decay=0.95)
        return play([jack_thrown(), a, c], 560)

    def test_ball_that_restarts_where_a_resting_ball_vanished_is_not_a_new_throw(self):
        _, _, td = self._scene()
        self.assertEqual(len(sure(td)), 1, [(t.start_frame, t.final_state) for t in td.throws])
        pushed = movements_of(td, MovementClass.DISPLACED)
        self.assertEqual(len(pushed), 1)
        self.assertIsNotNone(pushed[0].continues_track)
        self.assertIn("starts_where_resting_ball_just_vanished", pushed[0].reasons)

    def test_same_movement_is_a_throw_when_no_resting_ball_vanished(self):
        a = thrown(1, (200, Y), (30, 0), start=100, decay=0.95)  # ne disparaît pas
        first = play([jack_thrown(), a], 330)[2].throws[0].final_position
        c = thrown(3, (first[0], first[1] - 200), (0, -25), start=406, decay=0.95)  # loin : entre seule
        _, _, td = play([jack_thrown(), a, c], 560)
        self.assertFalse(movements_of(td, MovementClass.DISPLACED))
        self.assertEqual(len(sure(td)), 2)

    def test_a_ball_far_from_the_vanished_one_is_still_a_throw(self):
        _, _, td = self._scene(offset=(0, -400))
        self.assertFalse(movements_of(td, MovementClass.DISPLACED))
        self.assertEqual(len(sure(td)), 2)

    def test_a_vanished_ball_long_before_does_not_explain_the_movement(self):
        _, _, td = self._scene(restart_at=470)  # 70 frames après la disparition (> 0.5 s)
        self.assertFalse(movements_of(td, MovementClass.DISPLACED))
        self.assertEqual(len(sure(td)), 2)


    def test_a_ball_that_was_still_rolling_when_it_vanished_is_not_a_resting_ball(self):
        a = thrown(1, (200, Y), (30, 0), start=100, decay=0.95, last_frame=125)  # en plein mouvement
        tm = play([jack_thrown(), a], 200)[0]
        last = max((t for t in tm.tracks.values() if t.object_type == ObjectType.BALL), key=lambda t: t.last_detection_frame)
        c = thrown(3, (last.current_position[0], last.current_position[1] - 45), (0, -25), start=131, decay=0.95)
        _, _, td = play([jack_thrown(), a, c], 400)
        self.assertFalse(movements_of(td, MovementClass.DISPLACED))


class TestContextAndPlumbing(unittest.TestCase):
    def test_player_context_is_recorded_but_never_assigned(self):
        def setup(td):
            td.active_player = PlayerContext("PLAYER_A_IN_CIRCLE", source="TEST", confidence=0.9)

        _, _, td = play([jack_thrown(), thrown(1, (400, Y), (30, 0), start=100)], 330, td_setup=setup)
        t = td.throws[0]
        self.assertEqual(t.player_context["player_id"], "PLAYER_A_IN_CIRCLE")
        self.assertEqual(t.owner, Owner.UNASSIGNED)  # le contexte est une entrée future, pas une décision

    def test_throw_zone_is_optional_and_only_adjusts_confidence(self):
        base = play([jack_thrown(), thrown(1, (400, Y), (30, 0), start=100)], 330)[2].throws[0]
        self.assertNotIn("origin_distance_to_zone_diam", base.metrics)
        cfg = PipelineConfig()
        cfg.throws.throw_zone_center = [400.0, Y]
        cfg.throws.throw_zone_radius = 100.0
        near = play([jack_thrown(), thrown(1, (400, Y), (30, 0), start=100)], 330, cfg=cfg)[2].throws[0]
        self.assertIn("origin_near_throw_zone", near.detection_reasons)
        cfg.throws.throw_zone_center = [1500.0, 200.0]
        far = play([jack_thrown(), thrown(1, (400, Y), (30, 0), start=100)], 330, cfg=cfg)[2].throws[0]
        self.assertIn("origin_far_from_throw_zone", far.detection_reasons)
        self.assertLess(far.confidence, base.confidence)
        self.assertEqual(far.event_type, ThrowType.THROW)  # zone : indice, jamais règle obligatoire

    def test_stream_ending_mid_movement_is_unknown(self):
        _, _, td = play([jack_thrown(), thrown(1, (400, Y), (30, 0), start=100)], 125)
        self.assertEqual(len(td.throws), 1)
        self.assertEqual(td.throws[0].final_state, ThrowResult.UNKNOWN)
        self.assertEqual(td.throws[0].event_type, ThrowType.UNKNOWN)

    def test_events_are_logged_with_reasons(self):
        tm, _, td = play([jack_thrown(), thrown(1, (400, Y), (30, 0), start=100)], 330)
        evs = [e for e in tm.log.events if e.event == EventType.THROW_DETECTED]
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].data["throw_id"], 1)
        self.assertTrue(evs[0].reasons)
        self.assertTrue(any(e.event == EventType.JACK_MOVEMENT for e in tm.log.events))

    def test_throw_event_roundtrips_through_json(self):
        _, _, td = play([jack_thrown(), thrown(1, (400, Y), (30, 0), start=100)], 330)
        t = td.throws[0]
        back = ThrowEvent.from_dict(json.loads(json.dumps(t.to_dict())))
        self.assertEqual(back.to_dict(), t.to_dict())
        self.assertEqual(back.owner, Owner.UNASSIGNED)

    def test_detection_is_deterministic(self):
        objs = lambda: [jack_thrown(), thrown(1, (400, Y), (30, 0), start=100, hidden=[(125, 137)])]  # noqa: E731
        a = [t.to_dict() for t in play(objs(), 360, bt_patience=1)[2].throws]
        b = [t.to_dict() for t in play(objs(), 360, bt_patience=1)[2].throws]
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()


class TestFragmentsAndContactEnd(unittest.TestCase):
    """Cas réels du 3 min : boule floue détectée seulement en fin de course ; boule qui s'arrête contre une autre."""

    @staticmethod
    def _two_fragments(dy, gap, v=14, k=9):
        p1 = rolling((400, Y), (v, 0), 100, 0.985)
        end = 100 + k
        p2 = rolling((400 + v * k + v * gap, Y + dy), (v, 0), end + gap, 0.985)
        return [jack_thrown(), SimObject(1, p1, first_frame=100, last_frame=end), SimObject(2, p2, first_frame=end + gap)]

    def test_two_short_fragments_of_one_throw_are_recombined(self):
        _, _, td = play(self._two_fragments(dy=60, gap=4), 420)
        self.assertEqual(len(td.throws), 1)
        t = td.throws[0]
        self.assertEqual(t.event_type, ThrowType.THROW)
        self.assertAlmostEqual(t.start_frame, 101, delta=6)  # le départ est celui du 1er fragment
        self.assertTrue(any(r.startswith("fragments_recombined") for r in t.detection_reasons))
        self.assertGreaterEqual(len(t.track_id_history), 2)
        first = [m for m in td.movements if m.classification == MovementClass.IGNORED]
        self.assertTrue(first and all(m.throw_id == t.throw_id for m in first))  # le fragment est relié, pas perdu

    def test_distant_fragments_are_not_recombined(self):
        _, _, td = play(self._two_fragments(dy=400, gap=4), 420)
        self.assertFalse(any(any(r.startswith("fragments_recombined") for r in t.detection_reasons) for t in td.throws))
        ignored = [m for m in td.movements if m.classification == MovementClass.IGNORED]
        self.assertTrue(ignored and all(m.throw_id is None for m in ignored))

    def test_fragments_moving_in_opposite_directions_are_not_recombined(self):
        p1 = rolling((800, Y), (14, 0), 100, 0.985)
        p2 = rolling((926, Y), (-14, 0), 113, 0.985)  # repart en sens inverse, à 1 diam de la fin du 1er
        objs = [jack_thrown(), SimObject(1, p1, first_frame=100, last_frame=109),
                SimObject(2, p2, first_frame=113)]
        _, _, td = play(objs, 420)
        self.assertFalse(any(any(r.startswith("fragments_recombined") for r in t.detection_reasons) for t in td.throws))

    def test_track_lost_against_a_present_ball_is_a_throw_that_stopped_there(self):
        a = thrown(1, (400, Y), (30, 0), start=100, decay=0.93, last_frame=130)  # disparaît en ralentissant
        _, _, td = play([jack_thrown(), a, static_ball(2, 850)], 400)
        self.assertEqual(len(td.throws), 1)
        t = td.throws[0]
        self.assertEqual(t.event_type, ThrowType.THROW)
        self.assertEqual(t.final_state, ThrowResult.ON_FIELD)
        self.assertLessEqual(t.final_state_confidence, 0.5)  # résultat moins sûr qu'un arrêt observé
        self.assertLessEqual(t.confidence, PipelineConfig().throws.contact_end_cap)
        self.assertIn("lost_in_contact_with_ball_2", t.final_state_reasons[0])
        self.assertEqual(sure(td), [t])

    def test_same_disappearance_far_from_any_ball_stays_unknown(self):
        a = thrown(1, (400, Y), (30, 0), start=100, decay=0.93, last_frame=130)
        _, _, td = play([jack_thrown(), a], 400)
        t = td.throws[0]
        self.assertEqual(t.event_type, ThrowType.UNKNOWN)
        self.assertEqual(t.final_state, ThrowResult.UNKNOWN)

    def test_contact_end_can_be_disabled(self):
        cfg = PipelineConfig()
        cfg.throws.contact_end_distance = 0.0
        a = thrown(1, (400, Y), (30, 0), start=100, decay=0.93, last_frame=130)
        _, _, td = play([jack_thrown(), a, static_ball(2, 850)], 400, cfg=cfg)
        self.assertEqual(td.throws[0].event_type, ThrowType.UNKNOWN)

    def test_fast_ball_vanishing_next_to_a_ball_without_hitting_it_is_not_promoted(self):
        a = thrown(1, (400, Y), (30, 0), start=100, decay=0.995, last_frame=118)  # encore très rapide
        _, _, td = play([jack_thrown(), a, static_ball(2, 850)], 400)
        self.assertTrue(all(t.event_type == ThrowType.UNKNOWN for t in td.throws))
