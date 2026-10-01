"""Numérotation des boules lancées + vidéo « boules numérotées » : seules les boules lancées et restées en jeu."""

import unittest

import numpy as np

from app.petanque.ball_numbering import BallLabelRenderer, number_balls, relocate_contact_ends
from app.petanque.models import ObjectType, Observation
from app.petanque.throws import ThrowEvent, ThrowResult, ThrowType
from tests.test_throws import Y, jack_thrown, play, static_ball, thrown

SIZE = 50.0


def make_throw(tid, start, end, final, etype=ThrowType.THROW, state=ThrowResult.ON_FIELD, reasons=None, track=None):
    traj = [(f, final[0] - (end - f) * 5.0, final[1]) for f in range(start, end + 1)]
    return ThrowEvent(
        throw_id=tid, ball_track_id=track or tid, start_frame=start, end_frame=end, start_timestamp=start / 30,
        end_timestamp=end / 30, initial_position=(traj[0][1], traj[0][2]), final_position=final, trajectory=traj,
        final_state=state, confidence=0.9, detection_reasons=[], event_type=etype, final_state_reasons=reasons or [],
        metrics={"size_px": SIZE}, track_id_history=[track or tid],
    )


def obs(frame, x, y, typ=ObjectType.BALL):
    return Observation(frame, x - SIZE / 2, y - SIZE / 2, x + SIZE / 2, y + SIZE / 2, 0.9, typ)


class TestNumbering(unittest.TestCase):
    def test_only_confirmed_throws_in_play_are_numbered_in_chronological_order(self):
        b1 = thrown(1, (400, Y), (30, 0), start=100, decay=0.95)
        b2 = thrown(2, (400, Y + 150), (30, 0), start=300, decay=0.95)
        vanishing = thrown(3, (400, Y - 200), (20, 0), start=500, decay=0.98, last_frame=525)  # perdue en plein terrain
        tm, _, td = play([jack_thrown(), static_ball(9, 1400, Y + 300), b2, b1, vanishing], 800)
        balls = number_balls(td, tm)
        confirmed = [t for t in td.throws if t.event_type == ThrowType.THROW]
        self.assertEqual(len(confirmed), 2)
        self.assertTrue(any(t.event_type == ThrowType.UNKNOWN for t in td.throws))  # il y a bien un incertain
        self.assertEqual([b.number for b in balls], [1, 2])
        self.assertEqual([b.throw.start_frame < balls[-1].throw.start_frame for b in balls[:-1]], [True])
        self.assertEqual({b.throw.throw_id for b in balls}, {t.throw_id for t in confirmed})  # l'incertain n'a pas de n°
        self.assertEqual(len(balls), 2)  # la boule déjà posée (static) et le cochonnet : pas de numéro

    def test_out_of_play_ball_is_not_counted(self):
        a = make_throw(1, 100, 160, (500, 500))
        out = make_throw(2, 200, 260, (2000, 500), state=ThrowResult.OUT_OF_PLAY)
        td = type("TD", (), {"movements": [], "throws": [a, out]})()
        tm = type("TM", (), {"tracks": {}, "resolve": staticmethod(lambda i: i)})()
        self.assertEqual([b.throw.throw_id for b in number_balls(td, tm)], [1])

    def test_position_follows_the_throw_then_stays_put(self):
        b = number_balls(type("TD", (), {"movements": [], "throws": [make_throw(1, 100, 160, (500, 500))]})(),
                         type("TM", (), {"tracks": {}, "resolve": staticmethod(lambda i: i)})())[0]
        self.assertIsNone(b.position_at(99))  # pas de marque avant le lancer
        self.assertLess(b.position_at(120)[0], 500)
        self.assertEqual(b.position_at(160), (500, 500))
        self.assertEqual(b.position_at(5000), (500, 500))  # immobile ensuite


class TestPushedBallKeepsItsNumber(unittest.TestCase):
    def test_numbered_ball_pushed_away_under_a_new_track_is_followed_not_renumbered(self):
        a = thrown(1, (200, Y), (30, 0), start=100, decay=0.95, last_frame=400)
        rest = play([jack_thrown(), a], 330)[2].throws[0].final_position
        c = thrown(3, (rest[0], rest[1] - 45), (0, -25), start=406, decay=0.95)
        tm, _, td = play([jack_thrown(), a, c], 560)
        balls = number_balls(td, tm)
        self.assertEqual([b.number for b in balls], [1])  # pas de boule n°2 : c'est la n°1 qui a bougé
        self.assertAlmostEqual(balls[0].position_at(330)[0], rest[0], delta=40)  # (elle finit de rouler un peu)
        end = balls[0].position_at(559)
        self.assertLess(end[1], rest[1] - 200)  # elle est maintenant là où elle s'est arrêtée
        self.assertAlmostEqual(end[0], rest[0], delta=15)


class TestRelocation(unittest.TestCase):
    def setUp(self):
        self.first = make_throw(1, 100, 160, (850, 500))
        self.lost = make_throw(2, 300, 330, (700, 500), reasons=["lost_in_contact_with_ball_1", "probable_stop_against_ball"])
        self.td = type("TD", (), {"movements": [], "throws": [self.first, self.lost]})()
        self.tm = type("TM", (), {"tracks": {}, "resolve": staticmethod(lambda i: i)})()

    def frames(self, extra):
        fr = []
        for f in range(700):
            row = [obs(f, 850, 500)] if f > 160 else []  # la 1re boule, posée
            row += [obs(f, x, y) for (x, y, f0) in extra if f >= f0]
            fr.append(row)
        return fr

    def test_ball_lost_in_contact_is_relocated_to_the_stable_unclaimed_detection(self):
        balls = number_balls(self.td, self.tm)
        self.assertEqual(balls[1].position_source, "last_seen_in_contact")
        relocate_contact_ends(balls, self.frames([(790, 560, 340)]))  # elle s'est arrêtée un peu plus loin
        self.assertEqual(balls[1].position_source, "relocated_from_detections")
        x, y = balls[1].points[-1]
        self.assertAlmostEqual(x, 790, delta=3)
        self.assertAlmostEqual(y, 560, delta=3)
        self.assertEqual(balls[1].position_at(330), (700, 500))  # avant le re-calage : dernière position vue
        self.assertEqual(balls[0].position_source, "tracked")  # les autres ne bougent pas

    def test_detection_already_claimed_by_another_numbered_ball_is_not_used(self):
        balls = number_balls(self.td, self.tm)
        relocate_contact_ends(balls, self.frames([]))  # seule la boule n°1 est détectée près de là
        self.assertEqual(balls[1].position_source, "last_seen_in_contact")
        self.assertEqual(balls[1].points[-1], (700, 500))

    def test_unstable_detection_is_not_used(self):
        balls = number_balls(self.td, self.tm)
        fr = self.frames([])
        for f in range(345, 350):  # 5 frames seulement : un passage, pas une boule arrêtée
            fr[f].append(obs(f, 790, 560))
        relocate_contact_ends(balls, fr)
        self.assertEqual(balls[1].position_source, "last_seen_in_contact")

    def test_far_detection_is_not_used(self):
        balls = number_balls(self.td, self.tm)
        relocate_contact_ends(balls, self.frames([(1500, 900, 340)]))
        self.assertEqual(balls[1].position_source, "last_seen_in_contact")

    def test_jack_detection_is_not_used(self):
        balls = number_balls(self.td, self.tm)
        fr = self.frames([])
        for f in range(340, 700):
            fr[f].append(obs(f, 790, 560, ObjectType.JACK))
        relocate_contact_ends(balls, fr)
        self.assertEqual(balls[1].position_source, "last_seen_in_contact")


class TestRenderer(unittest.TestCase):
    def test_marks_only_numbered_balls_and_counts_them(self):
        tds = type("TD", (), {"movements": [], "throws": [make_throw(1, 100, 160, (400, 300)), make_throw(2, 200, 260, (700, 300))]})()
        tm = type("TM", (), {"tracks": {}, "resolve": staticmethod(lambda i: i)})()
        balls = number_balls(tds, tm)
        r = BallLabelRenderer(balls, 1.0)
        blank = np.zeros((600, 1000, 3), np.uint8)
        before = r.draw(blank.copy(), 50, 30.0)
        mid = r.draw(blank.copy(), 230, 30.0)
        end = r.draw(blank.copy(), 400, 30.0)

        def marks(img, x0, x1):  # pixels colorés dans la bande (hors compteur en haut)
            return int((img[80:, x0:x1].sum(axis=2) > 0).sum())

        self.assertEqual(marks(before, 0, 1000), 0)  # avant tout lancer : rien
        self.assertGreater(marks(mid, 300, 500), 0)  # n°1 posée
        self.assertGreater(marks(mid, 560, 800), 0)  # n°2 en vol
        self.assertEqual(marks(mid, 800, 1000), 0)  # nulle part ailleurs
        self.assertGreater(marks(end, 300, 500), 0)
        self.assertGreater(marks(end, 620, 800), 0)
        self.assertEqual(marks(end, 0, 300), 0)
        # couleur « en vol » != couleur « arrêtée » au même endroit
        flying = r.draw(blank.copy(), 230, 30.0)[300, 700 - 42:700 + 42].tolist()
        rest = r.draw(blank.copy(), 400, 30.0)[300, 700 - 42:700 + 42].tolist()
        self.assertNotEqual(flying, rest)


if __name__ == "__main__":
    unittest.main()


class TestApproximatePosition(unittest.TestCase):
    def test_ring_is_dashed_only_for_approximate_positions(self):
        tm = type("TM", (), {"tracks": {}, "resolve": staticmethod(lambda i: i)})()
        exact = number_balls(type("TD", (), {"movements": [], "throws": [make_throw(1, 100, 160, (400, 300))]})(), tm)
        contact = number_balls(type("TD", (), {"movements": [], "throws": [
            make_throw(1, 100, 160, (400, 300), reasons=["lost_in_contact_with_ball_9"])]})(), tm)
        blank = np.zeros((600, 1000, 3), np.uint8)
        a = BallLabelRenderer(exact).draw(blank.copy(), 400, 30.0)
        b = BallLabelRenderer(contact).draw(blank.copy(), 400, 30.0)
        self.assertGreater(int((a[80:] > 0).sum()), int((b[80:] > 0).sum()))  # pointillés : moins de pixels colorés
        self.assertGreater(int((b[80:] > 0).sum()), 0)
        live = BallLabelRenderer(contact).draw(blank.copy(), 130, 30.0)  # en vol : anneau plein même en contact
        full = BallLabelRenderer(exact).draw(blank.copy(), 130, 30.0)
        self.assertEqual(int((live[80:] > 0).sum()), int((full[80:] > 0).sum()))
