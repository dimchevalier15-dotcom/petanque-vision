"""JackLocator : la place du but par la couleur (jaune clair), suivie image par image, sans jamais deviner."""

import unittest

import cv2
import numpy as np

from app.petanque.jack_locator import JackLocator, find_jack_seed, yellow_blobs

BALL = 50.0


def hsv_bgr(h, s, v):
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


JACK = hsv_bgr(30, 70, 250)  # jaune clair
LEAF = hsv_bgr(9, 110, 230)  # feuille orangée : teinte du sol, pas du jack
STEEL = hsv_bgr(9, 35, 150)  # boule grise
GROUND = hsv_bgr(10, 30, 160)


def scene(jack=None, leaf=None, balls=(), extra_yellow=(), size=(600, 800)):
    img = np.full((size[0], size[1], 3), GROUND, np.uint8)
    for x, y in balls:
        cv2.circle(img, (x, y), 25, STEEL, -1)
    if leaf:
        cv2.circle(img, leaf, 14, LEAF, -1)
    for p in extra_yellow:
        cv2.circle(img, p, 9, JACK, -1)
    if jack:
        cv2.circle(img, jack, 13, JACK, -1)
    return img


class TestBlobs(unittest.TestCase):
    def test_only_the_jack_colour_is_found(self):
        blobs = yellow_blobs(scene(jack=(300, 300), leaf=(500, 200), balls=[(100, 100), (200, 450)]), min_area=75,
                             max_area=1500)
        self.assertEqual(len(blobs), 1)
        self.assertAlmostEqual(blobs[0].x, 300, delta=2)
        self.assertAlmostEqual(blobs[0].y, 300, delta=2)

    def test_tiny_specks_are_ignored(self):
        img = scene()
        img[100, 100] = JACK
        self.assertEqual(yellow_blobs(img, min_area=75, max_area=1500), [])


class TestShapeAndSize(unittest.TestCase):
    def test_yellow_ring_segment_is_not_a_jack(self):
        img = scene(jack=(300, 300))
        cv2.ellipse(img, (600, 400), (150, 150), 0, 200, 217, JACK, 12)  # arc du rond de lancer
        both = yellow_blobs(img, min_area=75, max_area=1500)
        self.assertEqual(len(both), 2)
        self.assertEqual(len(yellow_blobs(img, min_area=75, max_area=1500, round_only=True)), 1)

    def test_seed_ignores_large_yellow_objects_even_if_round(self):
        def big(_):
            img = scene()
            cv2.circle(img, (400, 300), 40, JACK, -1)  # 4 fois trop gros pour un but
            return img

        pos, _, info = find_jack_seed([big(i) for i in range(8)], [], BALL)
        self.assertIsNone(pos)
        self.assertEqual(info["reason"], "no_persistent_yellow_blob")

    def test_size_limits_follow_the_ball_size(self):
        from app.petanque.jack_locator import area_limits

        small, large = area_limits(30.0), area_limits(60.0)
        self.assertLess(small[1], large[1])
        self.assertLess(small[0], large[0])


class TestSeed(unittest.TestCase):
    def frames(self, n=8, **kw):
        return [scene(**kw) for _ in range(n)]

    def test_persistent_yellow_blob_is_the_seed(self):
        pos, distractors, info = find_jack_seed(self.frames(jack=(300, 300)), [], BALL)
        self.assertAlmostEqual(pos[0], 300, delta=3)
        self.assertEqual(distractors, [])
        self.assertGreaterEqual(info["persistence"], 0.9)

    def test_yolo_hint_breaks_the_tie_with_a_fixed_yellow_patch(self):
        frames = self.frames(jack=(300, 300), extra_yellow=[(360, 280)])
        pos, distractors, info = find_jack_seed(frames, [(302, 301)], BALL)
        self.assertLess(abs(pos[0] - 300), 5)
        self.assertEqual(len(distractors), 1)
        self.assertLess(abs(distractors[0][0] - 360), 5)
        self.assertTrue(info["hinted_by_yolo"])
        self.assertNotIn("ambiguous", info)

    def test_distractors_far_from_the_jack_are_dropped(self):
        frames = self.frames(jack=(300, 300), extra_yellow=[(360, 280), (700, 500)])
        _, distractors, _ = find_jack_seed(frames, [(300, 300)], BALL)
        self.assertEqual(len(distractors), 1)  # (700, 500) est à plus de 6 diamètres : sans risque de confusion

    def test_two_equal_candidates_without_hint_are_flagged_ambiguous(self):
        frames = self.frames(jack=(300, 300), extra_yellow=[(360, 280)])
        pos, _, info = find_jack_seed(frames, [], BALL)
        self.assertIsNotNone(pos)
        self.assertTrue(info.get("ambiguous"))

    def test_yolo_hint_on_a_non_yellow_object_is_not_enough(self):
        pos, _, info = find_jack_seed(self.frames(leaf=(300, 300)), [(300, 300)], BALL)
        self.assertIsNone(pos)  # un « cochonnet » YOLO qui n'est pas jaune (feuille) n'est pas le but
        self.assertEqual(info["reason"], "no_persistent_yellow_blob")

    def test_passing_yellow_object_is_not_persistent(self):
        frames = [scene(jack=(100 + 40 * i, 300)) for i in range(8)]  # traverse l'image
        pos, _, _ = find_jack_seed(frames, [], BALL)
        self.assertIsNone(pos)

    def test_inside_filter_excludes_outside_the_field(self):
        pos, _, _ = find_jack_seed(self.frames(jack=(300, 300)), [], BALL, inside=lambda p: p[0] > 500)
        self.assertIsNone(pos)


class TestLocator(unittest.TestCase):
    def test_follows_a_jack_pushed_by_a_ball_and_reports_the_move(self):
        loc = JackLocator((300, 300), BALL)
        for f in range(30):
            fix = loc.update(f, scene(jack=(300, 300)))
        self.assertEqual(fix.status, "seen")
        self.assertEqual(loc.moves, [])
        for f in range(30, 40):  # poussé de 45 px en quelques images
            loc.update(f, scene(jack=(300 + min(45, (f - 30) * 9), 300)))
        for f in range(40, 80):
            fix = loc.update(f, scene(jack=(345, 300)))
        self.assertAlmostEqual(fix.pos[0], 345, delta=3)
        self.assertEqual(len(loc.moves), 1)
        self.assertAlmostEqual(loc.moves[0]["shift_diam"], 0.9, delta=0.2)
        self.assertAlmostEqual(loc.summary()["final_position"][0], 345, delta=3)

    def test_short_glitch_is_not_a_move(self):
        loc = JackLocator((300, 300), BALL)
        for f in range(60):
            loc.update(f, scene(jack=(340, 300) if 20 <= f < 24 else (300, 300)))
        self.assertEqual(loc.moves, [])

    def test_occluded_jack_is_held_not_moved(self):
        loc = JackLocator((300, 300), BALL)
        for f in range(10):
            loc.update(f, scene(jack=(300, 300)))
        for f in range(10, 40):  # caché par une boule
            fix = loc.update(f, scene(balls=[(300, 300)]))
        self.assertEqual(fix.status, "held")
        self.assertEqual(fix.pos, loc.fixes[9].pos)
        fix = loc.update(40, scene(jack=(300, 300)))
        self.assertEqual(fix.status, "seen")

    def test_never_jumps_to_a_far_yellow_object(self):
        loc = JackLocator((300, 300), BALL)
        for f in range(10):
            loc.update(f, scene(jack=(300, 300)))
        for f in range(10, 40):  # le but disparaît, un autre objet jaune apparaît loin
            fix = loc.update(f, scene(extra_yellow=[(650, 500)]))
        self.assertEqual(fix.status, "held")
        self.assertLess(abs(fix.pos[0] - 300), 3)

    def test_fixed_yellow_distractor_next_to_the_jack_is_ignored(self):
        loc = JackLocator((300, 300), BALL, distractors=[(330, 300)])
        for f in range(30):
            fix = loc.update(f, scene(jack=(300, 300), extra_yellow=[(330, 300)]))
        self.assertEqual(fix.status, "seen")
        self.assertAlmostEqual(fix.pos[0], 300, delta=3)
        for f in range(30, 60):  # le vrai but disparaît : on ne bascule pas sur la tache fixe
            fix = loc.update(f, scene(extra_yellow=[(330, 300)]))
        self.assertEqual(fix.status, "held")
        self.assertAlmostEqual(fix.pos[0], 300, delta=3)


class TestDriftResistance(unittest.TestCase):
    """Réel (180525) : le joueur porte du jaune à côté du but ; la marque ne doit jamais le suivre."""

    def test_wandering_yellow_thing_never_drags_the_mark_when_the_jack_is_hidden(self):
        loc = JackLocator((300, 300), BALL)
        for f in range(10):
            loc.update(f, scene(jack=(300, 300)))
        for f in range(10, 120):  # le but est caché ; un objet jaune rond se promène à 12 px/image
            fix = loc.update(f, scene(extra_yellow=[(330 + 12 * ((f - 10) % 15), 330 - 6 * ((f - 10) % 15))]))
        self.assertEqual(fix.status, "held")
        self.assertEqual(loc.ref, (300, 300))
        self.assertEqual(loc.moves, [])

    def test_a_second_stable_yellow_blob_is_not_a_move_while_the_jack_is_still_there(self):
        loc = JackLocator((300, 300), BALL)
        for f in range(60):
            fix = loc.update(f, scene(jack=(300, 300), extra_yellow=[(400, 330)]))
        self.assertEqual(fix.status, "seen")
        self.assertEqual(loc.moves, [])
        self.assertEqual(loc.ref, (300, 300))

    def test_jack_that_reappears_elsewhere_and_stays_is_a_move_after_the_wait(self):
        loc = JackLocator((300, 300), BALL)
        for f in range(10):
            loc.update(f, scene(jack=(300, 300)))
        statuses = [loc.update(f, scene(jack=(380, 340))).status for f in range(10, 30)]
        self.assertEqual(statuses[:9], ["held"] * 9)  # pas encore confirmé : position approchée
        self.assertEqual(statuses[-1], "seen")
        self.assertEqual(len(loc.moves), 1)
        self.assertEqual(loc.moves[0]["frame"], 10)

    def test_elongated_yellow_shape_is_never_the_jack(self):
        def piping(f):
            img = scene()
            cv2.rectangle(img, (310, 290 + (f % 5)), (360, 304 + (f % 5)), JACK, -1)  # liseré de vêtement
            return img

        loc = JackLocator((300, 300), BALL)
        for f in range(60):
            fix = loc.update(f, piping(f))
        self.assertEqual(fix.status, "held")
        self.assertEqual(loc.ref, (300, 300))


if __name__ == "__main__":
    unittest.main()
