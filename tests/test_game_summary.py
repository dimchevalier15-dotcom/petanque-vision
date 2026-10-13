import unittest

from app.petanque.game_summary import build_played_ball_lines, distance_to_jack_cm, score_mene
from app.petanque.throws import Owner, ThrowResult


class TestGameSummary(unittest.TestCase):
    def test_distance_scales_with_ball_size_px(self):
        d = distance_to_jack_cm((100.0, 100.0), (200.0, 100.0), 50.0)
        self.assertAlmostEqual(d, 14.1, places=1)

    def test_holder_is_closest_in_play(self):
        rows = [
            {"throw_id": 1, "ball_number": 1, "owner": Owner.PLAYER_A, "final_state": ThrowResult.ON_FIELD,
             "position": (100.0, 100.0), "size_px": 50.0},
            {"throw_id": 2, "ball_number": 2, "owner": Owner.PLAYER_B, "final_state": ThrowResult.ON_FIELD,
             "position": (130.0, 100.0), "size_px": 50.0},
        ]
        lines = build_played_ball_lines(rows, jack=(105.0, 100.0))
        holders = [ln for ln in lines if ln.is_holder]
        self.assertEqual(len(holders), 1)
        self.assertEqual(holders[0].ball_number, 1)

    def test_out_line_in_summary(self):
        rows = [
            {"throw_id": 1, "ball_number": 1, "owner": Owner.PLAYER_A, "final_state": ThrowResult.ON_FIELD,
             "position": (100.0, 100.0), "size_px": 50.0},
            {"throw_id": 3, "ball_number": 3, "owner": Owner.PLAYER_B, "final_state": ThrowResult.OUT_OF_PLAY,
             "position": (110.0, 100.0), "size_px": 50.0},
        ]
        lines = build_played_ball_lines(rows, jack=(105.0, 100.0))
        out_line = next(ln for ln in lines if ln.ball_number == 3)
        self.assertIn("OUT", out_line.format())

    def test_undetected_ball_in_summary_no_impact_on_holder(self):
        rows = [
            {"throw_id": 1, "ball_number": 1, "owner": Owner.PLAYER_A, "final_state": ThrowResult.ON_FIELD,
             "position": (100.0, 100.0), "size_px": 50.0},
            {"throw_id": -1, "ball_number": 2, "owner": Owner.PLAYER_B, "final_state": ThrowResult.UNKNOWN,
             "position": None, "size_px": 50.0, "undetected": True},
            {"throw_id": 2, "ball_number": 3, "owner": Owner.PLAYER_B, "final_state": ThrowResult.ON_FIELD,
             "position": (130.0, 100.0), "size_px": 50.0},
        ]
        lines = build_played_ball_lines(rows, jack=(105.0, 100.0))
        self.assertEqual(lines[1].ball_number, 2)
        self.assertIn("boule 2", lines[1].format())
        self.assertIn("non détectée", lines[1].format())
        self.assertFalse(lines[0].is_tiree)
        self.assertTrue(lines[0].is_holder)

    def test_ffpjp_score_holder_plus_same_team_closer_than_opponent(self):
        rows = [
            {"throw_id": 1, "ball_number": 1, "owner": Owner.PLAYER_A, "final_state": ThrowResult.ON_FIELD,
             "position": (100.0, 100.0), "size_px": 50.0},
            {"throw_id": 2, "ball_number": 2, "owner": Owner.PLAYER_A, "final_state": ThrowResult.ON_FIELD,
             "position": (108.0, 100.0), "size_px": 50.0},
            {"throw_id": 3, "ball_number": 3, "owner": Owner.PLAYER_B, "final_state": ThrowResult.ON_FIELD,
             "position": (130.0, 100.0), "size_px": 50.0},
        ]
        lines = build_played_ball_lines(rows, jack=(105.0, 100.0))
        s = score_mene(lines)
        self.assertEqual((s.points_a, s.points_b), (2, 0))
        self.assertTrue(lines[0].is_scoring and lines[1].is_scoring)
        self.assertFalse(lines[2].is_scoring)

    def test_ffpjp_score_stops_at_first_opponent_ball(self):
        rows = [
            {"throw_id": 1, "ball_number": 1, "owner": Owner.PLAYER_A, "final_state": ThrowResult.ON_FIELD,
             "position": (100.0, 100.0), "size_px": 50.0},
            {"throw_id": 2, "ball_number": 2, "owner": Owner.PLAYER_B, "final_state": ThrowResult.ON_FIELD,
             "position": (110.0, 100.0), "size_px": 50.0},
            {"throw_id": 3, "ball_number": 3, "owner": Owner.PLAYER_A, "final_state": ThrowResult.ON_FIELD,
             "position": (200.0, 100.0), "size_px": 50.0},
        ]
        lines = build_played_ball_lines(rows, jack=(105.0, 100.0))
        s = score_mene(lines)
        self.assertEqual((s.points_a, s.points_b), (1, 0))

    def test_ffpjp_out_and_undetected_not_scored(self):
        rows = [
            {"throw_id": 1, "ball_number": 1, "owner": Owner.PLAYER_A, "final_state": ThrowResult.ON_FIELD,
             "position": (100.0, 100.0), "size_px": 50.0},
            {"throw_id": -1, "ball_number": 2, "owner": Owner.PLAYER_B, "final_state": ThrowResult.UNKNOWN,
             "position": None, "size_px": 50.0, "undetected": True},
            {"throw_id": 3, "ball_number": 3, "owner": Owner.PLAYER_B, "final_state": ThrowResult.OUT_OF_PLAY,
             "position": (105.0, 100.0), "size_px": 50.0},
        ]
        lines = build_played_ball_lines(rows, jack=(105.0, 100.0))
        s = score_mene(lines)
        self.assertEqual((s.points_a, s.points_b), (1, 0))

    def test_opponent_out_marks_previous_holder_tiree(self):
        rows = [
            {"throw_id": 1, "ball_number": 1, "owner": Owner.PLAYER_A, "final_state": ThrowResult.ON_FIELD,
             "position": (100.0, 100.0), "size_px": 50.0},
            {"throw_id": 2, "ball_number": 2, "owner": Owner.PLAYER_B, "final_state": ThrowResult.ON_FIELD,
             "position": (200.0, 100.0), "size_px": 50.0},
            {"throw_id": 3, "ball_number": 3, "owner": Owner.PLAYER_B, "final_state": ThrowResult.OUT_OF_PLAY,
             "position": (300.0, 100.0), "size_px": 50.0},
        ]
        lines = build_played_ball_lines(rows, jack=(105.0, 100.0))
        a = next(ln for ln in lines if ln.ball_number == 1)
        self.assertTrue(a.is_tiree)
        self.assertIn("tirée", a.format())
        self.assertFalse(a.is_holder)


if __name__ == "__main__":
    unittest.main()
