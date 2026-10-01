"""Annotation manuelle, export JSON/CSV, évaluation des erreurs, clips."""

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from app.petanque.annotations import AnnotationSession, load_document, merge_previous, run_cli
from app.petanque.config import PipelineConfig
from app.petanque.throw_eval import CATEGORIES, expected_from_annotations, format_scores, score_throws
from app.petanque.throw_export import format_summary, summary, throw_card, throws_csv, throws_document, throws_report, tracks_document
from app.petanque.throws import Owner, OwnerSource, ThrowEvent, ThrowResult, ThrowType
from tests.test_throws import Y, hit_paths, jack_thrown, play, thrown
from tests.synthetic import SimObject


def make_throw(tid, start, end, ball=10, result=ThrowResult.ON_FIELD, etype=ThrowType.THROW, pos=(900.0, 600.0), **kw):
    return ThrowEvent(
        throw_id=tid, ball_track_id=ball, start_frame=start, end_frame=end, start_timestamp=start / 30.0,
        end_timestamp=end / 30.0, initial_position=(300.0, 600.0), final_position=pos,
        trajectory=[(start, 300.0, 600.0), (end, pos[0], pos[1])], final_state=result, confidence=0.9,
        detection_reasons=["test"], event_type=etype, metrics={"size_px": 60.0, "travel_diam": 10.0}, **kw)


def make_doc(throws, **extra):
    return {"match_id": "match_001", "mene_id": 1, "video": "x.mp4", "fps": 30.0, "frame_offset": 0,
            "throws": [t.to_dict() for t in throws], "collisions": [], **extra}


class TestAnnotationSession(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "annotations.json"
        self.throws = [make_throw(1, 100, 190, ball=17), make_throw(2, 400, 500, ball=23), make_throw(3, 700, 800, ball=17)]
        self.doc = make_doc(self.throws)

    def test_assignment_is_manual_and_saved_with_full_throw_data(self):
        s = AnnotationSession(self.doc, self.path)
        self.assertTrue(all(t.owner == Owner.UNASSIGNED for t in s.throws))  # avant annotation
        s.assign(0, Owner.PLAYER_A)
        s.assign(1, Owner.PLAYER_B)
        s.assign(2, Owner.UNKNOWN)
        s.save()
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["match_id"], "match_001")
        self.assertEqual(saved["mene_id"], 1)
        first = saved["throws"][0]
        for key in ("throw_id", "ball_track_id", "start_frame", "end_frame", "owner", "final_state", "trajectory",
                    "initial_position", "final_position", "confidence", "detection_reasons", "owner_source", "owner_confidence"):
            self.assertIn(key, first)
        self.assertEqual([t["owner"] for t in saved["throws"]], ["PLAYER_A", "PLAYER_B", "UNKNOWN"])
        self.assertEqual(first["owner_source"], OwnerSource.MANUAL.value)
        self.assertEqual(first["owner_confidence"], 1.0)
        self.assertIsNone(saved["throws"][2]["owner_confidence"])  # UNKNOWN humain : pas de confiance à porter
        self.assertEqual(saved["throws"][0]["ball_track_id"], 17)

    def test_only_a_b_unknown_can_be_assigned(self):
        s = AnnotationSession(self.doc, self.path)
        with self.assertRaises(ValueError):
            s.assign(0, Owner.UNASSIGNED)

    def test_not_a_throw_is_recorded_and_clear_undoes_everything(self):
        s = AnnotationSession(self.doc, self.path)
        s.assign(0, Owner.PLAYER_A)
        s.mark_not_throw(0)
        self.assertEqual(s.throws[0].owner, Owner.UNASSIGNED)
        self.assertIs(s.throws[0].human_is_throw, False)
        self.assertTrue(s.is_done(0))
        s.clear(0)
        self.assertFalse(s.is_done(0))
        self.assertIsNone(s.throws[0].human_is_throw)

    def test_reload_keeps_annotations(self):
        s = AnnotationSession(self.doc, self.path)
        s.assign(1, Owner.PLAYER_B)
        s.save()
        again = AnnotationSession(self.doc, self.path)
        self.assertEqual(again.throws[1].owner, Owner.PLAYER_B)
        self.assertEqual(again.carried, 1)
        self.assertEqual(again.progress(), (1, 3))
        self.assertEqual(again.pending(), [0, 2])

    def test_rerunning_the_detector_does_not_lose_annotations(self):
        s = AnnotationSession(self.doc, self.path)
        s.assign(0, Owner.PLAYER_A)
        s.assign(1, Owner.PLAYER_B)
        s.save()
        # nouvelle analyse : frames légèrement décalées, un lancer disparu, un nouveau apparu
        new = [make_throw(1, 105, 196, ball=17), make_throw(2, 1200, 1300, ball=40), make_throw(3, 702, 801, ball=17)]
        again = AnnotationSession(make_doc(new), self.path)
        self.assertEqual(again.throws[0].owner, Owner.PLAYER_A)  # même boule, début proche
        self.assertEqual(again.throws[1].owner, Owner.UNASSIGNED)  # nouveau lancer : pas d'héritage abusif
        self.assertEqual(len(again.orphans), 1)  # l'annotation du lancer disparu n'est pas perdue
        self.assertEqual(again.orphans[0]["owner"], "PLAYER_B")
        again.save()
        self.assertEqual(len(json.loads(self.path.read_text())["orphan_annotations"]), 1)

    def test_merge_does_not_match_two_throws_to_one_annotation(self):
        prev = [dict(make_throw(1, 100, 190).to_dict(), owner="PLAYER_A")]
        a, b = make_throw(1, 100, 190), make_throw(2, 104, 194)
        matched, orphans = merge_previous([a, b], prev)
        self.assertEqual((matched, orphans), (1, []))
        self.assertEqual([t.owner for t in (a, b)].count(Owner.PLAYER_A), 1)

    def test_missed_throws_are_kept_for_ground_truth(self):
        s = AnnotationSession(self.doc, self.path)
        s.add_missed(1000, 1100, Owner.PLAYER_A, "OUT_OF_PLAY")
        s.save()
        self.assertEqual(AnnotationSession(self.doc, self.path).missed[0]["start_frame"], 1000)

    def test_save_is_atomic_no_temp_file_left(self):
        AnnotationSession(self.doc, self.path).save()
        self.assertEqual([p.name for p in self.path.parent.iterdir()], ["annotations.json"])


class TestCli(unittest.TestCase):
    def run_session(self, answers, only_pending=False, clip_fn=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "annotations.json"
        throws = [make_throw(i, 100 * i, 100 * i + 50, ball=i) for i in range(1, 5)]
        session = AnnotationSession(make_doc(throws), path)
        it = iter(answers)
        out: list[str] = []
        run_cli(session, clip_fn=clip_fn, input_fn=lambda _prompt: next(it), print_fn=out.append, only_pending=only_pending)
        return session, path, "\n".join(out)

    def test_annotate_a_b_a_b(self):
        session, path, _ = self.run_session(["a", "b", "a", "b"])
        self.assertEqual([t.owner.value for t in session.throws], ["PLAYER_A", "PLAYER_B", "PLAYER_A", "PLAYER_B"])
        self.assertEqual(len(json.loads(path.read_text())["throws"]), 4)

    def test_keys_unknown_skip_not_a_throw_and_quit(self):
        session, path, _ = self.run_session(["u", "", "x", "q"])
        self.assertEqual([t.owner for t in session.throws], [Owner.UNKNOWN, Owner.UNASSIGNED, Owner.UNASSIGNED, Owner.UNASSIGNED])
        self.assertIs(session.throws[2].human_is_throw, False)
        self.assertTrue(path.exists())  # sauvegardé à la sortie

    def test_previous_and_clear_and_unknown_command(self):
        session, _, out = self.run_session(["a", "p", "c", "zzz", "b", "q"])
        self.assertEqual(session.throws[0].owner, Owner.PLAYER_B)  # « a » puis retour, effacé, puis « b »
        self.assertIn("Commande inconnue", out)

    def test_view_clip_uses_the_callback(self):
        calls = []
        _, _, out = self.run_session(["v", "q"], clip_fn=lambda t: calls.append(t.throw_id) or "/tmp/clip.mp4")
        self.assertEqual(calls, [1])
        self.assertIn("/tmp/clip.mp4", out)

    def test_view_without_video_explains(self):
        _, _, out = self.run_session(["v", "q"])
        self.assertIn("--video", out)

    def test_only_pending_skips_done_throws(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "a.json"
        session = AnnotationSession(make_doc([make_throw(i, 100 * i, 100 * i + 50) for i in range(1, 4)]), path)
        session.assign(0, Owner.PLAYER_A)
        answers = iter(["b", "b"])
        run_cli(session, input_fn=lambda _p: next(answers), print_fn=lambda _s: None, only_pending=True)
        self.assertEqual([t.owner.value for t in session.throws], ["PLAYER_A", "PLAYER_B", "PLAYER_B"])

    def test_add_missed_throw_from_cli(self):
        session, _, _ = self.run_session(["m", "1000", "1100", "a", "q"])
        self.assertEqual(session.missed[0]["owner"], "PLAYER_A")


class TestExports(unittest.TestCase):
    def setUp(self):
        self.tm, self.gs, self.td = play([jack_thrown(), thrown(1, (400, Y), (30, 0), start=100)], 330)

    def test_card_has_the_requested_fields(self):
        card = throw_card(self.td.throws[0])
        for needle in ("THROW #1", "Ball:", "Time: 00:0", "Duration:", "Result: ON_FIELD", "Confidence:", "Owner: UNASSIGNED"):
            self.assertIn(needle, card)
        self.assertTrue(throws_report(self.td.throws).startswith("MÈNE 1"))

    def test_csv_and_summary(self):
        rows = list(csv.DictReader(io.StringIO(throws_csv(self.td.throws))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["owner"], "UNASSIGNED")
        s = summary(self.td, self.tm, 330)
        self.assertEqual((s["throw_events"], s["throws"], s["throws_high_confidence"], s["throws_ambiguous"]), (1, 1, 1, 0))
        self.assertIn("ThrowEvents détectés: 1", format_summary(s))

    def test_documents_are_json_serialisable_and_complete(self):
        doc = throws_document(self.td, PipelineConfig(), {"video": "v.mp4", "fps": 30.0, "frame_count": 330, "start_frame": 0}, "m1")
        again = json.loads(json.dumps(doc))
        self.assertEqual(again["match_id"], "m1")
        self.assertEqual(again["mene_id"], 1)
        self.assertEqual(len(again["throws"]), 1)
        self.assertTrue(again["jack_movements"])  # le cochonnet est tracé à part
        self.assertIn("throws", again["config"])
        tracks = json.loads(json.dumps(tracks_document(self.tm, 30.0)))
        self.assertTrue(all(t["trajectory"] for t in tracks["tracks"]))
        self.assertEqual(ThrowEvent.from_dict(again["throws"][0]).ball_track_id, self.td.throws[0].ball_track_id)


class TestEvaluation(unittest.TestCase):
    def errors(self, throws, expected, **kw):
        return score_throws(throws, expected, **kw)["errors"]

    def test_perfect_detection(self):
        t = make_throw(1, 100, 200)
        s = score_throws([t], {"throws": [{"start_frame": 105, "end_frame": 195, "final_state": "ON_FIELD", "final_position": [910, 600]}]})
        self.assertEqual(s["errors"], {c: 0 for c in CATEGORIES})
        self.assertEqual((s["precision"], s["recall"], s["fully_correct"]), (1.0, 1.0, 1))

    def test_missed_and_occlusion_flagged_missed(self):
        e = self.errors([], {"throws": [{"start_frame": 100, "end_frame": 200}, {"start_frame": 500, "end_frame": 600, "occluded": True}]})
        self.assertEqual((e["MISSED_THROW"], e["OCCLUSION_FAILURE"]), (1, 1))

    def test_false_throw_vs_collision_as_throw(self):
        pushed = make_throw(2, 400, 480)
        spurious = make_throw(1, 100, 180)
        truth = {"throws": [], "not_throws": [{"start_frame": 400, "end_frame": 480, "cause": "collision"}]}
        e = self.errors([spurious, pushed], truth)
        self.assertEqual((e["FALSE_THROW"], e["COLLISION_AS_THROW"]), (1, 1))
        hit = make_throw(3, 700, 780, collisions_as_target=[{"collision_id": 1, "frame": 702, "source_ball": 5, "target_ball": 10, "confidence": 0.6}])
        self.assertEqual(self.errors([hit], {"throws": []})["COLLISION_AS_THROW"], 1)  # le système avait l'indice de collision

    def test_wrong_ball_and_out_of_play_misclassification(self):
        e = self.errors([make_throw(1, 100, 200, pos=(1500.0, 300.0))],
                        {"throws": [{"start_frame": 100, "end_frame": 200, "final_position": [900, 600]}]})
        self.assertEqual(e["WRONG_BALL"], 1)
        e = self.errors([make_throw(1, 100, 200, result=ThrowResult.ON_FIELD)],
                        {"throws": [{"start_frame": 100, "end_frame": 200, "final_state": "OUT_OF_PLAY"}]})
        self.assertEqual(e["OUT_OF_PLAY_MISCLASSIFICATION"], 1)

    def test_unknown_result_is_an_abstention_not_an_error(self):
        s = score_throws([make_throw(1, 100, 200, result=ThrowResult.UNKNOWN)],
                         {"throws": [{"start_frame": 100, "end_frame": 200, "final_state": "OUT_OF_PLAY"}]})
        self.assertEqual(s["errors"]["OUT_OF_PLAY_MISCLASSIFICATION"], 0)
        self.assertEqual(s["result_abstained"], 1)

    def test_fragmented_throw_is_an_occlusion_failure(self):
        s = score_throws([make_throw(1, 100, 150), make_throw(2, 152, 200, ball=11)], {"throws": [{"start_frame": 100, "end_frame": 200}]})
        self.assertEqual(s["errors"]["OCCLUSION_FAILURE"], 1)
        self.assertEqual(s["matched"], 1)

    def test_unknown_type_events_are_abstentions_never_false_throws(self):
        unsure = make_throw(1, 100, 200, etype=ThrowType.UNKNOWN)
        s = score_throws([unsure], {"throws": [], "not_throws": [{"start_frame": 100, "end_frame": 200, "cause": "pickup"}]})
        self.assertEqual(s["errors"]["FALSE_THROW"], 0)
        self.assertTrue(s["abstentions"][0]["matches_not_throw"])
        s = score_throws([unsure], {"throws": [{"start_frame": 100, "end_frame": 200}]})
        self.assertEqual(s["errors"]["MISSED_THROW"], 0)  # abstention, pas un oubli silencieux
        self.assertTrue(s["abstentions"][0]["matches_real_throw"])
        self.assertIn("abstention", format_scores(s))

    def test_truth_derived_from_annotations(self):
        a, b, c = make_throw(1, 100, 200), make_throw(2, 300, 400), make_throw(3, 500, 600)
        doc = make_doc([a, b, c], annotations_updated_at="now", missed_throws=[{"start_frame": 900, "end_frame": 1000}])
        doc["throws"][0]["owner"] = "PLAYER_A"
        doc["throws"][1]["human_is_throw"] = False  # l'humain dit : ce n'était pas un lancer
        truth = expected_from_annotations(doc)  # le 3e n'est pas annoté : ni vrai ni faux
        self.assertEqual([t["start_frame"] for t in truth["throws"]], [100, 900])
        self.assertEqual([t["start_frame"] for t in truth["not_throws"]], [300])


class TestEndToEnd(unittest.TestCase):
    def test_collision_scenario_to_annotation_file(self):
        a_path, b_path, _ = hit_paths(150, (200, Y), (30, 0), (920, Y), 60, a_after=(0, 0), b_after=(24, 0))
        tm, _, td = play([jack_thrown(), SimObject(1, a_path, first_frame=150), SimObject(2, b_path)], 420)
        doc = throws_document(td, PipelineConfig(), {"fps": 30.0, "frame_count": 420}, "sim")
        with tempfile.TemporaryDirectory() as tmp:
            s = AnnotationSession(doc, Path(tmp) / "annotations.json")
            s.assign(0, Owner.PLAYER_B)
            s.save()
            saved = load_document(Path(tmp) / "annotations.json")
        self.assertEqual(len(saved["throws"]), 1)  # la boule poussée n'a pas pollué la liste à annoter
        self.assertEqual(saved["throws"][0]["owner"], "PLAYER_B")
        self.assertEqual(len(saved["collisions"]), 1)
        self.assertTrue(any(m["classification"] == "DISPLACED" for m in saved["dismissed_movements"]))


if __name__ == "__main__":
    unittest.main()
