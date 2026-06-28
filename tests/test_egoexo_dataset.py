"""Unit tests for the EgoExo-Fitness dataset stage (pure helpers, isolated from data files)."""
from __future__ import annotations

import unittest

from src.egoexo.dataset import (
    JudgedAction,
    assign_participant_splits,
    build_record_info,
    parse_judged_actions,
    participant_id,
)


def _action(participant: str, sample_id: str = "r_action_1", action_name: str = "X") -> JudgedAction:
    return JudgedAction(
        sample_id=sample_id,
        record_id=sample_id.split("_action_")[0],
        action_index=1,
        action_name=action_name,
        st_frame=0,
        ed_frame=10,
        frame_root="frames_open/r",
        original_actor="0101-" + participant,
        participant=participant,
        views=["ego_l"],
        scores=[3],
        comments=["c"],
        guidance="g",
        criteria={},
    )


class ParticipantIdTests(unittest.TestCase):
    def test_strips_date_prefix(self):
        self.assertEqual(participant_id("0912-Huiwen"), "huiwen")
        self.assertEqual(participant_id("1017-Huiwen"), "huiwen")

    def test_same_name_across_dates_collapses_to_one_participant(self):
        # The core leakage guarantee: a person recorded on several dates is one participant.
        sessions = ["0925-Jianan", "0927-Jianan", "1019-Jianan", "1021-Jianan"]
        self.assertEqual(len({participant_id(s) for s in sessions}), 1)

    def test_distinct_names_stay_distinct(self):
        self.assertNotEqual(participant_id("0912-Huiwen"), participant_id("0912-Shenghao"))

    def test_handles_missing_date_prefix(self):
        self.assertEqual(participant_id("Huiwen"), "huiwen")


class ParseJudgedActionsTests(unittest.TestCase):
    def setUp(self):
        self.record_info = {
            "ABC123": {"original_actor": "0912-Huiwen", "participant": "huiwen", "views": ["ego_l", "exo_m"]}
        }
        self.iaj = {
            "ABC123_action_1": {
                "annotations": [
                    {
                        "key_point_verification": [["Keep your back straight.", "True"], ["Cross your feet.", "False"]],
                        "action_quality_score": 3,
                        "comment": "c1",
                        "action_name": "Push-ups",
                        "action_guidance": "do x",
                        "annotator": "a1",
                    },
                    {
                        "key_point_verification": [["Keep your back straight.", "False"], ["Cross your feet.", "False"]],
                        "action_quality_score": 4,
                        "comment": "c2",
                        "action_name": "Push-ups",
                        "action_guidance": "do x",
                        "annotator": "a2",
                    },
                ],
                "st_ed_frame": [10, 60],
                "frame_root": "frames_open/ABC123",
            }
        }

    def test_parses_key_and_segment(self):
        (action,) = parse_judged_actions(self.iaj, self.record_info)
        self.assertEqual(action.record_id, "ABC123")
        self.assertEqual(action.action_index, 1)
        self.assertEqual((action.st_frame, action.ed_frame), (10, 60))
        self.assertEqual(action.participant, "huiwen")
        self.assertEqual(action.views, ["ego_l", "exo_m"])

    def test_aggregates_multi_annotator_votes(self):
        (action,) = parse_judged_actions(self.iaj, self.record_info)
        self.assertEqual(action.scores, [3, 4])
        self.assertEqual(action.criteria["Keep your back straight."], {"n_true": 1, "n_false": 1})
        self.assertEqual(action.criteria["Cross your feet."], {"n_true": 0, "n_false": 2})

    def test_default_fault_is_strict_majority_false(self):
        (action,) = parse_judged_actions(self.iaj, self.record_info)
        faults = action.faults
        self.assertEqual(faults["Keep your back straight."], 0)  # 1 vs 1 tie -> pass
        self.assertEqual(faults["Cross your feet."], 1)          # 0 vs 2 -> fault


class SplitTests(unittest.TestCase):
    def test_splits_are_participant_disjoint_and_cover_everyone(self):
        actions = []
        for p in range(20):
            name = f"p{p:02d}"
            for k in range(p % 3 + 1):  # varying action counts per participant
                actions.append(_action(name, sample_id=f"{name}r_action_{k}"))
        splits = assign_participant_splits(actions, val_frac=0.2, test_frac=0.2)

        participants = {a.participant for a in actions}
        self.assertEqual(set(splits), participants)  # everyone assigned
        # No participant maps to more than one split (dict keys are unique by construction),
        # and every split label is valid.
        self.assertTrue(set(splits.values()) <= {"train", "val", "test"})
        # Train should hold the largest share.
        loads = {s: 0 for s in ("train", "val", "test")}
        for a in actions:
            loads[splits[a.participant]] += 1
        self.assertEqual(max(loads, key=loads.get), "train")


class RecordInfoTests(unittest.TestCase):
    def test_record_id_resolved_via_record_index(self):
        meta = {
            "records": [
                {"original_actor": "0912-Huiwen", "views": ["ego_l"]},
                {"original_actor": "0913-Jianan", "views": ["exo_m"]},
            ],
            "record_index": {"ThEnUZ": 0, "xYkvB0": 1},
        }
        info = build_record_info(meta)
        self.assertEqual(info["ThEnUZ"]["participant"], "huiwen")
        self.assertEqual(info["xYkvB0"]["participant"], "jianan")
        self.assertEqual(info["ThEnUZ"]["views"], ["ego_l"])


if __name__ == "__main__":
    unittest.main()
