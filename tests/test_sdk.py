"""Behavior that does not need a camera model or a network call."""

from __future__ import annotations

import json
import unittest

from typesafe_sdk import Choice, Noul, Score

from jev_sees.given_that import build_given_that
from jev_sees.memory import SceneMemory
from jev_sees.questions import compile_questions
from jev_sees.result import Result
from jev_sees.session import Sees
from jev_sees.tracking import TrackBank, box_gap, box_iou


class QuestionTests(unittest.TestCase):
    def test_list_is_choice(self) -> None:
        questions = compile_questions(["yellow", "red"], "What color is the bus?")
        self.assertIsInstance(questions["answer"], Choice)
        self.assertEqual(set(questions["answer"].criteria), {"yellow", "red"})

    def test_yes_no_is_noul(self) -> None:
        questions = compile_questions("yes/no", "Is the bus visible?")
        self.assertIsInstance(questions["answer"], Noul)
        described = compile_questions({"yes": "touching", "no": "apart"}, "Contact?")
        self.assertIsInstance(described["answer"], Noul)
        self.assertEqual(described["answer"].criteria["true"], "touching")

    def test_rubric_is_score(self) -> None:
        questions = compile_questions(("poor", "fair", "good"), "How close?")
        self.assertIsInstance(questions["answer"], Score)
        self.assertEqual(list(questions["answer"].criteria), ["poor", "fair", "good"])

    def test_mapping_is_several_questions(self) -> None:
        questions = compile_questions(
            {"bus_color": ["yellow", "red"], "is_bus": "yes/no"},
            "Look at the picture",
        )
        self.assertIsInstance(questions["bus_color"], Choice)
        self.assertIsInstance(questions["is_bus"], Noul)


class TrackingTests(unittest.TestCase):
    def test_same_box_keeps_id(self) -> None:
        bank = TrackBank("image")
        first = bank.update([{"label": "bus", "bbox_xyxy": [10, 10, 80, 80], "centroid_uv": [45, 45]}])
        second = bank.update([{"label": "bus", "bbox_xyxy": [12, 11, 82, 81], "centroid_uv": [47, 46]}])
        self.assertEqual(first[0]["object_id"], second[0]["object_id"])

    def test_distant_box_is_a_new_id(self) -> None:
        bank = TrackBank("image")
        first = bank.update([{"label": "bus", "bbox_xyxy": [10, 10, 40, 40]}])
        second = bank.update([{"label": "car", "bbox_xyxy": [400, 400, 500, 500]}])
        self.assertNotEqual(first[0]["object_id"], second[0]["object_id"])

    def test_camera_position_keeps_id(self) -> None:
        bank = TrackBank("camera")
        first = bank.update([{"label": "cup", "position_m": [0.1, 0.2, 0.6], "bbox_xyxy": [0, 0, 10, 10]}])
        second = bank.update([{"label": "cup", "position_m": [0.12, 0.21, 0.61], "bbox_xyxy": [1, 1, 11, 11]}])
        self.assertEqual(first[0]["object_id"], second[0]["object_id"])

    def test_overlap_and_gap(self) -> None:
        self.assertGreater(box_iou([0, 0, 10, 10], [5, 5, 15, 15]), 0)
        self.assertEqual(box_gap([0, 0, 10, 10], [12, 0, 20, 10]), 2)


class MemoryTests(unittest.TestCase):
    def test_label_vote_and_stale(self) -> None:
        memory = SceneMemory()
        memory.observe([{"object_id": "object_001", "label": "bus", "confidence": 0.4, "bbox_xyxy": [0, 0, 10, 10]}])
        memory.observe([{"object_id": "object_001", "label": "bus", "confidence": 0.8, "bbox_xyxy": [1, 1, 11, 11]}])
        memory.observe([{"object_id": "object_001", "label": "truck", "confidence": 0.3, "bbox_xyxy": [1, 1, 11, 11]}])
        self.assertEqual(memory.known_objects["object_001"]["label"], "bus")
        for _ in range(3):
            memory.observe([])
        record = memory.known_objects["object_001"]
        self.assertFalse(record["currently_visible"])
        self.assertTrue(record["stale"])
        self.assertLessEqual(len(record["pose_history"]), 20)


class GivenThatTests(unittest.TestCase):
    def test_facts_keep_pixels_and_drop_scores(self) -> None:
        memory = SceneMemory()
        memory.observe(
            [
                {
                    "object_id": "object_001",
                    "label": "bus",
                    "confidence": 0.91,
                    "bbox_xyxy": [10, 20, 100, 80],
                    "centroid_uv": [55, 50],
                    "attributes": {"color": "yellow"},
                    "semantic_scores": {"bus": 0.9, "car": 0.1},
                    "description": "yellow bus",
                },
                {
                    "object_id": "object_002",
                    "label": "person",
                    "confidence": 0.5,
                    "bbox_xyxy": [90, 20, 130, 90],
                    "centroid_uv": [110, 55],
                    "attributes": {"color": "blue"},
                },
            ]
        )
        state = build_given_that("What color is the bus?", memory, modality="rgb", image_size=(640, 480))
        payload = json.dumps(state)
        self.assertNotIn("semantic_scores", payload)
        self.assertNotIn("pose_history", payload)
        visible = state["current_scene"]["visible_objects"]
        self.assertEqual(visible[0]["bbox_xyxy"], [10, 20, 100, 80])
        self.assertTrue(state["relations"])
        self.assertIn("prompt_budget", state)
        self.assertEqual(state["prompt_budget"]["hard_limit_tokens"], 31000)
        self.assertLessEqual(state["prompt_budget"]["serialized_chars"], 24000)

    def test_budget_drops_low_priority_objects(self) -> None:
        memory = SceneMemory()
        observations = []
        for index in range(30):
            observations.append(
                {
                    "object_id": f"object_{index:03d}",
                    "label": "thing",
                    "confidence": 0.5,
                    "bbox_xyxy": [index, index, index + 5, index + 5],
                    "centroid_uv": [index, index],
                    "description": "x" * 4000,
                }
            )
        memory.observe(observations)
        state = build_given_that("count", memory, modality="rgb", image_size=(100, 100))
        self.assertLessEqual(len(state["scene_memory"]["known_objects"]), 16)
        self.assertGreaterEqual(len(state["scene_memory"]["known_objects"]), 4)
        self.assertLessEqual(state["prompt_budget"]["serialized_chars"], 24000)


class SessionTests(unittest.TestCase):
    def test_ask_hides_given_that(self) -> None:
        captured = {}

        def client(state, questions):
            captured["state"] = state
            captured["questions"] = questions

            class Answer:
                choice = "yellow"
                confidence = 0.88
                probabilities = {"yellow": 0.88, "red": 0.12}

            class Response:
                model = "jev-latest"
                usage = None
                answers = {"answer": Answer()}

                def model_dump(self):
                    return {"model": self.model, "answers": {"answer": {"choice": "yellow"}}}

            return Response()

        sees = Sees(client=client, vocabulary=["bus"])
        sees.memory.observe(
            [
                {
                    "object_id": "object_001",
                    "label": "bus",
                    "confidence": 0.9,
                    "bbox_xyxy": [1, 2, 3, 4],
                    "centroid_uv": [2, 3],
                    "attributes": {"color": "yellow"},
                }
            ]
        )
        sees.image_size = (640, 480)
        result = sees.ask("What color is the bus?", ["yellow", "red"])
        self.assertEqual(result.choice, "yellow")
        self.assertAlmostEqual(result.confidence, 0.88)
        self.assertNotIn("given_that", result.model_dump())
        self.assertIn("user_goal", captured["state"])
        self.assertIsInstance(captured["questions"]["answer"], Choice)


if __name__ == "__main__":
    unittest.main()
