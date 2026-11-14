import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.evaluation import (
    _make_mock_groq_service,
    run_offline_evaluation,
    sample_valid_lesson_dict,
    sample_valid_outline_dict,
    sample_valid_quiz_dict,
)
from src.persistence import (
    CoursePersistenceError,
    export_course_to_markdown,
    list_saved_courses,
    load_course_from_disk,
    restore_course_to_session_state,
    save_course_to_disk,
)
from src.prompts import chatbot_prompt
from src.validator import (
    ValidationErrorDetail,
    generate_with_retry,
    validate_course_outline_data,
    validate_lesson_content_data,
    validate_quiz_data,
)


class TestStage3Workflow(unittest.TestCase):
    def test_outline_lesson_quiz_generation_and_retries(self):
        bad_json = "not a json object"
        valid_outline = json.dumps(sample_valid_outline_dict(3))
        groq = _make_mock_groq_service([bad_json, valid_outline])

        course = generate_with_retry(
            groq,
            "Create 3-lesson course",
            lambda raw: validate_course_outline_data(raw, expected_lessons=3),
            max_retries=2,
        )
        self.assertEqual(len(course.lessons), 3)

        retrieved_chunks = [
            {
                "chunk_id": "p1_c1",
                "page_number": 1,
                "source_filename": "reference.pdf",
                "text": "Sample grounded chunk text.",
            }
        ]
        groq_lesson = _make_mock_groq_service([json.dumps(sample_valid_lesson_dict(with_citations=True))])
        lesson = generate_with_retry(
            groq_lesson,
            "Create grounded lesson",
            lambda raw: validate_lesson_content_data(raw, retrieved_chunks=retrieved_chunks),
            max_retries=1,
        )
        self.assertEqual(len(lesson.sections), 2)
        self.assertEqual(len(lesson.exercises), 2)
        self.assertTrue(all(len(s.citations) >= 1 for s in lesson.sections))

        groq_quiz = _make_mock_groq_service([json.dumps(sample_valid_quiz_dict())])
        quiz = generate_with_retry(groq_quiz, "Create quiz", validate_quiz_data, max_retries=1)
        self.assertEqual(len(quiz.questions), 5)

    def test_grounded_lesson_rejects_missing_section_citations(self):
        retrieved_chunks = [
            {
                "chunk_id": "p1_c1",
                "page_number": 1,
                "source_filename": "reference.pdf",
                "text": "Grounded text",
            }
        ]
        lesson_no_sec_cites = sample_valid_lesson_dict(with_citations=False)
        lesson_no_sec_cites["citations"] = [
            {"chunk_id": "p1_c1", "page_number": 1, "source_filename": "reference.pdf"}
        ]
        with self.assertRaises(ValidationErrorDetail):
            validate_lesson_content_data(
                lesson_no_sec_cites,
                retrieved_chunks=retrieved_chunks,
                require_section_citations=True,
            )

    def test_chatbot_conversation_history_and_lesson_isolation(self):
        course = validate_course_outline_data(sample_valid_outline_dict(2), expected_lessons=2)
        lesson_0 = validate_lesson_content_data(sample_valid_lesson_dict(with_citations=False))
        lesson_1 = validate_lesson_content_data(sample_valid_lesson_dict(with_citations=False))

        chat_histories = {}
        key_0 = f"{course.title}::0"
        key_1 = f"{course.title}::1"

        chat_histories.setdefault(key_0, []).append({"role": "user", "content": "Question in Lesson 1"})
        chat_histories[key_0].append({"role": "assistant", "content": "Answer in Lesson 1"})

        chat_histories.setdefault(key_1, []).append({"role": "user", "content": "Question in Lesson 2"})

        self.assertEqual(len(chat_histories[key_0]), 2)
        self.assertEqual(len(chat_histories[key_1]), 1)
        self.assertNotEqual(chat_histories[key_0][0]["content"], chat_histories[key_1][0]["content"])

        prompt_0 = chatbot_prompt(course.title, course.lessons[0].title, lesson_0, [], chat_histories[key_0])
        self.assertIn("Question in Lesson 1", prompt_0)
        self.assertNotIn("Question in Lesson 2", prompt_0)

        # Mock chat completion
        mock_client = MagicMock()
        msg = MagicMock()
        msg.content = "Tutor response referencing Lesson 1."
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        mock_client.chat.completions.create.return_value = resp

        from src.groq_client import GroqService

        service = GroqService(api_key="test", client=mock_client)
        reply = service.chat(prompt_0)
        self.assertEqual(reply, "Tutor response referencing Lesson 1.")

    def test_course_save_load_corrupted_handling_and_markdown_export(self):
        course = validate_course_outline_data(sample_valid_outline_dict(2), expected_lessons=2)
        lesson_0 = validate_lesson_content_data(sample_valid_lesson_dict(with_citations=True), retrieved_chunks=[
            {"chunk_id": "p1_c1", "page_number": 1, "source_filename": "reference.pdf", "text": "x"}
        ])
        quiz_0 = validate_quiz_data(sample_valid_quiz_dict())

        with tempfile.TemporaryDirectory() as tmpdir:
            storage = Path(tmpdir)
            record = save_course_to_disk(
                course=course,
                lessons={"0": lesson_0},
                quizzes={"0": quiz_0},
                course_config={"topic": "Applied AI", "level": "Intermediate", "lesson_count": 2},
                quiz_results={"0": 4},
                storage_dir=storage,
            )
            self.assertTrue((storage / f"{record.course_id}.json").exists())

            # Write a corrupted file to verify graceful handling
            (storage / "corrupted_course.json").write_text("{broken json", encoding="utf-8")

            listed = list_saved_courses(storage_dir=storage)
            self.assertEqual(len(listed), 2)
            valid_items = [i for i in listed if i["valid"]]
            invalid_items = [i for i in listed if not i["valid"]]
            self.assertEqual(len(valid_items), 1)
            self.assertEqual(len(invalid_items), 1)

            loaded = load_course_from_disk(record.course_id, storage_dir=storage)
            self.assertEqual(loaded.course.title, course.title)
            self.assertIn("0", loaded.lessons)
            self.assertIn("0", loaded.quizzes)

            session_state = {}
            restore_course_to_session_state(session_state, loaded)
            self.assertEqual(session_state["course"].title, course.title)
            self.assertEqual(session_state["quiz_results"]["0"], 4)

            # Verify missing and corrupted file errors
            with self.assertRaises(CoursePersistenceError):
                load_course_from_disk("non_existent_id", storage_dir=storage)
            with self.assertRaises(CoursePersistenceError):
                load_course_from_disk("corrupted_course", storage_dir=storage)

            # Verify Markdown export contains lessons, quizzes, and separated Answer Key
            md = export_course_to_markdown(
                course=loaded.course,
                lessons=loaded.lessons,
                quizzes=loaded.quizzes,
                course_config=loaded.course_config,
            )
            self.assertIn(f"# {course.title}", md)
            self.assertIn("Section 1: Data Validation", md)
            self.assertIn("Lesson 1 Quiz", md)
            self.assertIn("## Answer Key", md)
            self.assertIn("Correct Answer:", md)

    def test_offline_evaluation_suite_passes(self):
        results = run_offline_evaluation()
        self.assertGreaterEqual(len(results), 6)
        self.assertTrue(all(r.passed for r in results))


if __name__ == "__main__":
    unittest.main()
