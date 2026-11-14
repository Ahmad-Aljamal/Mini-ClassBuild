import unittest
from src.schemas import CourseOutline, LessonContent, Quiz


class TestSchemas(unittest.TestCase):
    def test_quiz_requires_five_questions(self):
        data = {"questions": []}
        with self.assertRaises(Exception):
            Quiz.model_validate(data)

    def test_course_rejects_too_few_lessons(self):
        data = {"title": "T", "description": "D", "learning_objectives": ["O"], "lessons": []}
        with self.assertRaises(Exception):
            CourseOutline.model_validate(data)

    def test_lesson_syncs_section_citations(self):
        raw = {
            "introduction": "Intro",
            "sections": [
                {
                    "heading": "S1",
                    "explanation": "E1",
                    "example": "Ex1",
                    "citations": [{"chunk_id": "p1_c1", "page_number": 1, "source_filename": "doc.pdf"}],
                },
                {
                    "heading": "S2",
                    "explanation": "E2",
                    "example": "Ex2",
                    "citations": [{"chunk_id": "p1_c1", "page_number": 1, "source_filename": "doc.pdf"}],
                },
            ],
            "summary": "Sum",
            "exercises": [
                {"question": "Q1", "hint": "H1"},
                {"question": "Q2", "hint": "H2"},
            ],
            "citations": [],
        }
        lesson = LessonContent.model_validate(raw)
        self.assertEqual(len(lesson.citations), 1)
        self.assertEqual(lesson.citations[0].chunk_id, "p1_c1")


if __name__ == "__main__":
    unittest.main()
