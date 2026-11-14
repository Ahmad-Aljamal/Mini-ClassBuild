#!/usr/bin/env python3
"""Step 12: Final Verification Script for Mini ClassBuild (Stages 1, 2, and 3)."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation import (
    run_offline_evaluation,
    sample_valid_lesson_dict,
    sample_valid_outline_dict,
    sample_valid_quiz_dict,
)
from src.pdf_rag import embed_chunks, process_pdf, semantic_search
from src.persistence import (
    export_course_to_markdown,
    load_course_from_disk,
    restore_course_to_session_state,
    save_course_to_disk,
)
from src.validator import (
    validate_course_outline_data,
    validate_lesson_content_data,
    validate_quiz_data,
)
from tests.test_pdf_rag import FakeEmbeddingModel, make_pdf


def run_check(label: str, fn) -> bool:
    try:
        detail = fn()
        print(f"[PASS] {label} -> {detail}")
        return True
    except Exception as exc:
        print(f"[FAIL] {label} -> {exc}")
        return False


def check_project_files() -> str:
    required = [
        "app.py",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "README.md",
        "scripts/run_evaluation.py",
        "scripts/final_check.py",
        "src/schemas.py",
        "src/validator.py",
        "src/persistence.py",
        "src/evaluation.py",
        "src/pdf_rag.py",
        "src/groq_client.py",
        "src/prompts.py",
    ]
    missing = [p for p in required if not (ROOT_DIR / p).exists()]
    if missing:
        raise AssertionError(f"Missing required files: {missing}")
    env_example = (ROOT_DIR / ".env.example").read_text(encoding="utf-8")
    if "GROQ_API_KEY" not in env_example or "GROQ_MODEL" not in env_example:
        raise AssertionError(".env.example must document GROQ_API_KEY and GROQ_MODEL.")
    return f"All {len(required)} required project files verified."


def check_stage1_generation_and_validation() -> str:
    course = validate_course_outline_data(sample_valid_outline_dict(3), expected_lessons=3)
    lesson = validate_lesson_content_data(sample_valid_lesson_dict(with_citations=False))
    quiz = validate_quiz_data(sample_valid_quiz_dict())
    assert len(course.lessons) == 3
    assert 2 <= len(lesson.sections) <= 4
    assert len(lesson.exercises) == 2
    assert len(quiz.questions) == 5
    return "CourseOutline (3 lessons), LessonContent (2 sections, 2 exercises), and Quiz (5 questions) validated."


def check_stage2_pdf_rag_and_citations() -> str:
    pdf_bytes = make_pdf("Python programming and deep neural networks for educational RAG systems.")
    doc = process_pdf(pdf_bytes, "course_notes.pdf")
    model = FakeEmbeddingModel()
    embeddings = embed_chunks(doc["chunks"], model)
    hits = semantic_search(
        "Python programming",
        doc["chunks"],
        embeddings,
        model=model,
        top_k=3,
        min_similarity=0.3,
    )
    assert len(hits) >= 1, "Expected semantic search hit for matching query."
    unrelated = semantic_search(
        "Cooking recipes",
        doc["chunks"],
        embeddings,
        model=model,
        top_k=3,
        min_similarity=0.5,
    )
    assert unrelated == [], "Expected unrelated query below similarity threshold to be rejected."

    grounded_dict = sample_valid_lesson_dict(with_citations=True)
    for sec in grounded_dict["sections"]:
        sec["citations"] = [
            {
                "chunk_id": hits[0]["chunk_id"],
                "page_number": hits[0]["page_number"],
                "source_filename": hits[0]["source_filename"],
            }
        ]
    grounded_dict["citations"] = sec["citations"]
    grounded_lesson = validate_lesson_content_data(
        grounded_dict,
        retrieved_chunks=hits,
        require_section_citations=True,
    )
    assert len(grounded_lesson.citations) == 1
    return "PDF extraction, embeddings, similarity threshold filtering, and per-section citation checks passed."


def check_stage3_evaluation_scenarios() -> str:
    scenarios = run_offline_evaluation()
    passed = [s for s in scenarios if s.passed]
    if len(passed) != len(scenarios):
        failed_names = [s.name for s in scenarios if not s.passed]
        raise AssertionError(f"Offline evaluation scenarios failed: {failed_names}")
    return f"{len(passed)}/{len(scenarios)} offline evaluation scenarios passed."


def check_stage3_persistence_and_markdown_export() -> str:
    course = validate_course_outline_data(sample_valid_outline_dict(2), expected_lessons=2)
    lesson = validate_lesson_content_data(sample_valid_lesson_dict(with_citations=False))
    quiz = validate_quiz_data(sample_valid_quiz_dict())

    with tempfile.TemporaryDirectory() as tmpdir:
        storage = Path(tmpdir)
        saved = save_course_to_disk(
            course=course,
            lessons={"0": lesson},
            quizzes={"0": quiz},
            storage_dir=storage,
        )
        loaded = load_course_from_disk(saved.course_id, storage_dir=storage)
        state = {}
        restore_course_to_session_state(state, loaded)
        assert state["course"].title == course.title
        md = export_course_to_markdown(loaded.course, loaded.lessons, loaded.quizzes, loaded.course_config)
        assert "## Answer Key" in md and "Lesson 1 Quiz" in md
    return "JSON save/load, session restoration, and Markdown export with Answer Key verified."


def check_unit_test_suite() -> str:
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(ROOT_DIR / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise AssertionError(
            f"Unit tests failed: {len(result.failures)} failure(s), {len(result.errors)} error(s)."
        )
    return f"Ran {result.testsRun} automated unit tests successfully."


def main() -> int:
    print("=" * 76)
    print("MINI CLASSBUILD - STEP 12: FINAL VERIFICATION CHECK")
    print("=" * 76)
    checks = [
        ("Project Structure & Configuration (.env.example)", check_project_files),
        ("Stage 1: Course, Lesson & Quiz Validation", check_stage1_generation_and_validation),
        ("Stage 2: PDF RAG, Similarity Threshold & Citations", check_stage2_pdf_rag_and_citations),
        ("Stage 3 (Step 10): Offline Evaluation Suite", check_stage3_evaluation_scenarios),
        ("Stage 3 (Step 11): Save, Load & Markdown Export", check_stage3_persistence_and_markdown_export),
        ("Stage 3 (Step 12): Full Automated Unit Test Suite", check_unit_test_suite),
    ]

    passed_checks = sum(1 for label, fn in checks if run_check(label, fn))
    total_checks = len(checks)
    print("=" * 76)
    print(f"FINAL RESULT: {passed_checks}/{total_checks} verification checks passed.")
    print("=" * 76)
    return 0 if passed_checks == total_checks else 1


if __name__ == "__main__":
    sys.exit(main())
