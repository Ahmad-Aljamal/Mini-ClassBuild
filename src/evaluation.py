import json
from dataclasses import dataclass
from typing import Callable, List
from unittest.mock import MagicMock
from src.groq_client import GroqService, GroqServiceError
from src.validator import (
    ValidationErrorDetail,
    generate_with_retry,
    validate_course_outline_data,
    validate_lesson_content_data,
    validate_quiz_data,
)


@dataclass
class EvaluationScenarioResult:
    name: str
    description: str
    expected_outcome: str
    actual_outcome: str
    passed: bool
    details: str


def _make_mock_groq_service(responses: List[str]) -> GroqService:
    """Create an offline GroqService backed by a mocked client returning predefined responses."""
    mock_client = MagicMock()
    side_effects = []
    for text in responses:
        msg = MagicMock()
        msg.content = text
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        side_effects.append(resp)
    mock_client.chat.completions.create.side_effect = side_effects
    return GroqService(api_key="offline_test_key", model="openai/gpt-oss-20b", client=mock_client)


def sample_valid_outline_dict(lesson_count: int = 3) -> dict:
    return {
        "title": "Introduction to Applied AI",
        "description": "Foundational course on AI engineering and RAG pipelines.",
        "learning_objectives": [
            "Understand structured prompting and validation",
            "Build retrieval-augmented generation workflows",
        ],
        "lessons": [
            {
                "title": f"Lesson {i}: Core Concepts {i}",
                "learning_objectives": [f"Objective {i}.1", f"Objective {i}.2"],
                "summary": f"Summary for lesson {i}.",
            }
            for i in range(1, lesson_count + 1)
        ],
    }


def sample_valid_lesson_dict(with_citations: bool = False) -> dict:
    citations = (
        [{"chunk_id": "p1_c1", "page_number": 1, "source_filename": "reference.pdf"}]
        if with_citations
        else []
    )
    return {
        "introduction": "Welcome to this structured lesson on AI pipelines.",
        "sections": [
            {
                "heading": "Section 1: Data Validation",
                "explanation": "Pydantic models enforce strict schema validation on LLM outputs.",
                "example": "CourseOutline.model_validate(raw_json)",
                "citations": citations,
            },
            {
                "heading": "Section 2: Retrieval Grounding",
                "explanation": "Semantic search retrieves relevant document chunks using cosine similarity.",
                "example": "semantic_search(query, chunks, embeddings, min_similarity=0.25)",
                "citations": citations,
            },
        ],
        "summary": "Validating schemas and citations prevents hallucinated course data.",
        "exercises": [
            {
                "question": "Why do we validate AI JSON output with Pydantic?",
                "hint": "Think about missing fields and type safety.",
            },
            {
                "question": "How does citation verification work in RAG?",
                "hint": "Compare cited chunk IDs and page numbers against retrieved chunks.",
            },
        ],
        "citations": citations,
    }


def sample_valid_quiz_dict() -> dict:
    return {
        "questions": [
            {
                "question": f"Sample question {i}?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": "Option A",
                "explanation": f"Option A is correct for question {i}.",
            }
            for i in range(1, 6)
        ]
    }


def run_offline_evaluation() -> List[EvaluationScenarioResult]:
    """Execute offline evaluation scenarios using predefined AI responses without real Groq API calls."""
    results: List[EvaluationScenarioResult] = []

    # Scenario 1: Valid Course Outline Generation (Success)
    try:
        service = _make_mock_groq_service([json.dumps(sample_valid_outline_dict(3))])
        outline = generate_with_retry(
            service,
            "Generate 3-lesson outline",
            lambda data: validate_course_outline_data(data, expected_lessons=3),
            max_retries=1,
        )
        passed = len(outline.lessons) == 3 and outline.title == "Introduction to Applied AI"
        results.append(
            EvaluationScenarioResult(
                name="1. Valid Course Outline Generation",
                description="Predefined valid 3-lesson JSON outline passes Pydantic validation.",
                expected_outcome="Success (3 lessons validated)",
                actual_outcome=f"Success ({len(outline.lessons)} lessons validated)",
                passed=passed,
                details="CourseOutline validated on first attempt.",
            )
        )
    except Exception as exc:
        results.append(
            EvaluationScenarioResult(
                name="1. Valid Course Outline Generation",
                description="Predefined valid 3-lesson JSON outline passes Pydantic validation.",
                expected_outcome="Success",
                actual_outcome=f"Failed: {exc}",
                passed=False,
                details=str(exc),
            )
        )

    # Scenario 2: Controlled Retry Recovery on Invalid JSON / Wrong Lesson Count (Success on Retry)
    try:
        bad_outline = sample_valid_outline_dict(2)  # Wrong count: 2 instead of 3
        good_outline = sample_valid_outline_dict(3)
        service = _make_mock_groq_service(
            ["{invalid json", json.dumps(bad_outline), json.dumps(good_outline)]
        )
        outline = generate_with_retry(
            service,
            "Generate 3-lesson outline",
            lambda data: validate_course_outline_data(data, expected_lessons=3),
            max_retries=2,
        )
        passed = len(outline.lessons) == 3
        results.append(
            EvaluationScenarioResult(
                name="2. Controlled Retry Recovery",
                description="First 2 responses fail (malformed JSON, then wrong lesson count); 3rd succeeds.",
                expected_outcome="Recovered on attempt 3",
                actual_outcome="Recovered on attempt 3",
                passed=passed,
                details="Controlled retry mechanism caught invalid JSON & count mismatch and succeeded.",
            )
        )
    except Exception as exc:
        results.append(
            EvaluationScenarioResult(
                name="2. Controlled Retry Recovery",
                description="First 2 responses fail; 3rd succeeds.",
                expected_outcome="Recovered on attempt 3",
                actual_outcome=f"Failed: {exc}",
                passed=False,
                details=str(exc),
            )
        )

    # Scenario 3: Persistent Invalid Schema Rejection (Expected Failure Caught)
    try:
        missing_fields_payload = {"title": "Incomplete Course"}
        service = _make_mock_groq_service(
            [json.dumps(missing_fields_payload), json.dumps(missing_fields_payload)]
        )
        try:
            generate_with_retry(
                service,
                "Generate outline",
                lambda data: validate_course_outline_data(data, expected_lessons=3),
                max_retries=1,
            )
            results.append(
                EvaluationScenarioResult(
                    name="3. Missing Fields Rejection",
                    description="Incomplete course JSON missing description and lessons is rejected.",
                    expected_outcome="Rejected with GroqServiceError",
                    actual_outcome="Unexpectedly accepted",
                    passed=False,
                    details="Validator failed to reject incomplete JSON.",
                )
            )
        except GroqServiceError as exc:
            results.append(
                EvaluationScenarioResult(
                    name="3. Missing Fields Rejection",
                    description="Incomplete course JSON missing description and lessons is rejected.",
                    expected_outcome="Rejected with GroqServiceError",
                    actual_outcome="Rejected as expected",
                    passed=True,
                    details=f"Caught expected validation error: {exc}",
                )
            )
    except Exception as exc:
        results.append(
            EvaluationScenarioResult(
                name="3. Missing Fields Rejection",
                description="Incomplete course JSON is rejected.",
                expected_outcome="Rejected with GroqServiceError",
                actual_outcome=f"Unexpected error: {exc}",
                passed=False,
                details=str(exc),
            )
        )

    # Scenario 4: PDF-Grounded Lesson with Valid Per-Section Citations (Success)
    retrieved_chunks = [
        {
            "chunk_id": "p1_c1",
            "page_number": 1,
            "source_filename": "reference.pdf",
            "text": "Pydantic validates data models and RAG grounds lessons in PDFs.",
        }
    ]
    try:
        service = _make_mock_groq_service([json.dumps(sample_valid_lesson_dict(with_citations=True))])
        lesson = generate_with_retry(
            service,
            "Generate grounded lesson",
            lambda data: validate_lesson_content_data(data, retrieved_chunks=retrieved_chunks),
            max_retries=1,
        )
        passed = len(lesson.citations) == 1 and all(len(s.citations) >= 1 for s in lesson.sections)
        results.append(
            EvaluationScenarioResult(
                name="4. Grounded Lesson Citation Validation (Valid)",
                description="PDF-grounded lesson with valid per-section citations passes verification.",
                expected_outcome="Success (citations verified)",
                actual_outcome="Success (citations verified)",
                passed=passed,
                details="All section and lesson citations matched retrieved PDF chunks.",
            )
        )
    except Exception as exc:
        results.append(
            EvaluationScenarioResult(
                name="4. Grounded Lesson Citation Validation (Valid)",
                description="PDF-grounded lesson with valid per-section citations passes verification.",
                expected_outcome="Success",
                actual_outcome=f"Failed: {exc}",
                passed=False,
                details=str(exc),
            )
        )

    # Scenario 5: Hallucinated PDF Citation Rejection (Expected Failure Caught)
    try:
        bad_lesson = sample_valid_lesson_dict(with_citations=True)
        bad_lesson["sections"][0]["citations"] = [
            {"chunk_id": "p99_c999", "page_number": 99, "source_filename": "ghost.pdf"}
        ]
        try:
            validate_lesson_content_data(bad_lesson, retrieved_chunks=retrieved_chunks)
            results.append(
                EvaluationScenarioResult(
                    name="5. Hallucinated PDF Citation Rejection",
                    description="Lesson citing non-existent chunk p99_c999 is rejected.",
                    expected_outcome="Rejected with ValidationErrorDetail",
                    actual_outcome="Unexpectedly accepted",
                    passed=False,
                    details="Failed to detect hallucinated PDF chunk citation.",
                )
            )
        except ValidationErrorDetail as exc:
            results.append(
                EvaluationScenarioResult(
                    name="5. Hallucinated PDF Citation Rejection",
                    description="Lesson citing non-existent chunk p99_c999 is rejected.",
                    expected_outcome="Rejected with ValidationErrorDetail",
                    actual_outcome="Rejected as expected",
                    passed=True,
                    details=f"Caught invalid citation: {exc}",
                )
            )
    except Exception as exc:
        results.append(
            EvaluationScenarioResult(
                name="5. Hallucinated PDF Citation Rejection",
                description="Lesson citing non-existent chunk p99_c999 is rejected.",
                expected_outcome="Rejected with ValidationErrorDetail",
                actual_outcome=f"Unexpected error: {exc}",
                passed=False,
                details=str(exc),
            )
        )

    # Scenario 6: Quiz Answer Key & Option Integrity Validation (Valid + Invalid Option Caught)
    try:
        valid_quiz = validate_quiz_data(sample_valid_quiz_dict())
        bad_quiz_dict = sample_valid_quiz_dict()
        bad_quiz_dict["questions"][0]["correct_answer"] = "Option Z (Not in options)"
        rejected_invalid_answer = False
        try:
            validate_quiz_data(bad_quiz_dict)
        except ValidationErrorDetail:
            rejected_invalid_answer = True

        passed = len(valid_quiz.questions) == 5 and rejected_invalid_answer
        results.append(
            EvaluationScenarioResult(
                name="6. Quiz Schema & Answer Integrity Check",
                description="Valid 5-question quiz passes; question with answer outside options is rejected.",
                expected_outcome="Valid passes & invalid answer rejected",
                actual_outcome="Valid passed & invalid answer rejected" if passed else "Check failed",
                passed=passed,
                details="Verified 5 questions, 4 options each, and correct_answer membership.",
            )
        )
    except Exception as exc:
        results.append(
            EvaluationScenarioResult(
                name="6. Quiz Schema & Answer Integrity Check",
                description="Valid 5-question quiz passes; invalid answer is rejected.",
                expected_outcome="Valid passes & invalid answer rejected",
                actual_outcome=f"Failed: {exc}",
                passed=False,
                details=str(exc),
            )
        )

    return results
