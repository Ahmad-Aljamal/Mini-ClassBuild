from typing import Callable, List, Optional, Sequence
from src.schemas import Citation, CourseOutline, LessonContent, Quiz


class ValidationErrorDetail(ValueError):
    pass


def validate_course_outline_data(raw: dict, expected_lessons: int) -> CourseOutline:
    """Validate course outline JSON against CourseOutline schema and lesson count."""
    if not isinstance(raw, dict):
        raise ValidationErrorDetail("Course outline payload must be a JSON object.")
    try:
        course = CourseOutline.model_validate(raw)
    except Exception as exc:
        raise ValidationErrorDetail(f"Invalid CourseOutline schema: {exc}") from exc
    if len(course.lessons) != expected_lessons:
        raise ValidationErrorDetail(
            f"AI returned {len(course.lessons)} lessons; expected {expected_lessons}."
        )
    return course


def validate_pdf_citations(
    citations: Sequence[Citation],
    retrieved_chunks: Sequence[dict],
) -> None:
    """Verify that every Citation references an existing retrieved PDF chunk."""
    valid_sources = {
        (c["chunk_id"], int(c["page_number"]), c["source_filename"])
        for c in retrieved_chunks
    }
    for citation in citations:
        citation_tuple = (
            citation.chunk_id,
            int(citation.page_number),
            citation.source_filename,
        )
        if citation_tuple not in valid_sources:
            raise ValidationErrorDetail(
                f"Invalid PDF citation not found in retrieved chunks: {citation_tuple}"
            )


def validate_lesson_content_data(
    raw: dict,
    retrieved_chunks: Optional[Sequence[dict]] = None,
    require_section_citations: bool = True,
) -> LessonContent:
    """Validate lesson JSON against LessonContent schema and PDF grounding rules."""
    if not isinstance(raw, dict):
        raise ValidationErrorDetail("Lesson payload must be a JSON object.")
    try:
        lesson = LessonContent.model_validate(raw)
    except Exception as exc:
        raise ValidationErrorDetail(f"Invalid LessonContent schema: {exc}") from exc

    if len(lesson.sections) < 2 or len(lesson.sections) > 4:
        raise ValidationErrorDetail("Lesson must contain 2-4 sections.")
    if len(lesson.exercises) != 2:
        raise ValidationErrorDetail("Lesson must contain exactly two exercises.")

    if retrieved_chunks:
        if not lesson.citations:
            raise ValidationErrorDetail(
                "PDF-grounded lesson must include at least one valid citation."
            )
        validate_pdf_citations(lesson.citations, retrieved_chunks)
        for sec_idx, section in enumerate(lesson.sections, start=1):
            if require_section_citations and not section.citations:
                raise ValidationErrorDetail(
                    f"PDF-grounded lesson section {sec_idx} ('{section.heading}') is missing per-section citations."
                )
            if section.citations:
                validate_pdf_citations(section.citations, retrieved_chunks)

    return lesson


def validate_quiz_data(raw: dict) -> Quiz:
    """Validate quiz JSON against Quiz schema."""
    if not isinstance(raw, dict):
        raise ValidationErrorDetail("Quiz payload must be a JSON object.")
    try:
        quiz = Quiz.model_validate(raw)
    except Exception as exc:
        raise ValidationErrorDetail(f"Invalid Quiz schema: {exc}") from exc
    if len(quiz.questions) != 5:
        raise ValidationErrorDetail("Quiz must contain exactly five questions.")
    if any(len(q.options) != 4 for q in quiz.questions):
        raise ValidationErrorDetail("Each question must have exactly four options.")
    return quiz


def generate_with_retry(
    groq_service,
    prompt: str,
    validator_fn: Callable[[dict], object],
    max_retries: int = 2,
):
    """Generate JSON using GroqService with controlled retry upon validation failure."""
    return groq_service.generate_json(
        prompt=prompt,
        validator=validator_fn,
        max_retries=max_retries,
    )
