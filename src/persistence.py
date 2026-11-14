import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from src.schemas import CourseOutline, LessonContent, Quiz, SavedCourseRecord

DEFAULT_STORAGE_DIR = Path("saved_courses")


class CoursePersistenceError(ValueError):
    pass


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip().lower()).strip("-")
    return slug[:40] or "course"


def generate_course_id(title: str) -> str:
    short_uid = uuid.uuid4().hex[:8]
    return f"{_slugify(title)}-{short_uid}"


def save_course_to_disk(
    course: CourseOutline,
    lessons: Optional[Dict[str, LessonContent]] = None,
    quizzes: Optional[Dict[str, Quiz]] = None,
    course_config: Optional[Dict[str, object]] = None,
    quiz_answers: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
    quiz_results: Optional[Dict[str, int]] = None,
    chat_histories: Optional[Dict[str, List[Dict[str, str]]]] = None,
    course_id: Optional[str] = None,
    storage_dir: Path = DEFAULT_STORAGE_DIR,
) -> SavedCourseRecord:
    """Persist a course and its generated lessons/quizzes locally as a validated JSON file."""
    try:
        storage_path = Path(storage_dir)
        storage_path.mkdir(parents=True, exist_ok=True)
        assigned_id = (course_id or generate_course_id(course.title)).strip()
        timestamp = datetime.now(timezone.utc).isoformat()

        record = SavedCourseRecord(
            course_id=assigned_id,
            saved_at=timestamp,
            course_config=course_config or {
                "topic": course.title,
                "level": "Intermediate",
                "lesson_count": len(course.lessons),
            },
            course=course,
            lessons=lessons or {},
            quizzes=quizzes or {},
            quiz_answers=quiz_answers or {},
            quiz_results=quiz_results or {},
            chat_histories=chat_histories or {},
        )

        file_path = storage_path / f"{assigned_id}.json"
        file_path.write_text(
            json.dumps(record.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return record
    except Exception as exc:
        raise CoursePersistenceError(f"Failed to save course: {exc}") from exc


def list_saved_courses(storage_dir: Path = DEFAULT_STORAGE_DIR) -> List[Dict[str, object]]:
    """List available saved courses in storage_dir, flagging any corrupted files gracefully."""
    storage_path = Path(storage_dir)
    if not storage_path.exists():
        return []

    items: List[Dict[str, object]] = []
    for file_path in sorted(storage_path.glob("*.json")):
        try:
            raw = json.loads(file_path.read_text(encoding="utf-8"))
            record = SavedCourseRecord.model_validate(raw)
            items.append(
                {
                    "course_id": record.course_id,
                    "title": record.course.title,
                    "saved_at": record.saved_at,
                    "lesson_count": len(record.course.lessons),
                    "generated_lessons": len(record.lessons),
                    "generated_quizzes": len(record.quizzes),
                    "file_path": str(file_path),
                    "valid": True,
                    "error": None,
                }
            )
        except Exception as exc:
            items.append(
                {
                    "course_id": file_path.stem,
                    "title": f"[Invalid/Corrupted] {file_path.name}",
                    "saved_at": "",
                    "lesson_count": 0,
                    "generated_lessons": 0,
                    "generated_quizzes": 0,
                    "file_path": str(file_path),
                    "valid": False,
                    "error": str(exc),
                }
            )
    return items


def load_course_from_disk(
    course_id_or_path: str,
    storage_dir: Path = DEFAULT_STORAGE_DIR,
) -> SavedCourseRecord:
    """Load and validate a saved course JSON file from disk."""
    candidate = Path(course_id_or_path)
    if not candidate.exists():
        candidate = Path(storage_dir) / f"{course_id_or_path}.json"
    if not candidate.exists():
        raise CoursePersistenceError(f"Saved course file not found: {course_id_or_path}")

    try:
        content = candidate.read_text(encoding="utf-8")
        if not content.strip():
            raise CoursePersistenceError("Saved course JSON file is empty.")
        raw = json.loads(content)
    except CoursePersistenceError:
        raise
    except json.JSONDecodeError as exc:
        raise CoursePersistenceError(
            f"Saved course file contains corrupted JSON: {exc}"
        ) from exc
    except Exception as exc:
        raise CoursePersistenceError(f"Could not read saved course file: {exc}") from exc

    try:
        return SavedCourseRecord.model_validate(raw)
    except Exception as exc:
        raise CoursePersistenceError(
            f"Saved course data is incompatible or failed schema validation: {exc}"
        ) from exc


def restore_course_to_session_state(session_state, record: SavedCourseRecord) -> None:
    """Restore a validated SavedCourseRecord into Streamlit session_state."""
    session_state["course_id"] = record.course_id
    session_state["course"] = record.course
    session_state["course_config"] = dict(record.course_config)
    session_state["lessons"] = dict(record.lessons)
    session_state["quizzes"] = dict(record.quizzes)
    session_state["quiz_answers"] = dict(record.quiz_answers)
    session_state["quiz_results"] = dict(record.quiz_results)
    session_state["chat_histories"] = dict(record.chat_histories)
    session_state["selected_lesson"] = "0" if record.course.lessons else None


def export_course_to_markdown(
    course: CourseOutline,
    lessons: Optional[Dict[str, LessonContent]] = None,
    quizzes: Optional[Dict[str, Quiz]] = None,
    course_config: Optional[Dict[str, object]] = None,
) -> str:
    """Export course outline, generated lessons, quizzes, and a clearly separated Answer Key to Markdown."""
    lessons_map = lessons or {}
    quizzes_map = quizzes or {}
    config = course_config or {}

    lines: List[str] = []
    lines.append(f"# {course.title}")
    lines.append("")
    if config.get("level") or config.get("topic"):
        level_str = config.get("level", "N/A")
        topic_str = config.get("topic", course.title)
        lines.append(
            f"**Topic:** {topic_str} | **Difficulty:** {level_str} | **Lessons:** {len(course.lessons)}"
        )
        lines.append("")
    lines.append("## Course Description")
    lines.append(course.description)
    lines.append("")
    lines.append("## Course Learning Objectives")
    for obj in course.learning_objectives:
        lines.append(f"- {obj}")
    lines.append("")
    lines.append("## Course Outline")
    for idx, lesson_meta in enumerate(course.lessons, start=1):
        lines.append(f"### Lesson {idx}: {lesson_meta.title}")
        lines.append(f"**Summary:** {lesson_meta.summary}")
        lines.append("")
        lines.append("**Objectives:**")
        for obj in lesson_meta.learning_objectives:
            lines.append(f"- {obj}")
        lines.append("")

    lines.append("## Detailed Lessons & Quizzes")
    lines.append("")
    answer_key_entries: List[str] = []

    for idx, lesson_meta in enumerate(course.lessons):
        key = str(idx)
        lesson_num = idx + 1
        lines.append(f"### Lesson {lesson_num}: {lesson_meta.title}")
        lines.append("")

        if key in lessons_map:
            content = lessons_map[key]
            lines.append("#### Introduction")
            lines.append(content.introduction)
            lines.append("")

            for sec_idx, section in enumerate(content.sections, start=1):
                lines.append(f"#### Section {sec_idx}: {section.heading}")
                lines.append(section.explanation)
                lines.append("")
                lines.append(f"> **Example:** {section.example}")
                if section.citations:
                    cite_str = ", ".join(
                        f"{c.source_filename} (p.{c.page_number}, {c.chunk_id})"
                        for c in section.citations
                    )
                    lines.append(f"> *Citations:* {cite_str}")
                lines.append("")

            lines.append("#### Lesson Summary")
            lines.append(content.summary)
            lines.append("")

            lines.append("#### Practice Exercises")
            for ex_idx, ex in enumerate(content.exercises, start=1):
                lines.append(f"{ex_idx}. **Question:** {ex.question}")
                lines.append(f"   - *Hint:* {ex.hint}")
            lines.append("")

            if content.citations:
                lines.append("#### Referenced PDF Sources")
                for c in content.citations:
                    lines.append(
                        f"- `{c.source_filename}`, page {c.page_number} (chunk `{c.chunk_id}`)"
                    )
                lines.append("")
        else:
            lines.append("*Lesson content has not been generated yet.*")
            lines.append("")

        if key in quizzes_map:
            quiz = quizzes_map[key]
            lines.append(f"#### Lesson {lesson_num} Quiz")
            lines.append("")
            answer_key_entries.append(
                f"### Lesson {lesson_num}: {lesson_meta.title} — Quiz Answer Key"
            )
            for q_idx, q in enumerate(quiz.questions, start=1):
                lines.append(f"**Q{q_idx}. {q.question}**")
                for opt_idx, opt in enumerate(q.options):
                    letter = chr(ord("A") + opt_idx)
                    lines.append(f"- ({letter}) {opt}")
                lines.append("")
                answer_key_entries.append(f"{q_idx}. **Correct Answer:** {q.correct_answer}")
                answer_key_entries.append(f"   - *Explanation:* {q.explanation}")
            answer_key_entries.append("")
        else:
            lines.append(f"*Quiz for Lesson {lesson_num} has not been generated yet.*")
            lines.append("")

    lines.append("================================================================================")
    lines.append("## Answer Key")
    lines.append("================================================================================")
    lines.append("")
    if answer_key_entries:
        lines.extend(answer_key_entries)
    else:
        lines.append("*No quizzes have been generated yet.*")
    lines.append("")
    return "\n".join(lines)
