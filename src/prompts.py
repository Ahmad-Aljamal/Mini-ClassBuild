import json


def course_outline_prompt(topic: str, level: str, lesson_count: int) -> str:
    return f"""
Create a course outline as JSON with keys: title, description, learning_objectives, lessons.
Each lesson must have title, learning_objectives (array), summary.
Topic: {topic}
Difficulty: {level}
Return EXACTLY {lesson_count} lessons. No markdown fences.
"""


def lesson_prompt(lesson_meta, course_title: str, passages=None) -> str:
    source = passages or []
    context = [
        {
            "chunk_id": p["chunk_id"],
            "page_number": p["page_number"],
            "source_filename": p["source_filename"],
            "text": p["text"],
        }
        for p in source
    ]
    return f"""
Create a complete lesson as JSON with keys: introduction, sections, summary, exercises, citations.
Course: {course_title}
Lesson title: {lesson_meta.title}
Lesson objectives: {lesson_meta.learning_objectives}
Lesson summary: {lesson_meta.summary}
Requirements:
- sections must contain 2 to 4 objects, each with heading, explanation, example, and citations.
- Create exactly two exercises, each with question and hint.
- Citations (both per-section and top-level) must be arrays of objects with chunk_id, page_number, source_filename. Cite ONLY supplied chunks.
- When Retrieved PDF context is non-empty, EVERY section in `sections` MUST include at least one valid citation in `section.citations`, and `citations` must list the cited chunks.
- If no source context is supplied, return `citations` as [] and each `section.citations` as [] and teach from general knowledge.
- If supplied context is insufficient, state this in the introduction and do not invent PDF facts.
Retrieved PDF context:
{json.dumps(context, ensure_ascii=False)}
Return JSON only.
"""


def quiz_prompt(lesson_meta, lesson_content) -> str:
    return f"""
Create exactly five multiple-choice questions about this lesson.
Return JSON: {{"questions":[{{"question":"...","options":["A","B","C","D"],"correct_answer":"must exactly match one option","explanation":"..."}}]}}
Each question must have exactly four options and one correct answer.
Lesson: {lesson_meta.title}
Objectives: {lesson_meta.learning_objectives}
Content: {lesson_content.model_dump_json()}
Return JSON only.
"""


def chatbot_prompt(course_title: str, lesson_title: str, lesson_content, passages, history) -> str:
    context = [
        {"page": p["page_number"], "chunk_id": p["chunk_id"], "text": p["text"]}
        for p in (passages or [])
    ]
    return f"""
Course: {course_title}
Selected lesson: {lesson_title}
Lesson content: {lesson_content.model_dump_json()}
PDF passages (if any): {json.dumps(context, ensure_ascii=False)}
Conversation so far: {json.dumps(history[-10:], ensure_ascii=False)}
Answer the student's latest question. When using PDF passages, cite page numbers and chunk IDs.
If PDF evidence is insufficient, say so clearly.
"""
