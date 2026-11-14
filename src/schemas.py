import json
from typing import Any, Dict, List, Optional
import pydantic
from pydantic import BaseModel, Field

PYDANTIC_V2 = int(pydantic.__version__.split(".")[0]) >= 2
if PYDANTIC_V2:
    from pydantic import ConfigDict


def _list_field(min_len: int, max_len: Optional[int] = None, **kwargs):
    if PYDANTIC_V2:
        return Field(min_length=min_len, max_length=max_len, **kwargs)
    return Field(min_items=min_len, max_items=max_len, **kwargs)


class StrictModel(BaseModel):
    if PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")
    else:
        class Config:
            extra = "forbid"

    @classmethod
    def model_validate(cls, obj: Any):
        if PYDANTIC_V2:
            return super().model_validate(obj)
        return cls.parse_obj(obj)

    def model_dump(self, **kwargs) -> dict:
        if PYDANTIC_V2:
            return super().model_dump(**kwargs)
        return self.dict(**kwargs)

    def model_dump_json(self, **kwargs) -> str:
        if PYDANTIC_V2:
            return super().model_dump_json(**kwargs)
        return json.dumps(self.dict(**kwargs), ensure_ascii=False)


class LessonOutline(StrictModel):
    title: str = Field(min_length=1)
    learning_objectives: List[str] = _list_field(1)
    summary: str = Field(min_length=1)


class CourseOutline(StrictModel):
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    learning_objectives: List[str] = _list_field(1)
    lessons: List[LessonOutline] = _list_field(2, 5)


class Citation(StrictModel):
    chunk_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    source_filename: str = Field(min_length=1)


class LessonSection(StrictModel):
    heading: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    example: str = Field(min_length=1)
    citations: List[Citation] = Field(default_factory=list)


class Exercise(StrictModel):
    question: str = Field(min_length=1)
    hint: str = Field(min_length=1)


class LessonContent(StrictModel):
    introduction: str = Field(min_length=1)
    sections: List[LessonSection] = _list_field(2, 4)
    summary: str = Field(min_length=1)
    exercises: List[Exercise] = _list_field(2, 2)
    citations: List[Citation] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        self._sync_citations()

    def _sync_citations(self):
        seen = {(c.chunk_id, c.page_number, c.source_filename) for c in self.citations}
        merged = list(self.citations)
        for section in self.sections:
            for c in section.citations:
                key = (c.chunk_id, c.page_number, c.source_filename)
                if key not in seen:
                    seen.add(key)
                    merged.append(c)
        object.__setattr__(self, "citations", merged)


class QuizQuestion(StrictModel):
    question: str = Field(min_length=1)
    options: List[str] = _list_field(4, 4)
    correct_answer: str = Field(min_length=1)
    explanation: str = Field(min_length=1)

    def __init__(self, **data):
        super().__init__(**data)
        self.validate_correct_answer()

    def validate_correct_answer(self):
        if self.correct_answer not in self.options:
            raise ValueError("Correct answer must be one of the four options.")


class Quiz(StrictModel):
    questions: List[QuizQuestion] = _list_field(5, 5)

    def __init__(self, **data):
        super().__init__(**data)
        for question in self.questions:
            question.validate_correct_answer()


class SavedCourseRecord(StrictModel):
    course_id: str = Field(min_length=1)
    saved_at: str = Field(min_length=1)
    course_config: Dict[str, Any] = Field(default_factory=dict)
    course: CourseOutline
    lessons: Dict[str, LessonContent] = Field(default_factory=dict)
    quizzes: Dict[str, Quiz] = Field(default_factory=dict)
    quiz_answers: Dict[str, Dict[str, Optional[str]]] = Field(default_factory=dict)
    quiz_results: Dict[str, int] = Field(default_factory=dict)
    chat_histories: Dict[str, List[Dict[str, str]]] = Field(default_factory=dict)
