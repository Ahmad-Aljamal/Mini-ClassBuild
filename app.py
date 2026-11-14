import hashlib
import streamlit as st
from src.config import configure_page
from src.groq_client import GroqService, GroqServiceError
from src.pdf_rag import (
    DEFAULT_SIMILARITY_THRESHOLD,
    PDFProcessingError,
    clear_document_state,
    embed_chunks,
    process_pdf,
    semantic_search,
)
from src.persistence import (
    CoursePersistenceError,
    export_course_to_markdown,
    list_saved_courses,
    load_course_from_disk,
    restore_course_to_session_state,
    save_course_to_disk,
)
from src.prompts import (
    chatbot_prompt,
    course_outline_prompt,
    lesson_prompt,
    quiz_prompt,
)
from src.validator import (
    generate_with_retry,
    validate_course_outline_data,
    validate_lesson_content_data,
    validate_quiz_data,
)

configure_page()
st.title("🎓 Mini ClassBuild")
st.caption("AI-powered course generator with PDF-grounded RAG, validation, persistence & Markdown export")


def init_state():
    defaults = {
        "course_id": None,
        "course": None,
        "course_config": None,
        "lessons": {},
        "quizzes": {},
        "quiz_answers": {},
        "quiz_results": {},
        "pdf_doc": None,
        "pdf_embeddings": None,
        "pdf_fingerprint": None,
        "chat_histories": {},
        "selected_lesson": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


@st.cache_resource
def get_groq():
    return GroqService()


@st.cache_resource
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


try:
    groq = get_groq()
except Exception as exc:
    groq = None
    st.error(f"Groq configuration error: {exc}")

with st.sidebar:
    st.header("⚙️ Course Configuration")
    topic = st.text_input("Course topic", placeholder="e.g. Python programming")
    level = st.selectbox("Difficulty", ["Beginner", "Intermediate", "Advanced"])
    lesson_count = st.selectbox("Number of lessons", [2, 3, 4, 5], index=2)
    generate_course = st.button("Generate course", type="primary", use_container_width=True)

    st.divider()
    st.header("💾 Save / Load Courses")

    if st.session_state.course is not None:
        if st.button("Save current course as JSON", use_container_width=True):
            try:
                record = save_course_to_disk(
                    course=st.session_state.course,
                    lessons=st.session_state.lessons,
                    quizzes=st.session_state.quizzes,
                    course_config=st.session_state.course_config,
                    quiz_answers=st.session_state.quiz_answers,
                    quiz_results=st.session_state.quiz_results,
                    chat_histories=st.session_state.chat_histories,
                    course_id=st.session_state.course_id,
                )
                st.session_state.course_id = record.course_id
                st.success(f"Saved course `{record.course_id}`.")
            except CoursePersistenceError as exc:
                st.error(str(exc))

        md_export = export_course_to_markdown(
            course=st.session_state.course,
            lessons=st.session_state.lessons,
            quizzes=st.session_state.quizzes,
            course_config=st.session_state.course_config,
        )
        export_filename = f"{(st.session_state.course_id or 'course_export')}.md"
        st.download_button(
            label="📥 Export Course to Markdown",
            data=md_export,
            file_name=export_filename,
            mime="text/markdown",
            use_container_width=True,
        )

    saved_items = list_saved_courses()
    if saved_items:
        options = {
            f"{item['title']} ({item['course_id']})": item for item in saved_items
        }
        selected_label = st.selectbox("Available saved courses", list(options.keys()))
        selected_item = options[selected_label]
        if not selected_item["valid"]:
            st.warning(f"Corrupted/incompatible file: {selected_item['error']}")
        if st.button("Load selected course", use_container_width=True):
            try:
                loaded_record = load_course_from_disk(str(selected_item["file_path"]))
                restore_course_to_session_state(st.session_state, loaded_record)
                st.success(f"Loaded course '{loaded_record.course.title}'.")
                st.rerun()
            except CoursePersistenceError as exc:
                st.error(f"Could not load course: {exc}")
    else:
        st.caption("No saved courses found in `saved_courses/` yet.")

    st.divider()
    st.caption("Required: add your GROQ_API_KEY to the .env file.")

if generate_course:
    if not topic.strip():
        st.warning("Please enter a course topic.")
    elif groq is None:
        st.error("Groq is not configured. Add your API key to .env and restart the app.")
    else:
        with st.spinner("Generating and validating course outline..."):
            try:
                course = generate_with_retry(
                    groq,
                    course_outline_prompt(topic.strip(), level, lesson_count),
                    lambda raw: validate_course_outline_data(raw, expected_lessons=lesson_count),
                    max_retries=2,
                )
                st.session_state.course_id = None
                st.session_state.course = course
                st.session_state.course_config = {
                    "topic": topic.strip(),
                    "level": level,
                    "lesson_count": lesson_count,
                }
                st.session_state.lessons = {}
                st.session_state.quizzes = {}
                st.session_state.quiz_answers = {}
                st.session_state.quiz_results = {}
                st.session_state.chat_histories = {}
                st.session_state.selected_lesson = None
                st.success("Course created and validated successfully.")
            except Exception as exc:
                st.error(f"Could not generate a valid course: {exc}")

course = st.session_state.course

if course:
    st.header(course.title)
    st.write(course.description)
    st.subheader("Learning objectives")
    for obj in course.learning_objectives:
        st.markdown(f"- {obj}")

    st.subheader("Course outline")
    lesson_titles = []
    for i, lesson in enumerate(course.lessons, 1):
        with st.expander(f"Lesson {i}: {lesson.title}", expanded=False):
            st.markdown("**Objectives**")
            for obj in lesson.learning_objectives:
                st.markdown(f"- {obj}")
            st.write(lesson.summary)
        lesson_titles.append(f"{i}. {lesson.title}")

    chosen = st.selectbox("Select a lesson", lesson_titles)
    lesson_index = lesson_titles.index(chosen)
    lesson_meta = course.lessons[lesson_index]
    lesson_key = f"{lesson_index}"
    st.session_state.selected_lesson = lesson_key

    use_pdf = st.checkbox(
        "Ground this lesson in the uploaded PDF",
        value=False,
        disabled=st.session_state.pdf_doc is None,
    )
    retrieved = []
    if use_pdf and st.session_state.pdf_doc:
        with st.spinner("Retrieving relevant PDF passages..."):
            try:
                query = (
                    f"{lesson_meta.title}. {lesson_meta.summary}. "
                    + " ".join(lesson_meta.learning_objectives)
                )
                retrieved = semantic_search(
                    query,
                    st.session_state.pdf_doc["chunks"],
                    st.session_state.pdf_embeddings,
                    model=get_embedding_model() if st.session_state.pdf_embeddings is not None else None,
                    top_k=3,
                    min_similarity=DEFAULT_SIMILARITY_THRESHOLD,
                )
                if not retrieved:
                    st.warning(
                        f"No sufficiently relevant passages (similarity >= {DEFAULT_SIMILARITY_THRESHOLD:.2f}) "
                        "were found in this PDF. Lesson will be generated without PDF grounding."
                    )
            except Exception as exc:
                st.error(f"PDF retrieval error: {exc}")

    st.divider()
    st.header(f"📘 {lesson_meta.title}")
    if lesson_key not in st.session_state.lessons:
        if st.button("Generate lesson content", type="primary", disabled=groq is None):
            with st.spinner("Generating and validating lesson..."):
                try:
                    lesson_content = generate_with_retry(
                        groq,
                        lesson_prompt(lesson_meta, course.title, retrieved),
                        lambda raw: validate_lesson_content_data(
                            raw,
                            retrieved_chunks=retrieved,
                            require_section_citations=bool(retrieved),
                        ),
                        max_retries=2,
                    )
                    st.session_state.lessons[lesson_key] = lesson_content
                    st.rerun()
                except Exception as exc:
                    st.error(f"Lesson generation failed: {exc}")
    else:
        content = st.session_state.lessons[lesson_key]
        st.markdown("### Introduction")
        st.write(content.introduction)
        for section in content.sections:
            with st.expander(section.heading, expanded=True):
                st.write(section.explanation)
                st.info(f"Example: {section.example}")
                if section.citations:
                    sec_cites = ", ".join(
                        f"{c.source_filename} (p.{c.page_number}, {c.chunk_id})"
                        for c in section.citations
                    )
                    st.caption(f"Section citations: {sec_cites}")
        st.markdown("### Summary")
        st.write(content.summary)
        st.markdown("### Exercises")
        for i, exercise in enumerate(content.exercises, 1):
            st.markdown(f"**Exercise {i}:** {exercise.question}")
            st.caption(f"Hint: {exercise.hint}")
        if content.citations:
            st.markdown("### PDF sources")
            for cite in content.citations:
                st.caption(
                    f"{cite.source_filename}, page {cite.page_number} (chunk {cite.chunk_id})"
                )

        st.divider()
        st.header("📝 Lesson quiz")
        if lesson_key not in st.session_state.quizzes:
            if st.button("Generate 5-question quiz", disabled=groq is None):
                with st.spinner("Creating and validating quiz..."):
                    try:
                        quiz = generate_with_retry(
                            groq,
                            quiz_prompt(lesson_meta, content),
                            validate_quiz_data,
                            max_retries=2,
                        )
                        st.session_state.quizzes[lesson_key] = quiz
                        st.session_state.quiz_answers.pop(lesson_key, None)
                        st.session_state.quiz_results.pop(lesson_key, None)
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Quiz generation failed: {exc}")
        else:
            quiz = st.session_state.quizzes[lesson_key]
            with st.form(f"quiz_form_{lesson_key}"):
                answers = {}
                for q_index, question in enumerate(quiz.questions):
                    st.markdown(f"**{q_index + 1}. {question.question}**")
                    answers[str(q_index)] = st.radio(
                        "Choose one:",
                        question.options,
                        index=None,
                        key=f"answer_{lesson_key}_{q_index}",
                        label_visibility="collapsed",
                    )
                submitted = st.form_submit_button("Submit answers", type="primary")
            if submitted:
                if any(v is None for v in answers.values()):
                    st.warning("Please answer all five questions before submitting.")
                else:
                    score = sum(
                        1
                        for i, q in enumerate(quiz.questions)
                        if answers[str(i)] == q.correct_answer
                    )
                    st.session_state.quiz_answers[lesson_key] = answers
                    st.session_state.quiz_results[lesson_key] = score
                    st.rerun()
            if lesson_key in st.session_state.quiz_results:
                score = st.session_state.quiz_results[lesson_key]
                st.success(f"Your score: {score}/5")
                for i, q in enumerate(quiz.questions):
                    chosen_answer = st.session_state.quiz_answers[lesson_key][str(i)]
                    is_correct = chosen_answer == q.correct_answer
                    st.markdown(
                        f"**Question {i + 1}:** {'✅ Correct' if is_correct else '❌ Incorrect'}"
                    )
                    st.write(f"Your answer: {chosen_answer}")
                    st.write(f"Correct answer: {q.correct_answer}")
                    st.caption(q.explanation)
                if st.button("Retake same quiz", key=f"retake_{lesson_key}"):
                    st.session_state.quiz_answers.pop(lesson_key, None)
                    st.session_state.quiz_results.pop(lesson_key, None)
                    for i in range(5):
                        st.session_state.pop(f"answer_{lesson_key}_{i}", None)
                    st.rerun()

        st.divider()
        st.header("💬 Ask the lesson assistant")
        pdf_chat = st.checkbox(
            "Use uploaded PDF as chatbot source",
            key=f"pdf_chat_{lesson_key}",
            disabled=st.session_state.pdf_doc is None,
        )
        chat_key = f"{st.session_state.course.title}::{lesson_key}"
        history = st.session_state.chat_histories.setdefault(chat_key, [])
        for msg in history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        question = st.chat_input("Ask a question about this lesson...")
        if question:
            if groq is None:
                st.error("Groq is not configured.")
            else:
                history.append({"role": "user", "content": question})
                passages = []
                if pdf_chat and st.session_state.pdf_doc:
                    passages = semantic_search(
                        question,
                        st.session_state.pdf_doc["chunks"],
                        st.session_state.pdf_embeddings,
                        model=get_embedding_model() if st.session_state.pdf_embeddings is not None else None,
                        top_k=3,
                        min_similarity=DEFAULT_SIMILARITY_THRESHOLD,
                    )
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            answer = groq.chat(
                                chatbot_prompt(
                                    course.title,
                                    lesson_meta.title,
                                    content,
                                    passages,
                                    history[:-1],
                                )
                            )
                            st.markdown(answer)
                            history.append({"role": "assistant", "content": answer})
                        except Exception as exc:
                            history.pop()
                            st.error(f"Chat request failed: {exc}")
        if st.button("Clear chat for this lesson", key=f"clear_chat_{lesson_key}"):
            st.session_state.chat_histories[chat_key] = []
            st.rerun()

st.divider()
st.header("📄 PDF knowledge source (RAG)")
uploaded = st.file_uploader("Upload a PDF document", type=["pdf"], help="Maximum size: 20 MB")
if uploaded is not None:
    fingerprint = hashlib.sha256(uploaded.getvalue()).hexdigest()
    if fingerprint != st.session_state.pdf_fingerprint:
        with st.spinner("Extracting and splitting PDF text..."):
            try:
                doc = process_pdf(uploaded.getvalue(), uploaded.name)
                st.session_state.pdf_doc = doc
                st.session_state.pdf_fingerprint = fingerprint
                st.session_state.pdf_embeddings = None
                st.success(
                    f"Loaded {doc['page_count']} pages and {len(doc['chunks'])} text chunks."
                )
            except PDFProcessingError as exc:
                clear_document_state(st.session_state)
                st.error(str(exc))
if st.session_state.pdf_doc:
    doc = st.session_state.pdf_doc
    c1, c2, c3 = st.columns(3)
    c1.metric("Pages", doc["page_count"])
    c2.metric("Chunks", len(doc["chunks"]))
    c3.metric("File", doc["filename"])
    if st.button("Build / cache PDF embeddings", type="primary"):
        with st.spinner(
            "Loading local embedding model and encoding document (first run may download model files)..."
        ):
            try:
                model = get_embedding_model()
                st.session_state.pdf_embeddings = embed_chunks(doc["chunks"], model)
                st.success("Embeddings are ready.")
            except Exception as exc:
                st.error(f"Could not create embeddings: {exc}")
    if st.session_state.pdf_embeddings is not None:
        st.success("Semantic search is ready.")
        search_query = st.text_input("Search inside the uploaded PDF")
        if st.button("Search PDF", disabled=not bool(search_query.strip())):
            try:
                results = semantic_search(
                    search_query,
                    doc["chunks"],
                    st.session_state.pdf_embeddings,
                    model=get_embedding_model(),
                    top_k=3,
                    min_similarity=DEFAULT_SIMILARITY_THRESHOLD,
                )
                if not results:
                    st.warning(
                        f"No passages met the minimum similarity threshold ({DEFAULT_SIMILARITY_THRESHOLD:.2f})."
                    )
                for item in results:
                    st.markdown(
                        f"**Page {item['page_number']} · {item['chunk_id']} · Similarity {item['score']:.3f}**"
                    )
                    st.write(item["text"])
                    st.divider()
            except Exception as exc:
                st.error(f"Search failed: {exc}")
    if st.button("Remove current PDF"):
        clear_document_state(st.session_state)
        st.rerun()
