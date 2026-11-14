# Mini ClassBuild — AI Course Generator with PDF RAG, Validation & Persistence

Mini ClassBuild is a complete end-to-end educational web application built with **Python**, **Streamlit**, **Groq API** (`openai/gpt-oss-20b`), **Pydantic**, **pypdf**, and **Sentence Transformers** (`all-MiniLM-L6-v2`). It generates structured course outlines, multi-section lessons with practice exercises, and auto-graded 5-question quizzes, while supporting PDF-grounded Retrieval-Augmented Generation (RAG), a lesson-specific AI tutor chatbot, automated schema and citation validation with controlled retries, offline evaluation, local JSON course persistence, and Markdown export.

---

## Key Features Across All 3 Stages

### Stage 1 — Core AI Course Generation
- **Course Configuration:** Select any course topic, difficulty level (`Beginner`, `Intermediate`, `Advanced`), and number of lessons (`2–5`).
- **Structured Course Outlines:** Generates course title, description, learning objectives, and exact lesson count validated via Pydantic (`CourseOutline`).
- **Full Lesson Generation:** Generates an introduction, `2–4` detailed sections (each with heading, explanation, example, and optional PDF citations), summary, and `2` exercises with hints (`LessonContent`).
- **Interactive 5-Question Quizzes:** Generates 5 multiple-choice questions (`4` options each) with automatic Python grading, explanations, per-lesson score persistence, and same-quiz retakes.

### Stage 2 — PDF Processing, Semantic Search, RAG & Lesson Chatbot
- **PDF Ingestion (`pypdf`):** Upload PDF documents (up to 20 MB), extract and clean page-aware overlapping text chunks (`chunk_id`, `page_number`, `source_filename`), and gracefully reject empty, corrupted, encrypted, oversized, or scanned/image-only PDFs.
- **Local Embeddings & Similarity Threshold Filtering:** Encodes PDF chunks locally with `sentence-transformers/all-MiniLM-L6-v2` and retrieves the top-3 relevant passages using cosine similarity while enforcing a minimum similarity threshold (`DEFAULT_SIMILARITY_THRESHOLD = 0.25`) to reject unrelated passages.
- **RAG-Grounded Lessons with Per-Section Citations:** Grounds generated lessons in retrieved PDF chunks and validates that every section (`section.citations`) and the overall lesson (`lesson.citations`) reference only real retrieved `(chunk_id, page_number, source_filename)` tuples.
- **Lesson-Specific AI Chatbot:** Provides an interactive tutor per lesson (`course.title::lesson_index` isolation), optional PDF RAG context, conversation memory, and a `Clear chat for this lesson` button.

### Stage 3 — Validation, Offline Evaluation, Save/Load, Export & Final Verification
- **Validation & Controlled Retries (`src/validator.py`, `src/groq_client.py`):** Automatically retries Groq JSON generation when responses fail JSON parsing, Pydantic schema checks, lesson/section/exercise/question counts, or PDF citation verification.
- **Offline Evaluation (`scripts/run_evaluation.py`, `src/evaluation.py`):** Runs 6 predefined evaluation scenarios (both passing and failing/retry cases) using mocked Groq responses without consuming real API requests.
- **Course Persistence (`src/persistence.py`):** Saves courses, generated lessons, quizzes, scores, and chat history as validated JSON files in `saved_courses/` with unique IDs (`<slug>-<uuid8>`). Lists available saved courses in the sidebar, restores state into `st.session_state`, and handles missing, corrupted, or incompatible JSON files safely.
- **Markdown Export:** Exports the complete course (outline, generated lessons, citations, quizzes, and a clearly separated **Answer Key** section at the end) as a downloadable `.md` file.
- **Final Automated Verification (`scripts/final_check.py`):** Executes end-to-end checks and all `unittest` suites across Stages 1, 2, and 3 using mocked Groq responses and fake embeddings.

---

## Technologies Used

- **Frontend & State Management:** Streamlit (`st.session_state`, caching, forms, chat UI, download button)
- **LLM Integration:** Groq Python SDK (`openai/gpt-oss-20b` configurable via `GROQ_MODEL`)
- **Data Validation:** Pydantic v2 (`StrictModel` with `extra="forbid"`)
- **PDF & RAG Pipeline:** `pypdf`, `sentence-transformers` (`all-MiniLM-L6-v2`), `numpy` (cosine similarity with threshold filtering)
- **Testing & Offline Verification:** Python `unittest`, `unittest.mock`, `pytest`

---

## Project Structure

```text
Mini-ClassBuild/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── PROJECT_STRUCTURE.txt
├── README.md
├── scripts/
│   ├── run_evaluation.py
│   └── final_check.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── evaluation.py
│   ├── groq_client.py
│   ├── pdf_rag.py
│   ├── persistence.py
│   ├── prompts.py
│   ├── schemas.py
│   └── validator.py
└── tests/
    ├── test_pdf_rag.py
    ├── test_schemas.py
    └── test_stage3_workflow.py
```

---

## Installation & Environment Setup

### 1. Clone the Repository and Create a Virtual Environment

**Windows (PowerShell):**
```powershell
git clone https://github.com/Ahmad-Aljamal/Mini-ClassBuild.git
cd Mini-ClassBuild
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
git clone https://github.com/Ahmad-Aljamal/Mini-ClassBuild.git
cd Mini-ClassBuild
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure the Groq API Key Securely

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and set your Groq API key (and optional model name):
   ```env
   GROQ_API_KEY=your_actual_groq_api_key
   GROQ_MODEL=openai/gpt-oss-20b
   ```
  
---

## Running the Application & Required Testing Commands

### Step 10 — Run Offline Evaluation
Runs predefined pass/fail/retry validation and citation scenarios offline (no Groq API key or internet connection required):
```bash
python scripts/run_evaluation.py
```

### Step 11 — Run the Streamlit Application (Test Save, Load & Markdown Export)
Starts the interactive web UI at `http://localhost:8501`:
```bash
python -m streamlit run app.py
```

### Step 12 — Run Final Verification & Automated Tests
Runs all Stage 1, Stage 2, and Stage 3 verification checks and unit tests offline:
```bash
python scripts/final_check.py
```

You can also run the test suite directly via `unittest` or `pytest`:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## Demo Walkthrough & Reflection

### Application Workflow Walkthrough
1. **Generate & Validate Course:** Enter a topic (e.g., *Machine Learning Fundamentals*), pick difficulty and lesson count (`2–5`), and click **Generate course**. `generate_with_retry` validates the returned JSON against `CourseOutline`.
2. **Upload PDF & Build Embeddings (Optional RAG):** Upload a text-based PDF in the **PDF knowledge source (RAG)** section and click **Build / cache PDF embeddings**. Passages below the similarity threshold (`0.25`) are automatically filtered out.
3. **Generate Grounded Lesson & Quiz:** Enable **Ground this lesson in the uploaded PDF** and click **Generate lesson content**. Every section includes verified PDF chunk and page citations. Generate and submit the 5-question quiz or chat with the lesson tutor.
4. **Save, Load & Export:** Use the sidebar under **Save / Load Courses** to save the course as JSON in `saved_courses/`, reload any previously saved course into session state, or click **Export Course to Markdown** to download a complete `.md` study guide with a separated **Answer Key**.

### Engineering Reflection
Building Mini ClassBuild across three stages highlighted the importance of combining deterministic software guardrails with probabilistic LLM outputs:
- **Strict Schema & Citation Validation:** Relying on raw LLM JSON alone can cause UI crashes when fields are missing or counts drift. Introducing `Pydantic` models with `extra="forbid"`, per-section citation checks, and controlled retry loops (`generate_with_retry`) made content generation resilient.
- **Similarity Thresholding in RAG:** Adding a minimum cosine similarity threshold (`0.25`) prevented the retrieval pipeline from injecting irrelevant PDF chunks when a user uploads a document unrelated to the lesson topic.
- **Offline Evaluation & Mocking:** Designing `scripts/run_evaluation.py` and `scripts/final_check.py` with `unittest.mock` and deterministic fake embeddings allowed fast, repeatable verification of edge cases (corrupted JSON files, hallucinated citations, retry recovery) without API costs or network dependencies.
