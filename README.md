# 🧠 AI Resume Analyzer & Job Match System

A production-ready AI system that analyzes resumes against job descriptions using **semantic embeddings**, **vector similarity search**, and **NLP skill extraction** — entirely open-source, no paid APIs required.

---

## 🎯 What It Does

Upload a resume + paste a job description → get:

| Output | Description |
|---|---|
| **Match Score (0–100)** | Weighted composite of semantic similarity, skill overlap, and keyword density |
| **Matched Skills** | Skills you have that the JD requires |
| **Missing Skills** | Skills the JD requires that you're lacking |
| **Section Relevance** | How relevant each resume section is to the role |
| **AI Suggestions** | Rule-based recommendations to improve your resume |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit Frontend                          │
│  Upload PDF │ Paste JD │ View Score │ Skill Gaps │ Suggestions  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP (or direct call)
┌──────────────────────────▼──────────────────────────────────────┐
│                    FastAPI Backend                                │
│  POST /upload_resume │ POST /analyze_resume │ POST /job_match   │
└──────┬───────────────┬──────────────────┬───────────────────────┘
       │               │                  │
┌──────▼──────┐ ┌──────▼──────┐ ┌────────▼────────────────────┐
│   Parser    │ │   Skill     │ │    Similarity Engine         │
│ pdfplumber  │ │  Extractor  │ │ Sentence Transformers        │
│  + PyPDF2   │ │  spaCy NLP  │ │ all-MiniLM-L6-v2 (384-dim)  │
│ Sections ✓  │ │ 200+ skills │ │ FAISS cosine similarity      │
└─────────────┘ └─────────────┘ └─────────────────────────────┘
                                         │
                               ┌─────────▼──────────┐
                               │  Recommendation     │
                               │  Engine (rule-based)│
                               │  + Resource links   │
                               └────────────────────┘
```

### Scoring Formula

```
Overall Score = 0.40 × Semantic Score
              + 0.40 × Skill Overlap Score
              + 0.20 × Keyword Density Score
```

---

## 📁 Project Structure

```
resume-analyzer/
│
├── api/
│   └── main.py              # FastAPI endpoints (upload, analyze, match)
│
├── utils/
│   ├── __init__.py
│   ├── parser.py            # PDF → text → structured sections
│   ├── skill_extractor.py   # spaCy + regex skill taxonomy (200+ skills)
│   ├── embeddings.py        # Sentence Transformer + FAISS index
│   └── similarity.py        # Scoring engine + recommendation generator
│
├── app/
│   └── streamlit_app.py     # Streamlit UI (4 tabs: Analyze, Results, Compare, Dashboard)
│
├── tests/
│   └── test_pipeline.py     # End-to-end test suite (no pytest required)
│
├── data/
│   └── sample_resumes/
│       ├── sample_ml_engineer.txt
│       └── sample_jd_ml_engineer.txt
│
├── models/                  # Sentence Transformer cache (auto-created)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.9+
- pip or conda
- ~500 MB disk space (model cache)

### 1. Clone & install

```bash
git clone https://github.com/yourusername/resume-analyzer
cd resume-analyzer

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Download spaCy language model

```bash
python -m spacy download en_core_web_sm
```

### 3. (Optional) Pre-download the embedding model

The Sentence Transformer model (~90 MB) downloads automatically on first use.
To pre-download:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

---

## 🚀 Running the Application

### Option A — Streamlit UI (recommended)

```bash
streamlit run app/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501)

### Option B — FastAPI Backend

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Option C — Both (recommended for production)

```bash
# Terminal 1: API
uvicorn api.main:app --port 8000 --reload

# Terminal 2: Streamlit
streamlit run app/streamlit_app.py
```

In Streamlit settings sidebar, select **API (FastAPI)** mode and set URL to `http://localhost:8000`.

---

## 🧪 Running Tests

```bash
python tests/test_pipeline.py
```

Or with pytest:

```bash
pytest tests/ -v
```

---

## 🌐 API Reference

### `POST /upload_resume`

Upload a PDF or TXT resume.

```bash
curl -X POST http://localhost:8000/upload_resume \
  -F "file=@my_resume.pdf"
```

**Response:**
```json
{
  "session_id": "abc123",
  "contact": {"name": "Alex Kim", "email": "alex@email.com"},
  "skills": ["Python", "PyTorch", "Docker"],
  "skills_by_category": {"ml_ai": ["PyTorch", "TensorFlow"], ...},
  "word_count": 342,
  "processing_time_ms": 87.3
}
```

---

### `POST /analyze_resume`

Full analysis pipeline.

```bash
curl -X POST http://localhost:8000/analyze_resume \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "...",
    "job_description": "..."
  }'
```

**Response:**
```json
{
  "overall_score": 72,
  "label": "Good Match",
  "semantic_score": 78.4,
  "skill_score": 65.0,
  "keyword_score": 71.2,
  "matched_skills": ["Python", "PyTorch", "Docker"],
  "missing_skills": ["Kubernetes", "Spark"],
  "section_relevance": [...],
  "recommendations": [
    {
      "type": "SKILL GAP",
      "priority": "high",
      "text": "Build a containerized project using Kubernetes...",
      "resource": "kubernetes.io/docs/tutorials"
    }
  ],
  "processing_time_ms": 1243.5
}
```

---

### `POST /job_match`

Quick match from skill lists.

```bash
curl -X POST http://localhost:8000/job_match \
  -H "Content-Type: application/json" \
  -d '{
    "resume_skills": ["Python", "Docker", "FastAPI"],
    "jd_skills": ["Python", "Kubernetes", "FastAPI"],
    "resume_text": "...",
    "job_description": "..."
  }'
```

---

## 🔧 Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **PDF Parsing** | pdfplumber + PyPDF2 | Extract text from resume PDFs |
| **NLP** | spaCy `en_core_web_sm` | Named entity recognition, noun chunks |
| **Skill Extraction** | Custom taxonomy (200+ skills) + regex | Identify technical skills |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim semantic vectors |
| **Vector Search** | FAISS (IndexFlatIP) | Cosine similarity search |
| **Backend API** | FastAPI + uvicorn | REST API with OpenAPI docs |
| **Frontend** | Streamlit | Interactive web UI |
| **Data** | Pandas + NumPy | Data processing |
| **Logging** | Loguru | Structured logging |

---

## 📊 Skill Taxonomy Coverage

The skill extractor covers **200+ skills** across 8 categories:

- **Programming Languages**: Python, JavaScript, TypeScript, Go, Rust, Java, C++, ...
- **ML / AI**: PyTorch, TensorFlow, scikit-learn, Hugging Face, spaCy, FAISS, LLMs, ...
- **Web Frameworks**: FastAPI, Django, React, Node.js, Spring Boot, ...
- **Data Engineering**: SQL, PostgreSQL, MongoDB, Kafka, Spark, Airflow, Snowflake, ...
- **DevOps / Cloud**: Docker, Kubernetes, AWS, GCP, Azure, Terraform, CI/CD, ...
- **API / Architecture**: REST, GraphQL, gRPC, Microservices, Event-driven, ...
- **Tools**: Git, GitHub, MLflow, Jupyter, Prometheus, ...
- **Soft Skills**: Agile, Scrum, Technical Leadership, ...

---

## 💡 Add to Your Resume

After completing this project:

```
• Built AI Resume Analyzer using FAISS vector search and Sentence Transformer
  embeddings to compute semantic resume–JD similarity scores
• Implemented NLP skill extraction pipeline with spaCy covering 200+ technical skills
  across 8 categories including ML, DevOps, and data engineering
• Developed FastAPI REST backend with 3 endpoints and Streamlit UI
  for real-time interactive resume analysis
```

---

## 📄 License

MIT License — free for personal and commercial use.
