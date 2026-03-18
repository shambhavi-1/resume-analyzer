# 🧠 AI Resume Analyzer & Job Match System

An AI system that analyzes resumes against job descriptions using semantic 
embeddings, vector similarity search, and NLP skill extraction — 
entirely open-source, no paid APIs required.

<img width="1920" height="890" alt="Screenshot (22)" src="https://github.com/user-attachments/assets/c30b28c7-d12d-4183-acf4-a98d9f5832e8" />


---

## 🎯 What It Does

Upload a resume + paste a job description → get an 8-dimension analysis:

| Output | Description |
|---|---|
| **Match Score (0–100)** | Weighted composite of 7 analysis dimensions |
| **Matched Skills** | Skills you have that the JD requires |
| **Missing Skills** | Skills the JD requires that you're lacking |
| **Experience Match** | Years of experience + seniority level vs requirement |
| **Education Match** | Degree level + field of study relevance |
| **ATS Simulation** | How a real ATS system would score your resume |
| **Resume Quality** | Format, metrics, action verbs, structure check |
| **Recommendations** | Prioritized suggestions to improve your resume |

---

## 📊 Scoring Formula

Overall Score = 25% × Semantic Similarity
              + 25% × Skill Overlap
              + 15% × Experience Match
              + 10% × Education Match
              + 10% × Job Title Match
              + 10% × Keyword Density
              +  5% × ATS Score

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| Vector Search | FAISS (cosine similarity) |
| NLP | spaCy en_core_web_sm |
| Skill Extraction | Custom taxonomy 200+ skills + regex |
| PDF Parsing | pdfplumber + PyPDF2 |
| Backend API | FastAPI + uvicorn |
| Frontend | Streamlit |
| Data Processing | Pandas + NumPy |

---

## 📁 Project Structure

resume-analyzer/
├── api/
│   └── main.py                  # FastAPI backend (3 endpoints)
├── utils/
│   ├── parser.py                # PDF → text → structured sections
│   ├── skill_extractor.py       # spaCy + regex (200+ skills)
│   ├── embeddings.py            # Sentence Transformers + FAISS
│   ├── similarity.py            # Master scoring engine
│   ├── experience_analyzer.py   # Years + seniority detection
│   ├── education_analyzer.py    # Degree + field relevance
│   └── job_title_analyzer.py    # Title match + ATS simulation
├── app/
│   └── streamlit_app.py         # Streamlit UI (4 tabs)
├── data/
│   └── sample_resumes/          # Sample resume + JD for testing
└── requirements.txt

---

## ⚙️ Installation

git clone https://github.com/shambhavi-1/resume-analyzer
cd resume-analyzer

python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm

---

## 🚀 Run

streamlit run app/streamlit_app.py

Open http://localhost:8501

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /upload_resume | POST | Upload PDF, extract text + skills |
| /analyze_resume | POST | Full 8-dimension analysis |
| /job_match | POST | Quick match score from skill lists |
| /health | GET | Health check |

FastAPI docs: http://localhost:8000/docs

---

## 📊 Skill Taxonomy — 8 Categories

- Programming Languages — Python, JavaScript, Go, Rust, Java, C++
- ML / AI — PyTorch, TensorFlow, Hugging Face, FAISS, LLMs, RAG, LoRA
- Web Frameworks — FastAPI, Django, React, Node.js
- Data Engineering — SQL, PostgreSQL, Kafka, Spark, Airflow
- DevOps / Cloud — Docker, Kubernetes, AWS, GCP, Terraform, CI/CD
- APIs — REST, GraphQL, gRPC, Microservices
- Tools — Git, MLflow, Jupyter
- Soft Skills — Agile, Scrum, Leadership

---

## 👩‍💻 Built By

Shambhavi V. Abbigeri
Aspiring AI/ML Engineer
