"""
app/streamlit_app.py  (v3 — high contrast light theme + fixed analysis bugs)
"""
import sys, time, re, subprocess
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="AI Resume Analyzer", page_icon="🧠", layout="wide")

@st.cache_resource
def load_spacy_model():
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

load_spacy_model()

st.markdown("""
<style>
/* ── Reset to high-contrast light theme ── */
.stApp { background: #f5f6fa !important; color: #1a1a2e !important; }
section[data-testid="stSidebar"] { background: #ffffff !important; border-right: 2px solid #e0e0e0; }
section[data-testid="stSidebar"] * { color: #1a1a2e !important; }
.stTextArea textarea { background:#fff !important; color:#1a1a2e !important; border:2px solid #c0c0d0 !important; border-radius:8px !important; font-size:0.88rem !important; }
.stTextArea textarea:focus { border-color:#6c5ce7 !important; }
.stFileUploader { background:#fff !important; border:2px dashed #b0b0c0 !important; border-radius:8px !important; }
.stButton > button { background: linear-gradient(135deg,#6c5ce7,#a78bfa) !important; color:#fff !important; font-weight:700 !important; border:none !important; border-radius:8px !important; padding:0.6rem 1.5rem !important; }
.stButton > button:disabled { background:#ccc !important; }
.stTabs [data-baseweb="tab"] { color:#666 !important; font-weight:600 !important; }
.stTabs [aria-selected="true"] { color:#6c5ce7 !important; border-bottom:3px solid #6c5ce7 !important; }
.stProgress > div > div { background: linear-gradient(90deg,#6c5ce7,#a78bfa) !important; }
.stExpander { background:#fff !important; border:1px solid #e0e0e0 !important; border-radius:8px !important; }
div[data-testid="metric-container"] { background:#fff !important; border:1px solid #e0e0e0 !important; border-radius:8px !important; padding:1rem !important; }
.stCaption { color:#666 !important; }
.stSuccess { background:#e8f8f0 !important; color:#1a6640 !important; border:1px solid #52c27a !important; border-radius:8px !important; }
.stInfo { background:#e8eeff !important; color:#1a2a6c !important; border:1px solid #6c8aff !important; border-radius:8px !important; }
.stWarning { background:#fff8e1 !important; color:#7a5700 !important; border:1px solid #ffcc02 !important; border-radius:8px !important; }
.stError { background:#ffeaea !important; color:#8b0000 !important; border:1px solid #ff6b6b !important; border-radius:8px !important; }
h1,h2,h3,h4,h5,h6,p,span,div,label { color: inherit !important; }

/* ── Custom components ── */
.page-header {
  background: linear-gradient(135deg, #4834d4, #6c5ce7, #a29bfe);
  padding: 1.8rem 2rem 1.4rem; border-radius: 14px;
  margin-bottom: 1.5rem; box-shadow: 0 4px 20px rgba(108,92,231,0.25);
}
.page-header h1 { color: #fff !important; font-size: 1.8rem; font-weight: 800; margin: 0; }
.page-header p  { color: rgba(255,255,255,0.82) !important; font-size: 0.88rem; margin: 0.4rem 0 0; }

.score-big {
  background: #fff; border: 2px solid #e0e0e0; border-radius: 14px;
  padding: 1.4rem; text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.07);
}
.score-big .num { font-size: 3.2rem; font-weight: 900; line-height: 1; }
.score-big .lbl { font-size: 0.82rem; color: #666 !important; margin-top: 4px; font-weight: 600; }

.sub-score {
  background: #fff; border: 1px solid #e8e8f0; border-radius: 10px;
  padding: 1rem 1.1rem; text-align: center;
  box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
.sub-score .num { font-size: 1.9rem; font-weight: 800; line-height: 1; }
.sub-score .lbl { font-size: 0.75rem; color: #777 !important; margin-top: 3px; }

.dim-row {
  background: #fff; border: 1px solid #e8e8f0; border-radius: 9px;
  padding: 0.75rem 1rem; margin-bottom: 7px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.dim-row .dlabel { display:flex; justify-content:space-between;
  font-size: 0.84rem; font-weight: 600; color: #2d2d44 !important; margin-bottom: 5px; }
.dim-row .dtip { font-size: 0.74rem; color: #888 !important; font-weight: 400; }
.dim-bar-bg { background: #eef0f7; border-radius: 5px; height: 9px; overflow: hidden; }

.deep-card {
  background: #fff; border: 1px solid #e0e0ec; border-radius: 11px;
  padding: 1.1rem 1.3rem; margin-bottom: 1rem;
  box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}
.deep-card .row { display:flex; justify-content:space-between; align-items:baseline;
  padding: 5px 0; border-bottom: 1px solid #f0f0f7; font-size: 0.85rem; }
.deep-card .row:last-child { border-bottom: none; }
.deep-card .key { color: #555 !important; font-weight: 600; }
.deep-card .val { color: #1a1a2e !important; font-weight: 500; text-align:right; max-width:60%; }

.chip-wrap { display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }
.chip-g { background:#e6f9f1; border:1px solid #52c27a; color:#1a6640 !important;
  padding:4px 13px; border-radius:20px; font-size:0.78rem; font-family:monospace; font-weight:600; }
.chip-r { background:#ffeaea; border:1px solid #ff6b6b; color:#8b0000 !important;
  padding:4px 13px; border-radius:20px; font-size:0.78rem; font-family:monospace; font-weight:600; }
.chip-b { background:#e8eeff; border:1px solid #6c8aff; color:#1a2a6c !important;
  padding:4px 13px; border-radius:20px; font-size:0.78rem; font-family:monospace; font-weight:600; }
.chip-a { background:#fff8e1; border:1px solid #ffb300; color:#7a5700 !important;
  padding:4px 13px; border-radius:20px; font-size:0.78rem; font-family:monospace; font-weight:600; }

.rec-card {
  background: #fff; border-left: 4px solid #6c5ce7;
  border-top: 1px solid #e8e8f0; border-right: 1px solid #e8e8f0;
  border-bottom: 1px solid #e8e8f0; border-radius: 0 10px 10px 0;
  padding: 0.9rem 1.1rem; margin-bottom: 9px;
  box-shadow: 0 1px 5px rgba(0,0,0,0.04);
}
.rec-type { font-size: 0.7rem; font-family: monospace; color: #6c5ce7 !important;
  font-weight: 700; letter-spacing: 0.6px; text-transform: uppercase; margin-bottom: 5px; }
.rec-text { font-size: 0.87rem; color: #2d2d44 !important; line-height: 1.65; }
.rec-res  { font-size: 0.78rem; color: #2563eb !important; margin-top: 5px; }
.priority-h { color: #c0392b !important; }
.priority-m { color: #d68910 !important; }
.priority-l { color: #1a6640 !important; }

.pass-item { color: #1a6640 !important; font-size: 0.83rem; padding: 2px 0; }
.fail-item { color: #8b0000 !important; font-size: 0.83rem; padding: 2px 0; }
.warn-item { color: #7a5700 !important; font-size: 0.83rem; padding: 2px 0; }

.sec-bar-bg { background: #eef0f7; border-radius: 4px; height: 8px; overflow: hidden; margin-top:3px; }
.verdict-box {
  padding: 0.8rem 1.1rem; border-radius: 9px; margin: 0.8rem 0 0;
  font-size: 0.88rem; font-weight: 600;
}
.verdict-green { background:#e6f9f1; border:1px solid #52c27a; color:#1a6640 !important; }
.verdict-amber { background:#fff8e1; border:1px solid #ffb300; color:#7a5700 !important; }
.verdict-red   { background:#ffeaea; border:1px solid #ff6b6b; color:#8b0000 !important; }

.section-title { font-size:1.05rem; font-weight:700; color:#2d2d44 !important;
  margin: 1.2rem 0 0.6rem; padding-bottom:5px; border-bottom:2px solid #e8e8f0; }

#MainMenu{visibility:hidden;}footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
def bar(val, color, height=9):
    return f"""<div class="dim-bar-bg">
      <div style="width:{min(val,100):.0f}%;background:{color};height:{height}px;border-radius:5px;"></div>
    </div>"""

def score_color(v):
    if v >= 70: return "#16a34a"
    if v >= 45: return "#d97706"
    return "#dc2626"

def verdict_class(v):
    if v >= 70: return "verdict-green"
    if v >= 45: return "verdict-amber"
    return "verdict-red"

# ── Samples ────────────────────────────────────────────────────────────────────
SAMPLE_RESUMES = {
    "ML Engineer (Senior)": """Alex Kim | Senior Machine Learning Engineer
Email: alex.kim@email.com | GitHub: github.com/alexkim | LinkedIn: linkedin.com/in/alexkim

SKILLS
Python, PyTorch, TensorFlow, scikit-learn, Pandas, NumPy, FastAPI, Docker, AWS, SQL,
PostgreSQL, Redis, Git, NLP, Computer Vision, Flask, REST APIs, Jupyter, MLflow,
Sentence Transformers, spaCy, FAISS, Hugging Face, Kafka, Kubernetes

EXPERIENCE
Senior ML Engineer — DataVision Corp (2021–Present)
• Built 8 production ML models serving 15M+ daily users
• Reduced inference latency by 45% via INT8 quantization
• Built MLOps pipeline: MLflow, Docker, GitHub Actions CI/CD

ML Engineer — StartupAI Inc (2019–2021)
• NLP sentiment analysis pipeline — 93% accuracy
• Trained CNN image classification models using TensorFlow

EDUCATION
M.S. Computer Science — Stanford University (2019)
B.S. Mathematics — UC Berkeley (2017)

PROJECTS
• AutoML Framework: 2,400 GitHub stars
• Real-time Fraud Detection: LSTM, 0.1% false positive rate""",

    "Fresher / Student (like Shambhavi)": """Shambhavi V. Abbigeri
Email: shambhaviabbigeri@gmail.com | LinkedIn: linkedin.com/in/shambhavi-abbigeri

SUMMARY
Aspiring AI/ML Engineer with strong Python foundations and hands-on experience building
machine learning models and LLM-powered applications. Skilled in RAG pipelines,
fine-tuning transformer models, and AI-powered assistants.

SKILLS
Python, JavaScript, Java, NumPy, Pandas, Scikit-learn, Matplotlib, PyTorch,
Hugging Face Transformers, LangChain, RAG, FAISS, LoRA/PEFT, FastAPI, Flask,
Streamlit, MongoDB, SQL, Git, GitHub, Machine Learning, NLP, Deep Learning

EDUCATION
Bachelor of Computer Applications (BCA) — CGPA 8.79
KLE Society JT BCA College, Gadag | 2023–2026

INTERNSHIP
Software Developer Intern — Atronz Innovations
• Built REST APIs using Node.js and Express.js
• Managed MongoDB workflows

PROJECTS
AI Document Q&A Assistant (RAG + LangChain) | Python, LangChain, FAISS, Streamlit
• Built RAG system for document Q&A — top-3 retrieval accuracy 0.91
• Semantic search with FAISS vector indexing

Fine-Tuned LLM for Domain FAQ | Python, Hugging Face, PyTorch
• Fine-tuned DistilGPT-2 — BLEU score 0.65 → 0.87
• Built API for model evaluation""",
}

SAMPLE_JDS = {
    "ML Engineer (Senior)": """Senior Machine Learning Engineer — 5+ years
Requirements:
• Python, PyTorch, TensorFlow
• Production ML deployment, MLOps
• Kubernetes, Docker
• AWS or GCP
• NLP or Computer Vision
• FAISS, vector databases
• CI/CD pipelines for ML""",

    "Fresher / Student (like Shambhavi)": """Microsoft Research Intern — AI/ML
Bachelor's Degree in relevant field (in progress accepted)
OR Master's Degree OR equivalent experience.

Preferred: Master's/PhD, academic papers, conference participation.

Responsibilities:
• Analyze and improve performance of advanced algorithms on large-scale datasets
• Implement prototypes of scalable systems in AI applications
• Collaborate with team members on systems from prototyping to production
• Develop solutions for real-world large-scale problems

Skills needed:
Natural Language Processing, Computer Vision, Machine Learning,
Deep Learning, Python, Data Mining, Artificial Intelligence,
Research, Collaboration, Optimization, Data Processing""",
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()
    st.markdown("### 📋 Load Sample")
    role = st.selectbox("Pick a sample", ["— none —"] + list(SAMPLE_RESUMES.keys()))
    if role != "— none —":
        if st.button("Load", use_container_width=True):
            st.session_state["resume_text"] = SAMPLE_RESUMES[role]
            st.session_state["jd_text"]     = SAMPLE_JDS[role]
            st.rerun()
    st.divider()
    st.markdown("""
**8 Analysis Dimensions:**
1. Semantic similarity
2. Skill overlap (200+ skills)
3. Keyword / ATS score
4. Years of experience
5. Seniority level
6. Education & degree
7. Job title match
8. Resume quality
    """)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h1>🧠 AI Resume Analyzer & Job Match System</h1>
  <p>8-dimension analysis · Semantic embeddings · Experience · Education · ATS simulation · Resume quality</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📄 Analyze", "📊 Results", "🏆 Compare", "📈 Dashboard"])

# ─────────────────────────── TAB 1: INPUT ─────────────────────────────────────
with tab1:
    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown('<div class="section-title">📄 Resume</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf","txt"])
        if uploaded:
            try:
                from utils.parser import extract_text_from_pdf, clean_text
                raw = extract_text_from_pdf(uploaded.read())
                st.session_state["resume_text"] = clean_text(raw)
                st.success(f"✅ Extracted {len(st.session_state['resume_text'])} chars from {uploaded.name}")
            except Exception as e:
                st.error(f"Parse error: {e}")

        resume_text = st.text_area("Or paste resume text",
            value=st.session_state.get("resume_text",""), height=340,
            placeholder="Paste resume here or upload PDF above...", key="resume_input")
        if resume_text:
            st.session_state["resume_text"] = resume_text
        wc = len(resume_text.split())
        ok_r = len(resume_text) > 100
        st.caption(f"{'✅' if ok_r else '⚠️'} {wc} words · {len(resume_text)} chars")

    with col_r:
        st.markdown('<div class="section-title">💼 Job Description</div>', unsafe_allow_html=True)
        jd_text = st.text_area("Paste job description",
            value=st.session_state.get("jd_text",""), height=400,
            placeholder="Paste job description here...", key="jd_input")
        if jd_text:
            st.session_state["jd_text"] = jd_text
        ok_j = len(jd_text) > 100
        st.caption(f"{'✅' if ok_j else '⚠️'} {len(jd_text.split())} words · {len(jd_text)} chars")

    st.divider()
    c1, c2 = st.columns([1,3])
    with c1:
        go = st.button("⚡ Analyze Resume", use_container_width=True, type="primary",
            disabled=not (resume_text and jd_text and len(resume_text)>50 and len(jd_text)>50))
    with c2:
        st.info("💡 First run downloads the embedding model (~90 MB). Later runs take ~3–5 s.")

    if go:
        from utils.parser import parse_resume_text
        from utils.skill_extractor import extract_skills, extract_skills_from_sections
        from utils.similarity import compute_match_score, score_sections, generate_recommendations

        with st.spinner("Running 8-dimension analysis…"):
            prog = st.progress(0, "Parsing resume…")
            sections = parse_resume_text(st.session_state["resume_text"])
            prog.progress(15, "Extracting skills…")
            resume_sr = extract_skills_from_sections(sections)
            jd_sr     = extract_skills(st.session_state["jd_text"])
            prog.progress(35, "Generating embeddings…")
            t0 = time.perf_counter()
            match = compute_match_score(
                resume_text=st.session_state["resume_text"],
                jd_text=st.session_state["jd_text"],
                resume_skills=resume_sr["all_skills"],
                jd_skills=jd_sr["all_skills"],
                sections=sections,
            )
            elapsed = time.perf_counter() - t0
            prog.progress(75, "Scoring sections…")
            sec_scores = score_sections(sections, st.session_state["jd_text"])
            prog.progress(88, "Generating recommendations…")
            recs = generate_recommendations(
                missing_skills=match["missing_skills"],
                matched_skills=match["matched_skills"],
                overall_score=match["overall_score"],
                sections=sections,
                jd_text=st.session_state["jd_text"],
                experience_analysis=match.get("experience_analysis"),
                education_analysis=match.get("education_analysis"),
                title_analysis=match.get("title_analysis"),
                ats_analysis=match.get("ats_analysis"),
                quality_analysis=match.get("quality_analysis"),
            )
            prog.progress(100, "Done!")
            time.sleep(0.3); prog.empty()

        st.session_state["result"] = {
            "match": match, "resume_skills": resume_sr, "jd_skills": jd_sr,
            "sections": sections, "section_scores": sec_scores,
            "recommendations": recs, "elapsed": elapsed,
        }
        if "history" not in st.session_state:
            st.session_state["history"] = []
        contact = sections.get("contact", {})
        st.session_state["history"].append({
            "name": contact.get("name", f"Candidate {len(st.session_state['history'])+1}"),
            "score": match["overall_score"], "label": match["label"],
            "matched": len(match["matched_skills"]),
            "missing": len(match["missing_skills"]),
            "resume_skills_count": len(resume_sr["all_skills"]),
            "exp_years": match.get("experience_analysis",{}).get("candidate",{}).get("total_years",0),
            "edu_degree": match.get("education_analysis",{}).get("candidate",{}).get("highest_degree_label","?"),
        })
        st.success(f"✅ Done in {elapsed:.1f}s — switch to **Results** tab")

# ─────────────────────────── TAB 2: RESULTS ───────────────────────────────────
with tab2:
    result = st.session_state.get("result")
    if not result:
        st.info("Run an analysis in the **Analyze** tab first.")
        st.stop()

    match      = result["match"]
    recs       = result["recommendations"]
    sec_scores = result["section_scores"]
    sections   = result["sections"]
    resume_sr  = result["resume_skills"]
    jd_sr      = result["jd_skills"]
    score      = match["overall_score"]
    sc         = score_color(score)

    # ── Top score row ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🎯 Overall Match Score</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="score-big"><div class="num" style="color:{sc}">{score}</div>'
                    f'<div class="lbl">{match["label"]}</div></div>', unsafe_allow_html=True)
    for col, key, label, color in [
        (c2, "semantic_score",  "Semantic",  "#7c3aed"),
        (c3, "skill_score",     "Skills",    "#2563eb"),
        (c4, "ats_score",       "ATS",       "#d97706"),
    ]:
        v = match[key]
        with col:
            st.markdown(f'<div class="sub-score"><div class="num" style="color:{color}">'
                        f'{v:.0f}</div><div class="lbl">{label}</div></div>', unsafe_allow_html=True)

    # Verdict banner
    vc = verdict_class(score)
    verdicts = {
        "verdict-green": "🟢 Strong match — polish and apply with confidence.",
        "verdict-amber": "🟡 Moderate match — bridge a few gaps before applying.",
        "verdict-red":   "🔴 Weak match — significant gaps to address.",
    }
    st.markdown(f'<div class="verdict-box {vc}">{verdicts[vc]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">📊 All 8 Dimensions</div>', unsafe_allow_html=True)
    dims = [
        ("Semantic Similarity", match["semantic_score"],  "#7c3aed", "Overall meaning alignment between resume & JD"),
        ("Skill Overlap",        match["skill_score"],    "#2563eb", "% of JD's required skills found in your resume"),
        ("Keyword Density",      match["keyword_score"],  "#059669", "JD-specific keywords present in resume text"),
        ("Experience Match",     match["experience_score"],"#d97706","Years of experience + seniority level match"),
        ("Education Match",      match["education_score"],"#db2777", "Degree level and field of study relevance"),
        ("Job Title Match",      match["title_score"],    "#0891b2", "Your job title vs JD target role alignment"),
        ("ATS Score",            match["ats_score"],      "#ea580c", "Simulated Applicant Tracking System filter score"),
        ("Resume Quality",       result["match"].get("quality_analysis",{}).get("quality_score",70),
                                                          "#65a30d", "Format, metrics, action verbs, structure"),
    ]
    col_a, col_b = st.columns(2)
    for i, (label, val, color, tip) in enumerate(dims):
        with (col_a if i % 2 == 0 else col_b):
            st.markdown(f"""
            <div class="dim-row">
              <div class="dlabel">
                <span>{label} <span class="dtip">— {tip}</span></span>
                <span style="color:{color};font-size:0.95rem">{val:.0f}</span>
              </div>
              {bar(val, color)}
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Experience + Education ────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔍 Deep Analysis</div>', unsafe_allow_html=True)
    col_exp, col_edu = st.columns(2)

    with col_exp:
        st.markdown("**💼 Experience**")
        exp_a = match.get("experience_analysis")
        if exp_a:
            cand = exp_a["candidate"]
            req  = exp_a["required"]
            gap  = exp_a["gap_analysis"]
            rows = [
                ("Your experience", f"{cand['total_years']:.0f} years"),
                ("Your seniority",  cand['seniority_inferred']),
                ("Required",        f"{req['min_years']}+ yrs ({req['seniority_required']})"),
                ("Years verdict",   gap['years_verdict']),
                ("Seniority",       gap['seniority_verdict']),
            ]
            if cand.get("job_titles"):
                rows.insert(2, ("Detected titles", " | ".join(cand["job_titles"][:2])))
            rows_html = "".join(
                f'<div class="row"><span class="key">{k}</span>'
                f'<span class="val">{v}</span></div>'
                for k,v in rows
            )
            st.markdown(f'<div class="deep-card">{rows_html}</div>', unsafe_allow_html=True)
        else:
            st.info("No date ranges detected. Add employment dates like '2022–Present'.")

    with col_edu:
        st.markdown("**🎓 Education**")
        edu_a = match.get("education_analysis")
        if edu_a:
            cand = edu_a["candidate"]
            req  = edu_a["required"]
            gap  = edu_a["gap_analysis"]
            rows = [
                ("Your degree",    cand['highest_degree_label']),
                ("Field of study", cand['field_of_study']),
                ("Field relevance",cand['field_relevance_label']),
                ("Required degree",req['min_degree_label']),
                ("Verdict",        gap['degree_verdict']),
            ]
            if cand.get("institutions"):
                rows.append(("Institution", cand["institutions"][0][:45]))
            rows_html = "".join(
                f'<div class="row"><span class="key">{k}</span>'
                f'<span class="val">{v}</span></div>'
                for k,v in rows
            )
            st.markdown(f'<div class="deep-card">{rows_html}</div>', unsafe_allow_html=True)

    # ── Title + ATS ────────────────────────────────────────────────────────────
    col_ti, col_at = st.columns(2)
    with col_ti:
        st.markdown("**🏷️ Title & Domain**")
        ta = match.get("title_analysis")
        if ta:
            dm_ok = ta["domain_match"]
            rows = [
                ("JD target role",    ta['jd_title'][:50]),
                ("Your best title",   ta['best_candidate_title']),
                ("JD domain",         ta['jd_domain']),
                ("Your domain",       ta['candidate_domain']),
                ("Match verdict",     ta['verdict'][:80]),
            ]
            rows_html = "".join(
                f'<div class="row"><span class="key">{k}</span>'
                f'<span class="val">{v}</span></div>'
                for k,v in rows
            )
            st.markdown(f'<div class="deep-card">{rows_html}</div>', unsafe_allow_html=True)

    with col_at:
        st.markdown("**🤖 ATS Simulation**")
        ats = match.get("ats_analysis")
        if ats:
            rows = [
                ("ATS Score",       f"{ats['ats_score']:.0f}/100"),
                ("Matched phrases", f"{ats['matched_count']} / {ats['total_jd_phrases']}"),
                ("Verdict",         ats['verdict'][:70]),
            ]
            rows_html = "".join(
                f'<div class="row"><span class="key">{k}</span>'
                f'<span class="val">{v}</span></div>'
                for k,v in rows
            )
            st.markdown(f'<div class="deep-card">{rows_html}</div>', unsafe_allow_html=True)
            if ats.get("impact_words_found"):
                chips = " ".join(f'<span class="chip-g">{w}</span>' for w in ats["impact_words_found"])
                st.markdown(f'<small style="color:#555">Impact words found:</small><br>'
                            f'<div class="chip-wrap">{chips}</div>', unsafe_allow_html=True)
            if ats.get("missing_phrases"):
                top = [p for p in ats["missing_phrases"][:8]
                       if len(p) > 4 and p not in ("ahead","allow","along","areas","begin","about")]
                if top:
                    st.caption("Key missing phrases: " + " · ".join(f'"{p}"' for p in top[:5]))

    st.divider()

    # ── Skills ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🛠️ Skill Gap Analysis</div>', unsafe_allow_html=True)
    col_m, col_x = st.columns(2)
    with col_m:
        st.markdown(f"**✅ Matched Skills — {len(match['matched_skills'])}**")
        if match["matched_skills"]:
            chips = " ".join(f'<span class="chip-g">✓ {s}</span>' for s in match["matched_skills"])
            st.markdown(f'<div class="chip-wrap">{chips}</div>', unsafe_allow_html=True)
        else:
            st.warning("No skill matches. Check that your resume lists skills explicitly.")
    with col_x:
        st.markdown(f"**❌ Missing Skills — {len(match['missing_skills'])}**")
        if match["missing_skills"]:
            chips = " ".join(f'<span class="chip-r">✗ {s}</span>' for s in match["missing_skills"])
            st.markdown(f'<div class="chip-wrap">{chips}</div>', unsafe_allow_html=True)
        else:
            st.success("🎉 You have all required skills!")
    if match.get("extra_skills"):
        with st.expander(f"📦 Your extra skills not in JD ({len(match['extra_skills'])})"):
            chips = " ".join(f'<span class="chip-b">{s}</span>' for s in match["extra_skills"][:30])
            st.markdown(f'<div class="chip-wrap">{chips}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Resume quality ─────────────────────────────────────────────────────────
    qa = match.get("quality_analysis")
    if qa:
        st.markdown(f'<div class="section-title">📋 Resume Quality — {qa["quality_score"]}/100</div>',
                    unsafe_allow_html=True)
        cp, cf = st.columns(2)
        with cp:
            for p in qa.get("passes",[]):
                st.markdown(f'<div class="pass-item">{p}</div>', unsafe_allow_html=True)
        with cf:
            for issue in qa.get("issues",[]):
                css = "fail-item" if "❌" in issue else "warn-item"
                st.markdown(f'<div class="{css}">{issue}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Section relevance + recommendations ───────────────────────────────────
    col_s, col_rec = st.columns([1,1], gap="large")
    with col_s:
        st.markdown('<div class="section-title">📂 Section Relevance</div>', unsafe_allow_html=True)
        for sec in sec_scores:
            bc = score_color(sec["score"])
            st.markdown(f"""
            <div style="margin-bottom:13px;">
              <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span style="font-size:0.88rem;font-weight:700;color:#2d2d44">{sec['section']}</span>
                <span style="font-size:0.84rem;color:{bc};font-weight:600">{sec['score']:.0f}% — {sec['label']}</span>
              </div>
              <div class="sec-bar-bg">
                <div style="width:{sec['score']}%;background:{bc};height:8px;border-radius:4px;"></div>
              </div>
              <div style="font-size:0.75rem;color:#888;margin-top:3px;line-height:1.4;">{sec['preview']}</div>
            </div>""", unsafe_allow_html=True)

    with col_rec:
        st.markdown('<div class="section-title">💡 Recommendations</div>', unsafe_allow_html=True)
        p_colors = {"high":"#c0392b","medium":"#d68910","low":"#1a6640"}
        for rec in recs[:8]:
            pc = p_colors.get(rec.get("priority","medium"),"#666")
            bc_rec = {"high":"#dc2626","medium":"#d97706","low":"#16a34a"}.get(rec.get("priority"),"#6c5ce7")
            res_html = f'<div class="rec-res">📚 {rec["resource"]}</div>' if rec.get("resource") else ""
            st.markdown(f"""
            <div class="rec-card" style="border-left-color:{bc_rec}">
              <div class="rec-type">{rec['type']}
                <span style="color:{pc}"> ● {rec.get('priority','medium').upper()}</span>
              </div>
              <div class="rec-text">{rec['text']}</div>
              {res_html}
            </div>""", unsafe_allow_html=True)

# ─────────────────────────── TAB 3: COMPARE ───────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">🏆 Multi-Candidate Comparison</div>', unsafe_allow_html=True)
    history = st.session_state.get("history",[])
    if len(history) < 2:
        st.info("Analyze 2+ resumes to compare candidates side by side.")
        if history:
            h = history[0]
            st.markdown(f"**1 candidate so far:** {h['name']} — Score: **{h['score']}** ({h['label']})")
    else:
        from utils.similarity import rank_candidates
        ranked = rank_candidates([dict(h) for h in history])
        medals = {1:"🥇",2:"🥈",3:"🥉"}
        for c in ranked:
            sc2 = c["score"]
            bc = score_color(sc2)
            medal = medals.get(c["rank"],f"#{c['rank']}")
            st.markdown(f"""
            <div style="background:#fff;border:1px solid #e0e0e0;border-radius:11px;
              padding:1rem 1.2rem;margin-bottom:0.6rem;display:flex;align-items:center;
              gap:1rem;box-shadow:0 1px 6px rgba(0,0,0,0.05);">
              <div style="font-size:1.5rem">{medal}</div>
              <div style="flex:1">
                <div style="font-weight:700;font-size:1rem;color:#1a1a2e">{c['name']}</div>
                <div style="font-size:0.8rem;color:#666">
                  {c['matched']} matched · {c['missing']} missing ·
                  {c['resume_skills_count']} skills ·
                  {c.get('exp_years',0):.0f} yrs exp ·
                  {c.get('edu_degree','?')}
                </div>
              </div>
              <div style="text-align:right">
                <div style="font-size:1.9rem;font-weight:900;color:{bc}">{sc2}</div>
                <div style="font-size:0.75rem;color:#666">{c['label']}</div>
              </div>
            </div>""", unsafe_allow_html=True)
        if st.button("🗑️ Clear history"):
            st.session_state["history"] = []
            st.rerun()

# ─────────────────────────── TAB 4: DASHBOARD ─────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">📈 Analytics Dashboard</div>', unsafe_allow_html=True)
    history = st.session_state.get("history",[])
    result  = st.session_state.get("result")
    if not result and not history:
        st.info("Run an analysis to populate this dashboard.")
    else:
        total = len(history)
        avg   = sum(h["score"] for h in history)/total if total else 0
        best  = max((h["score"] for h in history), default=0)
        skills_n = len(result["resume_skills"]["all_skills"]) if result else 0
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Analyses Run", total)
        c2.metric("Avg Match Score", f"{avg:.0f}")
        c3.metric("Best Score", best)
        c4.metric("Skills Tracked", skills_n)

        if result:
            st.divider()
            m = result["match"]
            cl, cr = st.columns(2)
            with cl:
                st.markdown("**Score Breakdown**")
                breakdown = [
                    ("Semantic",    m["semantic_score"],   "#7c3aed"),
                    ("Skills",      m["skill_score"],       "#2563eb"),
                    ("Keywords",    m["keyword_score"],     "#059669"),
                    ("Experience",  m["experience_score"],  "#d97706"),
                    ("Education",   m["education_score"],   "#db2777"),
                    ("Title",       m["title_score"],       "#0891b2"),
                    ("ATS",         m["ats_score"],         "#ea580c"),
                ]
                for label, val, color in breakdown:
                    st.markdown(f"""
                    <div style="margin-bottom:8px;">
                      <div style="display:flex;justify-content:space-between;font-size:0.83rem;
                                  font-weight:600;color:#2d2d44;margin-bottom:3px;">
                        <span>{label}</span><span style="color:{color}">{val:.1f}</span>
                      </div>
                      <div style="background:#eef0f7;border-radius:4px;height:7px;overflow:hidden;">
                        <div style="width:{val}%;background:{color};height:100%;border-radius:4px;"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)
            with cr:
                st.markdown("**Skill Categories in Resume**")
                by_cat = result["resume_skills"].get("by_category",{})
                for cat, skills in sorted(by_cat.items(), key=lambda x: -len(x[1])):
                    pct = min(100, len(skills)*12)
                    st.markdown(f"""
                    <div style="margin-bottom:8px;">
                      <div style="display:flex;justify-content:space-between;font-size:0.83rem;
                                  font-weight:600;color:#2d2d44;margin-bottom:3px;">
                        <span>{cat.replace('_',' ').title()}</span>
                        <span style="color:#7c3aed">{len(skills)}</span>
                      </div>
                      <div style="background:#eef0f7;border-radius:4px;height:7px;overflow:hidden;">
                        <div style="width:{pct}%;background:linear-gradient(90deg,#6c5ce7,#a29bfe);
                                    height:100%;border-radius:4px;"></div>
                      </div>
                    </div>""", unsafe_allow_html=True)
        if total >= 2:
            st.divider()
            st.markdown("**Score History**")
            import pandas as pd
            df = pd.DataFrame(history)
            st.bar_chart(df.set_index("name")["score"])