"""
api/main.py
────────────
FastAPI backend for the AI Resume Analyzer.

Endpoints:
  POST /upload_resume   — upload & parse a PDF
  POST /analyze_resume  — full analysis (skills + score + suggestions)
  POST /job_match       — quick match score from pre-parsed data
  GET  /health          — health check
"""

import io
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

# Add project root to path so utils is importable when running from api/
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.parser import parse_resume_text, extract_text_from_pdf, clean_text
from utils.skill_extractor import extract_skills, extract_skills_from_sections
from utils.similarity import compute_match_score, score_sections, generate_recommendations


# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Resume Analyzer API",
    description=(
        "NLP-powered resume analysis system using Sentence Transformers, "
        "FAISS vector search, and spaCy skill extraction."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (production: replace with Redis)
_sessions: dict[str, dict] = {}


# ── Pydantic Models ───────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    resume_text: str = Field(..., min_length=50, description="Resume plain text")
    job_description: str = Field(..., min_length=50, description="Job description text")
    session_id: Optional[str] = Field(None, description="Optional session ID from /upload_resume")


class JobMatchRequest(BaseModel):
    resume_skills: list[str] = Field(..., description="List of skills extracted from resume")
    jd_skills: list[str] = Field(..., description="List of skills from job description")
    resume_text: str = Field(..., description="Full resume text for semantic scoring")
    job_description: str = Field(..., description="Full job description text")


class ParsedResumeResponse(BaseModel):
    session_id: str
    contact: dict
    sections: dict
    skills: list[str]
    skills_by_category: dict
    word_count: int
    processing_time_ms: float


class AnalysisResponse(BaseModel):
    session_id: str
    overall_score: int
    label: str
    semantic_score: float
    skill_score: float
    keyword_score: float
    matched_skills: list[str]
    missing_skills: list[str]
    extra_skills: list[str]
    resume_skills: list[str]
    jd_skills: list[str]
    section_relevance: list[dict]
    recommendations: list[dict]
    processing_time_ms: float


class JobMatchResponse(BaseModel):
    overall_score: int
    label: str
    matched_skills: list[str]
    missing_skills: list[str]
    skill_match_pct: float
    semantic_score: float
    keyword_score: float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health_check():
    """Liveness + readiness probe."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "active_sessions": len(_sessions),
    }


@app.post(
    "/upload_resume",
    response_model=ParsedResumeResponse,
    tags=["Step 1 — Parse"],
    summary="Upload and parse a PDF resume",
)
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload a PDF resume file.
    - Extracts text with pdfplumber (PyPDF2 fallback)
    - Cleans and structures into sections
    - Extracts skills with spaCy + NLP
    - Returns a session_id for downstream calls
    """
    t0 = time.perf_counter()

    if not file.filename.lower().endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF and TXT files are supported.",
        )

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10 MB
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")

    try:
        if file.filename.lower().endswith(".pdf"):
            raw_text = extract_text_from_pdf(content)
        else:
            raw_text = content.decode("utf-8", errors="replace")
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    sections = parse_resume_text(raw_text)
    skill_result = extract_skills_from_sections(sections)

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "sections": sections,
        "skills": skill_result,
        "raw_text": raw_text,
        "filename": file.filename,
    }

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"Parsed '{file.filename}' in {elapsed:.0f}ms — session {session_id}")

    # Return only non-empty sections for cleaner response
    sections_out = {
        k: v for k, v in sections.items()
        if isinstance(v, str) and v.strip() and k != "full_text"
    }
    sections_out["contact"] = sections.get("contact", {})

    return ParsedResumeResponse(
        session_id=session_id,
        contact=sections.get("contact", {}),
        sections=sections_out,
        skills=skill_result["all_skills"],
        skills_by_category=skill_result["by_category"],
        word_count=sections.get("word_count", 0),
        processing_time_ms=round(elapsed, 1),
    )


@app.post(
    "/analyze_resume",
    response_model=AnalysisResponse,
    tags=["Step 2 — Analyze"],
    summary="Full analysis: score + skill gap + recommendations",
)
async def analyze_resume(req: AnalyzeRequest):
    """
    Full analysis pipeline:
    1. Parse resume text (or use uploaded session)
    2. Extract skills from resume and JD
    3. Compute semantic similarity with Sentence Transformers + FAISS
    4. Compute composite match score
    5. Score section relevance
    6. Generate improvement recommendations
    """
    t0 = time.perf_counter()
    session_id = req.session_id or str(uuid.uuid4())

    # Use cached session data if available
    if req.session_id and req.session_id in _sessions:
        session = _sessions[req.session_id]
        sections = session["sections"]
        resume_skill_result = session["skills"]
    else:
        sections = parse_resume_text(req.resume_text)
        resume_skill_result = extract_skills_from_sections(sections)

    # Extract JD skills
    jd_skill_result = extract_skills(req.job_description)

    resume_skills = resume_skill_result["all_skills"]
    jd_skills = jd_skill_result["all_skills"]

    # Match scoring
    match_result = compute_match_score(
        resume_text=req.resume_text,
        jd_text=req.job_description,
        resume_skills=resume_skills,
        jd_skills=jd_skills,
    )

    # Section relevance
    section_scores = score_sections(sections, req.job_description)

    # Recommendations
    recs = generate_recommendations(
        missing_skills=match_result["missing_skills"],
        matched_skills=match_result["matched_skills"],
        overall_score=match_result["overall_score"],
        sections=sections,
        jd_text=req.job_description,
    )

    # Store in session
    _sessions[session_id] = {
        **_sessions.get(session_id, {}),
        "sections": sections,
        "skills": resume_skill_result,
        "match_result": match_result,
    }

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(f"Analysis complete in {elapsed:.0f}ms — score={match_result['overall_score']}")

    return AnalysisResponse(
        session_id=session_id,
        overall_score=match_result["overall_score"],
        label=match_result["label"],
        semantic_score=match_result["semantic_score"],
        skill_score=match_result["skill_score"],
        keyword_score=match_result["keyword_score"],
        matched_skills=match_result["matched_skills"],
        missing_skills=match_result["missing_skills"],
        extra_skills=match_result.get("extra_skills", []),
        resume_skills=resume_skills,
        jd_skills=jd_skills,
        section_relevance=section_scores,
        recommendations=recs,
        processing_time_ms=round(elapsed, 1),
    )


@app.post(
    "/job_match",
    response_model=JobMatchResponse,
    tags=["Step 3 — Match"],
    summary="Quick job match score from skill lists",
)
async def job_match(req: JobMatchRequest):
    """
    Lightweight endpoint: compute match score from pre-extracted skill lists.
    Ideal for batch processing multiple JDs against a single resume.
    """
    t0 = time.perf_counter()

    result = compute_match_score(
        resume_text=req.resume_text,
        jd_text=req.job_description,
        resume_skills=req.resume_skills,
        jd_skills=req.jd_skills,
    )

    elapsed = (time.perf_counter() - t0) * 1000
    logger.debug(f"/job_match processed in {elapsed:.0f}ms")

    return JobMatchResponse(
        overall_score=result["overall_score"],
        label=result["label"],
        matched_skills=result["matched_skills"],
        missing_skills=result["missing_skills"],
        skill_match_pct=result["skill_match_pct"],
        semantic_score=result["semantic_score"],
        keyword_score=result["keyword_score"],
    )


@app.get("/sessions/{session_id}", tags=["Meta"])
def get_session(session_id: str):
    """Retrieve a stored analysis session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = _sessions[session_id]
    # Don't return full text in the listing
    return {
        "session_id": session_id,
        "skills_count": len(session.get("skills", {}).get("all_skills", [])),
        "has_match_result": "match_result" in session,
    }


# ── Run (dev mode) ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
