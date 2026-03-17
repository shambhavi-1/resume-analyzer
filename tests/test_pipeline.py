"""
tests/test_pipeline.py
───────────────────────
End-to-end pipeline tests. Run with:
    python tests/test_pipeline.py
or
    pytest tests/test_pipeline.py -v
"""

import sys
import time
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Colors for console output ────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def section(title: str):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")


def ok(msg: str):
    print(f"  {GREEN}✓ {msg}{RESET}")


def warn(msg: str):
    print(f"  {YELLOW}⚠ {msg}{RESET}")


def fail(msg: str):
    print(f"  {RED}✗ {msg}{RESET}")


# ── Test data ────────────────────────────────────────────────────────────────
SAMPLE_RESUME = """
Alex Kim | Senior ML Engineer
Email: alex@email.com | GitHub: github.com/alexkim

SKILLS
Python, PyTorch, TensorFlow, scikit-learn, Pandas, NumPy, FastAPI, Docker, AWS,
SQL, PostgreSQL, Redis, Git, NLP, spaCy, FAISS, Hugging Face, MLflow

EXPERIENCE
Senior ML Engineer — DataVision Corp (2021–Present)
• Deployed 8 production ML models serving 15M+ daily users
• Built MLOps pipeline with MLflow, Docker, GitHub Actions CI/CD
• Reduced inference latency by 45% via INT8 quantization

EDUCATION
M.S. Computer Science — Stanford University (2019)

PROJECTS
AutoML Framework: 2,400 GitHub stars
Real-time Fraud Detection: LSTM model, 10K transactions/second
"""

SAMPLE_JD = """
Senior Machine Learning Engineer

Requirements:
• Python, PyTorch, TensorFlow (required)
• Kubernetes, Docker containerization
• MLOps: MLflow, model monitoring
• AWS SageMaker or GCP Vertex AI
• FAISS, vector databases
• NLP or Computer Vision specialization
• CI/CD for ML pipelines
• Spark for large-scale data processing
• LLM fine-tuning (nice to have)
"""

PASS = 0
FAIL = 0


def run_test(name: str, fn):
    global PASS, FAIL
    try:
        result = fn()
        ok(f"{name}: {result}")
        PASS += 1
        return True
    except AssertionError as e:
        fail(f"{name}: ASSERTION FAILED — {e}")
        FAIL += 1
        return False
    except Exception as e:
        fail(f"{name}: ERROR — {e}")
        FAIL += 1
        return False


# ── Test 1: Parser ────────────────────────────────────────────────────────────
def test_parser():
    section("Step 1 — Resume Parser")
    from utils.parser import parse_resume_text, clean_text

    run_test("clean_text removes artifacts", lambda: (
        assert_true("fi" in clean_text("Some \ufb01le"), "ligature fix"),
        "OK"
    )[-1])

    def parse_test():
        result = parse_resume_text(SAMPLE_RESUME)
        assert "full_text" in result, "missing full_text"
        assert result["word_count"] > 50, f"word count too low: {result['word_count']}"
        contact = result.get("contact", {})
        assert contact.get("email") == "alex@email.com", f"email mismatch: {contact.get('email')}"
        assert contact.get("name") == "Alex Kim", f"name mismatch: {contact.get('name')}"
        return f"word_count={result['word_count']}, name={contact.get('name')}"

    run_test("parse_resume_text full pipeline", parse_test)


# ── Test 2: Skill Extractor ───────────────────────────────────────────────────
def test_skills():
    section("Step 2 — Skill Extractor (NLP)")
    from utils.skill_extractor import extract_skills, compare_skills

    def skill_extract_test():
        result = extract_skills(SAMPLE_RESUME)
        skills = result["all_skills"]
        assert len(skills) >= 5, f"too few skills: {len(skills)}"
        # Must find obvious ones
        skill_lower = {s.lower() for s in skills}
        for expected in ["python", "pytorch", "tensorflow", "docker", "fastapi"]:
            assert expected in skill_lower, f"missing expected skill: {expected}"
        return f"{len(skills)} skills extracted"

    run_test("extract_skills from resume", skill_extract_test)

    def compare_test():
        resume_skills = ["Python", "PyTorch", "Docker", "FastAPI", "AWS"]
        jd_skills = ["Python", "PyTorch", "Kubernetes", "TensorFlow", "FastAPI"]
        result = compare_skills(resume_skills, jd_skills)
        assert "Python" in result["matched"], "Python should match"
        assert "Kubernetes" in result["missing"], "Kubernetes should be missing"
        assert result["match_pct"] == 60.0, f"match_pct should be 60, got {result['match_pct']}"
        return f"matched={result['matched']}, missing={result['missing']}"

    run_test("compare_skills accuracy", compare_test)


# ── Test 3: Embeddings ────────────────────────────────────────────────────────
def test_embeddings():
    section("Step 3–4 — Embeddings (Sentence Transformers + FAISS)")
    from utils.embeddings import encode_text, EMBEDDING_DIM, ResumeIndex

    def encode_test():
        t0 = time.perf_counter()
        emb = encode_text("Python machine learning engineer with Docker experience")
        elapsed = time.perf_counter() - t0
        assert emb.shape == (EMBEDDING_DIM,), f"wrong shape: {emb.shape}"
        assert abs(emb.sum()) > 0, "zero embedding"
        # Check L2 norm ≈ 1 (normalized)
        import numpy as np
        norm = np.linalg.norm(emb)
        assert 0.99 < norm < 1.01, f"embedding not normalized: norm={norm}"
        return f"shape=({EMBEDDING_DIM},), norm={norm:.4f}, time={elapsed:.2f}s"

    run_test("encode_text shape + normalization", encode_test)

    def cache_test():
        t0 = time.perf_counter()
        encode_text("cache test text 12345", use_cache=True)
        first = time.perf_counter() - t0
        t1 = time.perf_counter()
        encode_text("cache test text 12345", use_cache=True)
        second = time.perf_counter() - t1
        assert second < first * 0.5, "cache not working"
        return f"first={first*1000:.1f}ms, cached={second*1000:.2f}ms"

    run_test("embedding cache speedup", cache_test)

    def faiss_test():
        idx = ResumeIndex()
        id0 = idx.add("Python machine learning engineer", {"name": "Alice"})
        id1 = idx.add("Java backend software engineer", {"name": "Bob"})
        id2 = idx.add("Data scientist with Python and statistics", {"name": "Charlie"})
        assert idx.size == 3, f"expected 3 entries, got {idx.size}"
        results = idx.search("Python ML engineer", k=2)
        assert len(results) == 2
        top_name = results[0]["metadata"]["name"]
        assert top_name in ("Alice", "Charlie"), f"unexpected top result: {top_name}"
        return f"top match: {top_name} (score={results[0]['score']:.4f})"

    run_test("FAISS index add + search", faiss_test)


# ── Test 4: Similarity Scoring ────────────────────────────────────────────────
def test_similarity():
    section("Step 5–6 — Similarity Scoring")
    from utils.similarity import compute_match_score, generate_recommendations

    def score_test():
        result = compute_match_score(
            resume_text=SAMPLE_RESUME,
            jd_text=SAMPLE_JD,
            resume_skills=["Python", "PyTorch", "Docker", "FastAPI", "AWS", "MLflow"],
            jd_skills=["Python", "PyTorch", "Kubernetes", "AWS", "MLflow", "Spark"],
        )
        assert 0 <= result["overall_score"] <= 100, f"score out of range: {result['overall_score']}"
        assert result["label"] in ("Excellent Match", "Good Match", "Fair Match", "Needs Work")
        assert "matched_skills" in result
        assert "missing_skills" in result
        assert "Kubernetes" in result["missing_skills"], "Kubernetes should be missing"
        return (
            f"score={result['overall_score']} ({result['label']}), "
            f"matched={result['matched_skills']}, missing={result['missing_skills']}"
        )

    run_test("compute_match_score full pipeline", score_test)

    def rec_test():
        recs = generate_recommendations(
            missing_skills=["Kubernetes", "Spark"],
            matched_skills=["Python", "PyTorch"],
            overall_score=55,
            sections={"experience": "helped with some ML tasks"},
            jd_text=SAMPLE_JD,
        )
        assert len(recs) > 0, "no recommendations generated"
        types = {r["type"] for r in recs}
        assert "SKILL GAP" in types, "expected SKILL GAP recommendation"
        return f"{len(recs)} recommendations, types={types}"

    run_test("generate_recommendations produces output", rec_test)


# ── Helper ────────────────────────────────────────────────────────────────────
def assert_true(condition: bool, msg: str = ""):
    if not condition:
        raise AssertionError(msg)
    return condition


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{BOLD}{'='*60}")
    print("  AI Resume Analyzer — Pipeline Test Suite")
    print(f"{'='*60}{RESET}")
    print(f"\n{YELLOW}Note: First run will download the embedding model (~90 MB).{RESET}")

    test_parser()
    test_skills()
    test_embeddings()
    test_similarity()

    # Summary
    total = PASS + FAIL
    section(f"Results: {PASS}/{total} tests passed")
    if FAIL == 0:
        print(f"  {GREEN}{BOLD}All tests passed! ✓{RESET}\n")
    else:
        print(f"  {RED}{BOLD}{FAIL} tests failed.{RESET}\n")
        sys.exit(1)
