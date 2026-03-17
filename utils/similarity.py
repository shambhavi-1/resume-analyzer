"""
utils/similarity.py  (v2 — complete analyzer)
───────────────────────────────────────────────
Master scoring engine combining ALL 8 analysis dimensions:
  1. Semantic similarity    (Sentence Transformers)
  2. Skill overlap          (spaCy + taxonomy)
  3. Keyword density        (regex)
  4. Experience match       (years + seniority)
  5. Education match        (degree + field)
  6. Job title match        (domain + title)
  7. ATS score              (keyword simulation)
  8. Resume quality         (format checks)
"""

import re
from loguru import logger
import numpy as np

from utils.embeddings import encode_text, EMBEDDING_DIM
from utils.skill_extractor import compare_skills

WEIGHTS = {
    "semantic":      0.25,
    "skill_overlap": 0.25,
    "keyword":       0.10,
    "experience":    0.15,
    "education":     0.10,
    "title":         0.10,
    "ats":           0.05,
}

def cosine_similarity(vec_a, vec_b):
    return max(0.0, min(1.0, float(np.dot(vec_a, vec_b))))

def semantic_similarity(text_a, text_b):
    return cosine_similarity(encode_text(text_a), encode_text(text_b))

def keyword_density_score(resume_text, jd_text):
    STOPWORDS = {
        "with","have","this","that","will","from","your","they","them","their",
        "been","were","about","more","than","into","also","such","other","both",
        "each","most","over","some","must","able","well","work","help","build",
        "using","used","team","good","great","strong","years","year","looking",
        "experience","knowledge","understanding","familiar","ability",
    }
    jd_words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9+#.]{2,}\b", jd_text.lower())
    jd_keywords = [w for w in jd_words if w not in STOPWORDS]
    if not jd_keywords:
        return 0.5
    resume_lower = resume_text.lower()
    found = sum(1 for kw in jd_keywords if kw in resume_lower)
    return found / len(jd_keywords)

def compute_match_score(resume_text, jd_text, resume_skills, jd_skills, sections=None):
    sections = sections or {}

    sem_sim = semantic_similarity(resume_text, jd_text)
    semantic_score = sem_sim * 100

    skill_comparison = compare_skills(resume_skills, jd_skills)
    skill_score = skill_comparison["match_pct"]

    kd = keyword_density_score(resume_text, jd_text)
    keyword_score = kd * 100

    experience_result = None
    experience_score = 70.0
    try:
        from utils.experience_analyzer import (
            extract_candidate_experience, extract_required_experience, analyze_experience_gap
        )
        candidate_exp = extract_candidate_experience(resume_text, sections)
        required_exp  = extract_required_experience(jd_text)
        exp_gap        = analyze_experience_gap(candidate_exp, required_exp)
        experience_score = exp_gap["combined_score"]
        experience_result = {"candidate": candidate_exp, "required": required_exp, "gap_analysis": exp_gap}
    except Exception as e:
        logger.warning(f"Experience analysis failed: {e}")

    education_result = None
    education_score = 70.0
    try:
        from utils.education_analyzer import (
            extract_candidate_education, extract_required_education, analyze_education_gap
        )
        candidate_edu = extract_candidate_education(resume_text, sections)
        required_edu  = extract_required_education(jd_text)
        edu_gap        = analyze_education_gap(candidate_edu, required_edu)
        education_score = edu_gap["combined_score"]
        education_result = {"candidate": candidate_edu, "required": required_edu, "gap_analysis": edu_gap}
    except Exception as e:
        logger.warning(f"Education analysis failed: {e}")

    title_result = None
    title_score = 60.0
    try:
        from utils.job_title_analyzer import analyze_title_match
        candidate_titles = _extract_titles_from_text(resume_text)
        title_res  = analyze_title_match(candidate_titles, jd_text, resume_text)
        title_score = title_res["combined_score"]
        title_result = title_res
    except Exception as e:
        logger.warning(f"Title analysis failed: {e}")

    ats_result = None
    ats_score = 50.0
    try:
        from utils.job_title_analyzer import compute_ats_score
        ats_res   = compute_ats_score(resume_text, jd_text)
        ats_score  = ats_res["ats_score"]
        ats_result = ats_res
    except Exception as e:
        logger.warning(f"ATS analysis failed: {e}")

    quality_result = None
    try:
        from utils.job_title_analyzer import check_resume_quality
        quality_result = check_resume_quality(resume_text, sections)
    except Exception as e:
        logger.warning(f"Quality check failed: {e}")

    overall = (
        WEIGHTS["semantic"]      * semantic_score   +
        WEIGHTS["skill_overlap"] * skill_score      +
        WEIGHTS["keyword"]       * keyword_score    +
        WEIGHTS["experience"]    * experience_score +
        WEIGHTS["education"]     * education_score  +
        WEIGHTS["title"]         * title_score      +
        WEIGHTS["ats"]           * ats_score
    )
    overall = round(min(100, max(0, overall)))

    if   overall >= 80: label, label_color = "Excellent Match", "green"
    elif overall >= 65: label, label_color = "Good Match",      "yellow"
    elif overall >= 45: label, label_color = "Fair Match",      "orange"
    else:               label, label_color = "Needs Work",      "red"

    logger.info(
        f"Score — Overall:{overall} Semantic:{semantic_score:.1f} "
        f"Skill:{skill_score:.1f} KW:{keyword_score:.1f} "
        f"Exp:{experience_score:.1f} Edu:{education_score:.1f} "
        f"Title:{title_score:.1f} ATS:{ats_score:.1f}"
    )

    return {
        "overall_score": overall,
        "label": label,
        "label_color": label_color,
        "semantic_score": round(semantic_score, 1),
        "skill_score": round(skill_score, 1),
        "keyword_score": round(keyword_score, 1),
        "experience_score": round(experience_score, 1),
        "education_score": round(education_score, 1),
        "title_score": round(title_score, 1),
        "ats_score": round(ats_score, 1),
        "matched_skills": skill_comparison["matched"],
        "missing_skills": skill_comparison["missing"],
        "extra_skills": skill_comparison.get("extra", []),
        "skill_match_pct": skill_comparison["match_pct"],
        "total_jd_skills": skill_comparison["total_jd_skills"],
        "total_resume_skills": skill_comparison["total_resume_skills"],
        "experience_analysis": experience_result,
        "education_analysis": education_result,
        "title_analysis": title_result,
        "ats_analysis": ats_result,
        "quality_analysis": quality_result,
        "weights": WEIGHTS,
    }

def _extract_titles_from_text(text):
    titles = []
    pat = r"^([A-Z][A-Za-z\s,./]+(?:Engineer|Developer|Scientist|Analyst|Manager|Lead|Architect|Director))\s*[—\-–|@]"
    for line in text.split("\n")[:30]:
        m = re.match(pat, line.strip())
        if m:
            t = m.group(1).strip()
            if 3 < len(t) < 60:
                titles.append(t)
    return titles[:5]

def score_sections(sections, jd_text):
    scored = []
    for key in ["skills","experience","education","projects","summary"]:
        text = sections.get(key,"").strip()
        if not text or len(text) < 20:
            continue
        sim = semantic_similarity(text, jd_text)
        score = round(sim * 100, 1)
        label = "High Relevance" if score >= 70 else "Medium Relevance" if score >= 45 else "Low Relevance"
        scored.append({"section": key.title(), "score": score, "label": label,
                        "preview": text[:120]+("..." if len(text)>120 else "")})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

SKILL_RESOURCES = {
    "docker":        "Docker official docs + 'Docker for Beginners' (docker.com)",
    "kubernetes":    "Kubernetes.io tutorials + CKA exam (Linux Foundation)",
    "tensorflow":    "TensorFlow.org tutorials — TF2 for Beginners",
    "pytorch":       "PyTorch.org tutorials, fast.ai practical deep learning",
    "aws":           "AWS Free Tier + AWS Cloud Practitioner cert",
    "gcp":           "Google Cloud Skills Boost free learning paths",
    "spark":         "Databricks free training (databricks.com/learn)",
    "kafka":         "Confluent Kafka tutorials (confluent.io/learn)",
    "mlflow":        "MLflow quickstart (mlflow.org)",
    "sql":           "Mode Analytics SQL Tutorial + SQLZoo — both free",
    "postgresql":    "PostgreSQL official docs + PgExercises.com",
    "mongodb":       "MongoDB University free courses",
    "fastapi":       "FastAPI official tutorial (fastapi.tiangolo.com)",
    "scikit-learn":  "Scikit-learn user guide + Kaggle ML micro-courses",
    "llm":           "Hugging Face NLP Course (huggingface.co/learn)",
    "terraform":     "HashiCorp Learn (developer.hashicorp.com/terraform)",
    "graphql":       "GraphQL official tutorial (graphql.org/learn)",
    "faiss":         "FAISS GitHub wiki + Meta AI research blog",
}

def generate_recommendations(
    missing_skills, matched_skills, overall_score, sections, jd_text,
    experience_analysis=None, education_analysis=None,
    title_analysis=None, ats_analysis=None, quality_analysis=None,
):
    recommendations = []

    for skill in missing_skills[:8]:
        skill_lower = skill.lower()
        high_p = {"docker","kubernetes","tensorflow","pytorch","aws","gcp","sql","spark","kafka","mlflow","fastapi"}
        recommendations.append({
            "type": "SKILL GAP",
            "priority": "high" if skill_lower in high_p else "medium",
            "text": f"Learn {skill}: Build one hands-on project and push to GitHub. "
                    f"Even a small project demonstrates initiative.",
            "resource": SKILL_RESOURCES.get(skill_lower),
            "skill": skill,
        })

    if experience_analysis:
        gap = experience_analysis.get("gap_analysis", {})
        years_gap = gap.get("years_gap", 0)
        if years_gap >= 2:
            recommendations.append({
                "type": "EXPERIENCE GAP", "priority": "high",
                "text": f"You're ~{years_gap:.0f} year(s) short. Highlight impact over duration: "
                        f"'Deployed 6 production models in 18 months' reads stronger than just listing dates.",
                "resource": None, "skill": None,
            })
        elif years_gap == 1:
            recommendations.append({
                "type": "EXPERIENCE GAP", "priority": "medium",
                "text": "Slightly under requirement. Include internships, freelance, open-source, "
                        "and personal projects — these all count.",
                "resource": None, "skill": None,
            })
        if gap.get("is_overqualified"):
            recommendations.append({
                "type": "OVERQUALIFICATION", "priority": "low",
                "text": "Your experience exceeds the requirement. Address this in your cover letter "
                        "— explain why you're interested in this role specifically.",
                "resource": None, "skill": None,
            })

    if education_analysis:
        edu_gap = education_analysis.get("gap_analysis", {})
        if edu_gap.get("degree_gap", 0) >= 1 and education_analysis.get("required", {}).get("education_required"):
            recommendations.append({
                "type": "EDUCATION GAP", "priority": "medium",
                "text": f"JD prefers {edu_gap.get('required_degree','higher degree')} but you have "
                        f"{edu_gap.get('candidate_degree','lower degree')}. "
                        f"AWS, GCP, or TensorFlow Developer certs compensate effectively.",
                "resource": "coursera.org/professional-certificates — many free to audit",
                "skill": None,
            })
        if edu_gap.get("field_score", 100) < 50:
            recommendations.append({
                "type": "FIELD MISMATCH", "priority": "medium",
                "text": "Your field of study isn't directly relevant. Emphasize self-taught skills, "
                        "projects, and relevant coursework. Many ML engineers are self-taught.",
                "resource": None, "skill": None,
            })

    if ats_analysis:
        missing_phrases = ats_analysis.get("missing_phrases", [])
        if ats_analysis.get("ats_score", 100) < 50 and missing_phrases:
            top = ", ".join(f'"{p}"' for p in missing_phrases[:4])
            recommendations.append({
                "type": "ATS OPTIMIZATION", "priority": "high",
                "text": f"Resume may be filtered before a human reads it. "
                        f"Add these JD phrases naturally: {top}.",
                "resource": "jobscan.co — free ATS checker",
                "skill": None,
            })

    if quality_analysis:
        for issue in quality_analysis.get("issues", [])[:4]:
            if "❌" in issue or "⚠️" in issue:
                recommendations.append({
                    "type": "RESUME QUALITY",
                    "priority": "medium" if "⚠️" in issue else "high",
                    "text": issue.replace("❌ ","").replace("⚠️ ","").replace("🟡 ",""),
                    "resource": None, "skill": None,
                })

    if title_analysis and title_analysis.get("combined_score", 100) < 50:
        recommendations.append({
            "type": "TITLE ALIGNMENT", "priority": "medium",
            "text": f"Your title '{title_analysis.get('best_candidate_title','')}' differs from "
                    f"'{title_analysis.get('jd_title','')}'. Mirror the JD's exact role title "
                    f"in your resume summary for immediate recruiter alignment.",
            "resource": None, "skill": None,
        })

    if overall_score < 40:
        recommendations.append({
            "type": "GENERAL STRATEGY", "priority": "high",
            "text": "Low match. Focus: (1) mirror JD skill keywords exactly, "
                    "(2) rewrite summary targeting this role, "
                    "(3) add projects using required technologies.",
            "resource": None, "skill": None,
        })
    elif overall_score >= 80:
        recommendations.append({
            "type": "STRONG MATCH", "priority": "low",
            "text": "Excellent match! Prep for technical interviews, quantify top 3 achievements, "
                    "and write a tailored cover letter — small differentiators matter at this level.",
            "resource": None, "skill": None,
        })

    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda r: priority_order.get(r.get("priority","medium"), 1))
    return recommendations

def rank_candidates(candidates):
    ranked = sorted(candidates, key=lambda c: c.get("overall_score", 0), reverse=True)
    for i, c in enumerate(ranked):
        c["rank"] = i + 1
    return ranked
