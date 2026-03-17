"""
utils/job_title_analyzer.py — title match, ATS simulation, resume quality
"""
import re

try:
    from loguru import logger
except ImportError:
    import logging; logger = logging.getLogger(__name__)

DOMAINS = {
    "machine_learning": ["machine learning","ml engineer","ai engineer","deep learning",
        "data scientist","research scientist","nlp engineer","computer vision",
        "mlops","applied scientist","ai researcher","llm engineer"],
    "software_engineering": ["software engineer","software developer","backend engineer",
        "frontend engineer","full stack","fullstack","sde","platform engineer","sre"],
    "data_engineering": ["data engineer","data architect","analytics engineer",
        "etl developer","pipeline engineer","data platform"],
    "devops_cloud": ["devops","cloud engineer","infrastructure engineer",
        "platform engineer","systems engineer","solutions architect","cloud architect"],
    "data_analytics": ["data analyst","business analyst","analytics","bi developer",
        "reporting analyst","insights analyst"],
    "research": ["research intern","research scientist","researcher",
        "research assistant","phd intern","graduate researcher"],
}

_TITLE_TO_DOMAIN = {}
for _d, _ts in DOMAINS.items():
    for _t in _ts:
        _TITLE_TO_DOMAIN[_t] = _d


def _detect_domain(text):
    text_lower = text.lower()
    counts = {}
    for domain, titles in DOMAINS.items():
        c = sum(1 for t in titles if t in text_lower)
        if c:
            counts[domain] = c
    return max(counts, key=counts.get) if counts else "general"


def analyze_title_match(candidate_titles, jd_text, resume_text):
    jd_lower = jd_text.lower()
    jd_lines = [l.strip() for l in jd_text.strip().split("\n")[:6] if l.strip()]
    jd_title  = jd_lines[0] if jd_lines else "Unknown Role"
    # Strip "Overview" type headings
    if jd_title.lower() in ("overview", "about", "description", "summary", "job description"):
        jd_title = jd_lines[1] if len(jd_lines) > 1 else jd_title

    jd_domain  = _detect_domain(jd_text)
    cand_domain = _detect_domain(" ".join(candidate_titles) + " " + resume_text[:800])

    domain_match  = (jd_domain == cand_domain) or (jd_domain == "general") or (cand_domain == "general")
    domain_score  = 100.0 if domain_match else 55.0

    jd_words = set(re.findall(r"\b[a-z]+\b", jd_title.lower()))
    jd_words -= {"the","a","an","and","or","for","of","in","at","–","-","overview","position","role"}

    best_score = 0.0
    best_title = "Student / Fresher"
    for title in (candidate_titles or []):
        tw = set(re.findall(r"\b[a-z]+\b", title.lower()))
        if not jd_words:
            s = 50.0
        else:
            s = len(tw & jd_words) / len(jd_words) * 100
        if s > best_score:
            best_score = s
            best_title = title

    # If no titles extracted but resume is student, be honest
    if not candidate_titles:
        best_title = "Student / Fresher"
        best_score = 30.0

    combined = round(best_score * 0.5 + domain_score * 0.5, 1)

    if combined >= 75:
        verdict = f"✅ Strong match: {best_title} → {jd_title}"
    elif combined >= 45:
        verdict = f"🟡 Partial match: {best_title} → {jd_title}"
    else:
        verdict = f"⚠️ Different level/domain: {best_title} vs {jd_title}"

    return {
        "jd_title": jd_title,
        "best_candidate_title": best_title,
        "title_score": round(best_score, 1),
        "domain_match": domain_match,
        "jd_domain": jd_domain.replace("_", " ").title(),
        "candidate_domain": cand_domain.replace("_", " ").title(),
        "combined_score": combined,
        "verdict": verdict,
    }


# ── Comprehensive stopword list for ATS ──────────────────────────────────────
_ATS_STOP = {
    # Generic English
    "with","have","this","that","will","from","your","they","them","their",
    "been","were","about","more","than","into","also","such","other","both",
    "each","most","over","some","must","able","well","work","help","build",
    "using","used","team","good","great","strong","years","year","looking",
    "required","require","minimum","prefer","preferred","include","including",
    "responsible","responsibilities","requirements","qualification","qualifications",
    "opportunity","position","join","seeking","hiring","apply","application",
    # Filler / prepositions
    "ahead","allow","along","areas","begin","while","where","every","these",
    "those","which","would","could","should","shall","might","often","when",
    "then","here","there","what","make","many","need","same","onto","upon",
    "across","after","before","between","during","within","without","toward",
    "through","forward","around","below","above","under",
    # Company/culture words (Microsoft JD specific)
    "microsoft","intern","interns","internship","students","student","employees",
    "employee","people","person","world","global","planet","lives","life","time",
    "together","shared","share","values","value","culture","goals","goal",
    "mission","vision","growth","mindset","innovate","integrity","respect",
    "accountability","inclusion","thrive","empower","diverse","diversity",
    "community","organization","company","passion","journey","chance",
    "being","doing","having","getting","making","taking","giving","going",
    "coming","finding","moving","providing","working","helping","aligns",
    "beyond","author","chance","begin","areas","allow","ahead","along",
    # Common verbs too generic
    "analyze","develop","implement","create","design","build","deliver",
    "achieve","ensure","provide","maintain","support","manage","lead",
    # Short words
    "and","the","for","are","but","not","you","all","can","her","was",
    "one","our","out","who","get","may","use","its","said","each",
}

HIGH_VALUE = ["led","built","designed","deployed","implemented","architected",
              "optimized","reduced","increased","launched","scaled","improved",
              "developed","created","mentored","owned","delivered"]


def compute_ats_score(resume_text, jd_text):
    resume_lower = resume_text.lower()
    jd_lower     = jd_text.lower()

    # Extract meaningful technical tokens from JD
    jd_tokens = set()

    # Multi-word technical phrases (2-3 words, min 10 chars total)
    for phrase in re.findall(r"\b([a-z][a-z0-9]+\s+[a-z][a-z0-9]+(?:\s+[a-z][a-z0-9]+)?)\b", jd_lower):
        words = phrase.split()
        if (not all(w in _ATS_STOP for w in words)
                and len(phrase) >= 10
                and not words[0] in _ATS_STOP):
            jd_tokens.add(phrase)

    # Single important technical words (length > 5, not stopword)
    for word in re.findall(r"\b([a-zA-Z][a-zA-Z0-9+#./]{4,})\b", jd_text):
        wl = word.lower()
        if wl not in _ATS_STOP and not wl.isdigit() and len(wl) > 5:
            jd_tokens.add(wl)

    if not jd_tokens:
        return {"ats_score": 50.0, "found_phrases": [], "missing_phrases": [],
                "impact_words_found": [], "total_jd_phrases": 0, "matched_count": 0,
                "verdict": "⚠️ Could not extract phrases from JD"}

    found   = [p for p in jd_tokens if p in resume_lower]
    missing = [p for p in jd_tokens if p not in resume_lower]

    ats_score = len(found) / len(jd_tokens) * 100

    # Filter missing to only show meaningful ones
    meaningful_missing = [
        p for p in sorted(missing, key=len, reverse=True)
        if len(p) > 6
        and not all(w in _ATS_STOP for w in p.split())
        and p not in _ATS_STOP
    ][:10]

    impact_found = [w for w in HIGH_VALUE if w in resume_lower]

    if ats_score >= 60:
        verdict = "✅ Good ATS match — resume language aligns with JD"
    elif ats_score >= 35:
        verdict = "🟡 Moderate ATS match — add more JD-specific language"
    else:
        verdict = "❌ Weak ATS match — resume may be filtered before human review"

    return {
        "ats_score": round(ats_score, 1),
        "found_phrases": sorted(found)[:20],
        "missing_phrases": meaningful_missing,
        "impact_words_found": impact_found,
        "total_jd_phrases": len(jd_tokens),
        "matched_count": len(found),
        "verdict": verdict,
    }


def check_resume_quality(resume_text, sections):
    issues, passes = [], []
    score = 100

    wc = len(resume_text.split())
    if wc < 150:
        issues.append("⚠️ Resume is short (under 150 words) — add more detail to projects/experience")
        score -= 15
    elif wc > 1000:
        issues.append("⚠️ Resume may be too long — aim for 400–700 words for a fresher")
        score -= 5
    else:
        passes.append(f"✅ Good length ({wc} words)")

    contact = sections.get("contact", {})
    if contact.get("email"):
        passes.append("✅ Email found")
    else:
        issues.append("❌ No email detected — add contact info"); score -= 10

    if contact.get("github") or contact.get("linkedin"):
        passes.append("✅ Professional profile links found")
    else:
        issues.append("⚠️ No GitHub/LinkedIn — add professional profile links"); score -= 5

    for sec in ["skills","education"]:
        if sections.get(sec,"").strip():
            passes.append(f"✅ {sec.title()} section present")
        else:
            issues.append(f"❌ Missing {sec.title()} section"); score -= 10

    # Check experience OR internship OR projects
    has_work = bool(sections.get("experience","").strip())
    has_proj = bool(sections.get("projects","").strip())
    if has_work:
        passes.append("✅ Experience/Internship section present")
    elif has_proj:
        passes.append("✅ Projects section present (compensates for limited experience)")
    else:
        issues.append("❌ No Experience or Projects section"); score -= 15

    nums = re.findall(r"\d+[%xX]?|\d+\s*(?:million|billion|thousand|users|ms|seconds|accuracy|score)", resume_text)
    if len(nums) >= 3:
        passes.append(f"✅ Good use of metrics ({len(nums)} numbers found)")
    else:
        issues.append("⚠️ Add quantified results — e.g. 'achieved 91% accuracy', 'served 500 users'")
        score -= 5

    strong = [v for v in ["built","designed","led","developed","implemented","deployed",
                           "created","achieved","improved","fine-tuned"] if v in resume_text.lower()]
    if len(strong) >= 3:
        passes.append(f"✅ Strong action verbs used ({', '.join(strong[:4])})")
    else:
        issues.append("⚠️ Use stronger verbs: Built, Implemented, Designed, Achieved")
        score -= 5

    return {"quality_score": max(0, score), "issues": issues, "passes": passes, "word_count": wc}