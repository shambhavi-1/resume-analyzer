"""
utils/education_analyzer.py — degree detection with full Indian education support
"""
import re

try:
    from loguru import logger
except ImportError:
    import logging; logger = logging.getLogger(__name__)

DEGREE_LEVELS = {
    # Below bachelor
    "high school": 0, "sslc": 0, "hsc": 0, "secondary": 0,
    "diploma": 0, "polytechnic": 0, "pu": 0, "puc": 0, "12th": 0, "10th": 0,
    # Associate
    "associate": 1,
    # Bachelor — international
    "bachelor": 2, "b.s.": 2, "b.e.": 2, "b.tech": 2, "btech": 2,
    "b.sc": 2, "bsc": 2, "b.a.": 2, "b.eng": 2, "bs ": 2, "ba ": 2,
    "undergraduate": 2,
    # Bachelor — Indian system
    "bca": 2, "b.c.a": 2, "b.c.a.": 2,
    "bcom": 2, "b.com": 2,
    "bba": 2, "b.b.a": 2,
    "b.voc": 2, "bvoc": 2,
    "b.arch": 2,
    # Master — international
    "master": 3, "m.s.": 3, "m.e.": 3, "m.tech": 3, "mtech": 3,
    "m.sc": 3, "msc": 3, "m.a.": 3, "mba": 3, "m.eng": 3, "ms ": 3,
    "postgraduate": 3, "graduate": 3,
    # Master — Indian system
    "mca": 3, "m.c.a": 3, "m.c.a.": 3,
    "mcom": 3, "m.com": 3,
    "m.voc": 3,
    # PhD
    "ph.d": 4, "phd": 4, "doctorate": 4, "doctoral": 4, "d.phil": 4,
}

DEGREE_LABEL = {0:"High School / Diploma", 1:"Associate", 2:"Bachelor's", 3:"Master's", 4:"PhD"}

# Indian and international CS/IT relevant fields
RELEVANT_FIELDS = {
    "highly_relevant": [
        # CS / IT degrees
        "computer science", "cs", "computer science and engineering", "cse",
        "computer engineering", "computer technology",
        "software engineering", "software development",
        "information technology", "it", "information technology and engineering",
        # Indian-specific degrees
        "computer applications", "computer application", "bca", "mca",
        "b.c.a", "m.c.a", "computer and information science",
        # AI / ML / Data
        "data science", "machine learning", "artificial intelligence", "ai",
        "data engineering", "data analytics",
        "computational intelligence", "cognitive computing",
        # Electronics / related
        "electronics", "electrical engineering", "electronics and communication",
        "ece", "eee", "electronics and computer science",
        # Others
        "computing", "computational", "cybersecurity", "information security",
        "robotics", "cognitive science", "information systems",
        "management information systems", "mis",
    ],
    "moderately_relevant": [
        "mathematics", "math", "statistics", "applied mathematics",
        "physics", "engineering", "systems engineering",
        "operations research", "bioinformatics", "quantitative",
        "scientific computing", "numerical methods",
        "industrial engineering", "instrumentation",
    ],
    "somewhat_relevant": [
        "business", "management", "finance", "economics",
        "accounting", "psychology", "biology", "chemistry",
        "mechanical engineering", "civil engineering",
    ],
}

_HIGH = set(RELEVANT_FIELDS["highly_relevant"])
_MOD  = set(RELEVANT_FIELDS["moderately_relevant"])
_SOME = set(RELEVANT_FIELDS["somewhat_relevant"])


def _field_relevance(field_text):
    f = field_text.lower()
    # Check longer terms first
    for term in sorted(_HIGH, key=len, reverse=True):
        if term in f:
            return 100, "Highly Relevant"
    for term in sorted(_MOD, key=len, reverse=True):
        if term in f:
            return 70, "Moderately Relevant"
    for term in sorted(_SOME, key=len, reverse=True):
        if term in f:
            return 40, "Somewhat Relevant"
    return 20, "Not Directly Relevant"


def extract_candidate_education(resume_text, sections):
    edu_text = sections.get("education", "") or resume_text
    text_lower = edu_text.lower()

    # ── Degree level ─────────────────────────────────────────────────────────
    highest = -1
    for kw, level in sorted(DEGREE_LEVELS.items(), key=lambda x: (-x[1], -len(x[0]))):
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
            highest = max(highest, level)

    # Special: "Bachelor of Computer Applications" → BCA → level 2
    if re.search(r"bachelor\s+of\s+computer\s+applications?", text_lower):
        highest = max(highest, 2)

    if highest == -1:
        if any(w in text_lower for w in ["university","college","institute","school"]):
            highest = 2
        else:
            highest = 0

    # ── Field of study ────────────────────────────────────────────────────────
    field = "Unknown"
    score, flabel = 20, "Not Detected"

    # Pattern matching: "Bachelor of X" / "B.Tech in X" / "degree in X"
    field_pats = [
        r"bachelor\s+of\s+([A-Za-z\s&()/]+?)(?:\n|,|\(|\d|from|at|–|—|-|\|)",
        r"(?:b\.?s\.?|b\.?e\.?|b\.?tech|btech|bca|mca|m\.?tech|m\.?s\.?|master)\s+(?:of\s+|in\s+)([A-Za-z\s&]+?)(?:\n|,|\(|\d|from|at|–|—|-|\|)",
        r"(?:degree|major|majoring)\s+in\s+([A-Za-z\s&]+?)(?:\n|,|\(|\d)",
        r"specializ(?:ation|ing)\s+in\s+([A-Za-z\s&,]+?)(?:\n|,|\(|\d)",
    ]
    for pat in field_pats:
        m = re.search(pat, edu_text, re.IGNORECASE)
        if m:
            raw = m.group(1).strip().rstrip(",;— ")
            if 2 < len(raw) < 80:
                field = raw
                score, flabel = _field_relevance(field)
                break

    # If still unknown, scan for known field terms
    if field == "Unknown" or score == 20:
        for term in sorted(_HIGH, key=len, reverse=True):
            if term in text_lower:
                field = term.title()
                score, flabel = 100, "Highly Relevant"
                break
        if score == 20:
            for term in sorted(_MOD, key=len, reverse=True):
                if term in text_lower:
                    field = term.title()
                    score, flabel = 70, "Moderately Relevant"
                    break

    # Institutions
    insts = []
    for m in re.finditer(r"([A-Z][A-Za-z\s']+(?:University|College|Institute|School|Academy)[A-Za-z\s,]*)", edu_text):
        inst = m.group(0).strip()[:80]
        if inst not in insts:
            insts.append(inst)

    # Graduation years
    gyears = sorted(set(int(y) for y in re.findall(r"\b(20\d{2}|19\d{2})\b", edu_text)
                        if 1990 <= int(y) <= 2030), reverse=True)

    return {
        "highest_degree_level": highest,
        "highest_degree_label": DEGREE_LABEL.get(highest, "Unknown"),
        "field_of_study": field,
        "field_relevance_score": score,
        "field_relevance_label": flabel,
        "institutions": insts[:3],
        "graduation_years": gyears[:3],
        "raw_education_text": edu_text[:400],
    }


def extract_required_education(jd_text):
    text_lower = jd_text.lower()
    min_level = 0
    pref_level = 0
    edu_required = False

    if any(k in text_lower for k in ["degree","bachelor","master","phd","education","university","graduate"]):
        edu_required = True

    for kw, level in sorted(DEGREE_LEVELS.items(), key=lambda x: -x[1]):
        if not re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
            continue
        idx = text_lower.find(kw)
        ctx = text_lower[max(0, idx-80):idx+80]
        if any(w in ctx for w in ["required","must","minimum","need"]):
            min_level = max(min_level, level)
            edu_required = True
        elif any(w in ctx for w in ["preferred","prefer","nice","plus","ideally","bonus"]):
            pref_level = max(pref_level, level)
        else:
            min_level = max(min_level, level)

    # "in progress accepted" = bachelor's is ok even if incomplete
    if re.search(r"in\s*progress|currently\s+pursuing|pursuing", text_lower):
        min_level = min(min_level, 2)

    req_fields = []
    for f in _HIGH:
        if f in text_lower:
            req_fields.append(f.title())

    return {
        "min_degree_level": min_level,
        "min_degree_label": DEGREE_LABEL.get(min_level, "Not specified"),
        "preferred_degree_level": pref_level,
        "preferred_degree_label": DEGREE_LABEL.get(pref_level, "Not specified"),
        "required_fields": req_fields[:5],
        "education_required": edu_required,
    }


def analyze_education_gap(candidate, required):
    cand_level  = candidate["highest_degree_level"]
    req_level   = required["min_degree_level"]
    field_score = candidate["field_relevance_score"]
    edu_req     = required["education_required"]

    deg_gap = req_level - cand_level
    if not edu_req:
        deg_score = 100.0
        deg_verdict = "✅ No specific degree requirement"
    elif deg_gap <= 0:
        deg_score = 100.0
        deg_verdict = f"✅ {candidate['highest_degree_label']} meets {required['min_degree_label']} requirement"
    elif deg_gap == 1:
        deg_score = 65.0
        deg_verdict = f"🟡 Have {candidate['highest_degree_label']}, {required['min_degree_label']} preferred"
    else:
        deg_score = 25.0
        deg_verdict = f"❌ Have {candidate['highest_degree_label']}, need {required['min_degree_label']}"

    combined = round(deg_score * 0.6 + field_score * 0.4, 1)
    return {
        "degree_score": deg_score,
        "field_score": field_score,
        "combined_score": combined,
        "degree_verdict": deg_verdict,
        "field_verdict": f"{candidate['field_relevance_label']}: {candidate['field_of_study']}",
        "candidate_degree": candidate["highest_degree_label"],
        "required_degree": required["min_degree_label"],
        "candidate_field": candidate["field_of_study"],
        "degree_gap": max(0, deg_gap),
        "education_required": edu_req,
    }