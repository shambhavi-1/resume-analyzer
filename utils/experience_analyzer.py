"""
utils/experience_analyzer.py — years of experience, seniority, gap analysis
"""
import re
from datetime import datetime

try:
    from loguru import logger
except ImportError:
    import logging; logger = logging.getLogger(__name__)

CURRENT_YEAR = datetime.now().year

SENIORITY_LEVELS = {
    "intern": 0, "internship": 0, "trainee": 0,
    "junior": 1, "associate": 1, "entry": 1,
    "mid": 2, "mid-level": 2, "intermediate": 2,
    "senior": 3, "sr.": 3,
    "lead": 4, "staff": 4,
    "principal": 5,
    "director": 6, "head of": 6,
    "vp": 7, "vice president": 7,
}
SENIORITY_LABEL = {0:"Intern/Fresher",1:"Junior",2:"Mid-level",3:"Senior",4:"Lead",5:"Principal",6:"Director",7:"VP+"}
SENIORITY_YEARS = {0:(0,1),1:(0,2),2:(2,5),3:(5,10),4:(7,15),5:(10,20),6:(12,25),7:(15,30)}


def _extract_year_ranges(text):
    ranges = []
    p1 = re.findall(
        r"(20\d{2}|19\d{2})\s*[–—\-/to]+\s*(20\d{2}|19\d{2}|[Pp]resent|[Cc]urrent|[Nn]ow|[Oo]ngoing)",
        text)
    for start, end in p1:
        try:
            s = int(start)
            e = CURRENT_YEAR if end.lower() in ("present","current","now","ongoing") else int(end)
            if 1990 <= s <= CURRENT_YEAR and s <= e <= CURRENT_YEAR + 1:
                ranges.append((s, e))
        except ValueError:
            pass
    months = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    p2 = re.findall(
        rf"{months}[\s/]*(20\d{{2}}|19\d{{2}})\s*[–—\-/to]+\s*(?:{months}[\s/]*)?(20\d{{2}}|19\d{{2}}|[Pp]resent|[Cc]urrent)",
        text)
    for _, start, end in p2:
        try:
            s = int(start)
            e = CURRENT_YEAR if end.lower() in ("present","current") else int(end)
            if 1990 <= s <= CURRENT_YEAR and s <= e <= CURRENT_YEAR + 1:
                ranges.append((s, e))
        except ValueError:
            pass
    return ranges


def _compute_total_years(ranges):
    if not ranges:
        return 0.0
    sorted_r = sorted(ranges, key=lambda x: x[0])
    merged = [list(sorted_r[0])]
    for s, e in sorted_r[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return float(sum(e - s for s, e in merged))


def extract_required_experience(jd_text):
    text_lower = jd_text.lower()
    min_years = 0
    mentions = []

    patterns = [
        r"(\d+)\s*\+\s*years?", r"(\d+)\s*or\s*more\s*years?",
        r"at\s+least\s+(\d+)\s*years?", r"minimum\s+(?:of\s+)?(\d+)\s*years?",
        r"(\d+)\s*[-–]\s*(\d+)\s*years?", r"(\d+)\s*years?\s+(?:of\s+)?(?:experience|exp)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text_lower):
            nums = [int(g) for g in m.groups() if g and g.isdigit()]
            if nums:
                mentions.append(m.group())
                min_years = max(min_years, min(nums))

    # Detect seniority in JD
    seniority_level = 0  # default: open/intern
    seniority_label = "Any"
    for kw, level in sorted(SENIORITY_LEVELS.items(), key=lambda x: -x[1]):
        if kw in text_lower:
            seniority_level = level
            seniority_label = SENIORITY_LABEL[level]
            break

    # Check if it's an intern/fresher JD
    is_entry_level = any(w in text_lower for w in [
        "intern", "internship", "fresher", "entry level", "entry-level",
        "bachelor", "undergraduate", "in progress", "student", "graduate student"
    ])
    if is_entry_level and min_years == 0:
        seniority_level = 0
        seniority_label = "Intern/Fresher"

    return {
        "min_years": min_years,
        "raw_mentions": list(set(mentions)),
        "seniority_required": seniority_label,
        "seniority_level": seniority_level,
        "is_entry_level": is_entry_level,
    }


def extract_candidate_experience(resume_text, sections):
    """
    Detect work experience from EXPERIENCE/INTERNSHIP sections only.
    Does NOT count education year ranges.
    """
    # Only look at experience section (which now includes INTERNSHIP thanks to parser fix)
    exp_section = sections.get("experience", "").strip()

    # Scan full text for internship blocks if experience section is empty
    if not exp_section:
        lines = resume_text.split("\n")
        capture = False
        collected = []
        for line in lines:
            upper = line.upper().strip()
            # Start capturing after INTERNSHIP heading
            if any(kw in upper for kw in ["INTERNSHIP","INTERN ","TRAINEE","WORK EXP"]):
                capture = True
                collected.append(line)
                continue
            # Stop capturing at next major heading
            if capture and upper and len(upper) < 40 and upper == upper.upper() and len(upper.split()) <= 4:
                if upper not in ("","•","-"):
                    break
            if capture:
                collected.append(line)
        exp_section = "\n".join(collected)

    year_ranges = _extract_year_ranges(exp_section) if exp_section else []
    total_years = _compute_total_years(year_ranges)

    # If still 0, look for "X months" or "X years" explicit statements in exp section
    if total_years == 0 and exp_section:
        months_match = re.findall(r"(\d+)\s*months?\s+(?:intern|experience|work)", exp_section.lower())
        if months_match:
            total_years = max(int(m) for m in months_match) / 12

    # Infer seniority
    seniority_level = 0  # default fresher
    if total_years >= 1:
        for level in sorted(SENIORITY_YEARS.keys()):
            lo, _ = SENIORITY_YEARS[level]
            if lo <= total_years:
                seniority_level = level

    # Check for intern/fresher keywords to keep seniority low
    if any(w in exp_section.lower() for w in ["intern","trainee","fresher"]):
        seniority_level = min(seniority_level, 1)

    titles = _extract_job_titles(exp_section)

    return {
        "total_years": total_years,
        "year_ranges": year_ranges,
        "seniority_inferred": SENIORITY_LABEL.get(seniority_level, "Junior"),
        "seniority_level": seniority_level,
        "job_titles": titles,
        "most_recent_title": titles[0] if titles else "Student/Fresher",
        "exp_section_found": bool(exp_section.strip()),
    }


def _extract_job_titles(text):
    titles = []
    pat = r"^([A-Z][A-Za-z\s,./]+(?:Engineer|Developer|Scientist|Analyst|Manager|Lead|Architect|Director|Intern|Consultant))\s*[—\-–|@•]"
    for line in text.split("\n"):
        m = re.match(pat, line.strip())
        if m:
            t = m.group(1).strip()
            if 3 < len(t) < 70:
                titles.append(t)
    return titles[:5]


def analyze_experience_gap(candidate, required):
    candidate_years   = candidate["total_years"]
    required_min      = required["min_years"]
    candidate_level   = candidate["seniority_level"]
    required_level    = required["seniority_level"]
    is_entry          = required.get("is_entry_level", False)

    # If it's an entry-level/intern JD, experience gap is not a real issue
    if is_entry:
        years_score   = 100.0
        years_verdict = "✅ Entry-level/intern role — no experience requirement"
        seniority_score   = 100.0
        seniority_verdict = "✅ Open to students and freshers"
    else:
        gap = required_min - candidate_years
        if gap <= 0:
            years_score   = 100.0
            years_verdict = f"✅ Meets requirement ({candidate_years:.0f} yrs ≥ {required_min} required)"
        elif gap <= 1:
            years_score   = 75.0
            years_verdict = f"🟡 Slightly under ({candidate_years:.0f} yrs, need {required_min} — {gap:.0f} yr gap)"
        else:
            years_score   = max(0, 40 - gap * 5)
            years_verdict = f"❌ Under by {gap:.0f} years ({candidate_years:.0f} vs {required_min} required)"

        s_gap = required_level - candidate_level
        if s_gap <= 0:
            seniority_score   = 100.0
            seniority_verdict = f"✅ Level match ({candidate['seniority_inferred']} meets {required['seniority_required']})"
        elif s_gap == 1:
            seniority_score   = 65.0
            seniority_verdict = f"🟡 One level below ({candidate['seniority_inferred']} vs {required['seniority_required']})"
        else:
            seniority_score   = 30.0
            seniority_verdict = f"❌ Level mismatch ({candidate['seniority_inferred']} vs {required['seniority_required']})"

    combined = round((years_score * 0.6) + (seniority_score * 0.4), 1)
    return {
        "years_score": years_score,
        "seniority_score": seniority_score,
        "combined_score": combined,
        "years_verdict": years_verdict,
        "seniority_verdict": seniority_verdict,
        "candidate_years": candidate_years,
        "required_min_years": required_min,
        "candidate_seniority": candidate["seniority_inferred"],
        "required_seniority": required["seniority_required"],
        "years_gap": max(0, required_min - candidate_years),
        "is_overqualified": (required_min > 0) and (candidate_years - required_min > 3),
        "is_entry_level": is_entry,
    }