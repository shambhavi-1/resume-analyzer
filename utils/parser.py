"""
utils/parser.py — resume parsing with full Indian resume + internship support
"""
import re, io
from pathlib import Path
from typing import Union

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

SECTION_PATTERNS = {
    "skills": [
        r"(?i)^(technical\s+)?skills?(\s+&\s+\w+)?[\s:]*$",
        r"(?i)^core\s+competencies[\s:]*$",
        r"(?i)^technologies[\s:]*$",
        r"(?i)^proficiencies[\s:]*$",
        r"(?i)^technical\s+expertise[\s:]*$",
    ],
    "experience": [
        r"(?i)^(work\s+)?experience[\s:]*$",
        r"(?i)^professional\s+experience[\s:]*$",
        r"(?i)^employment(\s+history)?[\s:]*$",
        r"(?i)^work\s+history[\s:]*$",
        r"(?i)^career[\s:]*$",
        # KEY FIX: internship headings map to experience
        r"(?i)^internship(\s+experience)?[\s:]*$",
        r"(?i)^intern(ship)?s?[\s:]*$",
        r"(?i)^(industry\s+)?training[\s:]*$",
        r"(?i)^work\s+experience[\s:]*$",
    ],
    "education": [
        r"(?i)^education(al)?(\s+(background|history|qualifications))?[\s:]*$",
        r"(?i)^academic\s+(background|qualifications|history)[\s:]*$",
        r"(?i)^degrees?[\s:]*$",
        r"(?i)^academics?[\s:]*$",
    ],
    "projects": [
        r"(?i)^(personal\s+|side\s+|open.?source\s+|key\s+)?projects?[\s:]*$",
        r"(?i)^portfolio[\s:]*$",
        r"(?i)^(key\s+)?project\s+work[\s:]*$",
        r"(?i)^academic\s+projects?[\s:]*$",
    ],
    "summary": [
        r"(?i)^(professional\s+)?(summary|profile|objective|about(\s+me)?|overview)[\s:]*$",
        r"(?i)^career\s+(objective|summary)[\s:]*$",
        r"(?i)^(personal\s+)?statement[\s:]*$",
    ],
    "certifications": [
        r"(?i)^certifications?(\s+(and|&)\s+\w+)?[\s:]*$",
        r"(?i)^licenses?(\s+(and|&)\s+certifications?)?[\s:]*$",
        r"(?i)^courses?\s+and\s+certifications?[\s:]*$",
        r"(?i)^professional\s+development[\s:]*$",
    ],
    "leadership": [
        r"(?i)^leadership(\s+(experience|activities|roles?))?[\s:]*$",
        r"(?i)^(extra.?curricular|co.?curricular)\s*(activities)?[\s:]*$",
        r"(?i)^activities[\s:]*$",
        r"(?i)^volunteer(ing)?[\s:]*$",
        r"(?i)^achievements?[\s:]*$",
        r"(?i)^awards?(\s+(and|&)\s+honors?)?[\s:]*$",
    ],
}


def _extract_text_pdfplumber(file_bytes):
    text_parts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text(x_tolerance=2, y_tolerance=2)
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


def _extract_text_pypdf2(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_text_from_pdf(source):
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            file_bytes = f.read()
    else:
        file_bytes = source
    text = ""
    if HAS_PDFPLUMBER:
        try:
            text = _extract_text_pdfplumber(file_bytes)
        except Exception:
            pass
    if not text and HAS_PYPDF2:
        text = _extract_text_pypdf2(file_bytes)
    if not text:
        raise RuntimeError("Could not extract text from PDF.")
    return text


def clean_text(raw):
    fixes = {"\ufb01":"fi","\ufb02":"fl","\u2019":"'","\u2018":"'",
              "\u201c":'"',"\u201d":'"',"\u2013":"-","\u2014":"-","\u00a0":" "}
    for bad, good in fixes.items():
        raw = raw.replace(bad, good)
    raw = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = re.sub(r"[ \t]{2,}", " ", raw)
    return raw.strip()


def _is_section_heading(line):
    stripped = line.strip()
    if len(stripped) > 60 or len(stripped) < 2:
        return None
    for section, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if re.match(pat, stripped):
                return section
    return None


def split_into_sections(text):
    sections = {k: [] for k in SECTION_PATTERNS}
    sections["other"] = []
    lines = text.split("\n")
    current = "other"
    for line in lines:
        heading = _is_section_heading(line)
        if heading:
            current = heading
        else:
            sections[current].append(line)
    result = {"full_text": text}
    for sec, lines_list in sections.items():
        joined = "\n".join(lines_list).strip()
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        result[sec] = joined
    return result


def extract_contact_info(text):
    contact = {}
    email = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
    if email:
        contact["email"] = email.group()
    phone = re.search(r"(\+?\d[\d\s\-().]{8,14}\d)", text)
    if phone:
        contact["phone"] = phone.group().strip()
    linkedin = re.search(r"linkedin\.com/in/[\w-]+", text, re.IGNORECASE)
    if linkedin:
        contact["linkedin"] = "https://" + linkedin.group()
    github = re.search(r"github\.com/[\w-]+", text, re.IGNORECASE)
    if github:
        contact["github"] = "https://" + github.group()
    for line in text.split("\n")[:8]:
        line = line.strip()
        if (line and 2 <= len(line.split()) <= 5
                and not re.search(r"[@/|]", line)
                and not re.search(r"\d", line)
                and line == line.title()):
            contact["name"] = line
            break
    return contact


def parse_resume_text(text):
    clean = clean_text(text)
    sections = split_into_sections(clean)
    contact = extract_contact_info(clean)
    sections["contact"] = contact
    sections["word_count"] = len(clean.split())
    return sections


def parse_resume(source):
    raw = extract_text_from_pdf(source)
    return parse_resume_text(raw)