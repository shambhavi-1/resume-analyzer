"""
utils/skill_extractor.py
─────────────────────────
NLP-powered skill extraction using spaCy + curated skill taxonomy.
Extracts programming languages, frameworks, tools, databases, and
soft skills from resume and job description text.
"""

import re
from typing import Optional
from functools import lru_cache

from loguru import logger

# ── Comprehensive Skill Taxonomy ─────────────────────────────────────────────
SKILL_TAXONOMY = {
    "programming_languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "c", "go",
        "golang", "rust", "ruby", "php", "swift", "kotlin", "scala", "r",
        "matlab", "perl", "bash", "shell", "powershell", "haskell", "elixir",
        "clojure", "lua", "julia", "dart", "groovy", "fortran", "cobol",
        "assembly", "vba", "objective-c", "f#", "erlang",
    ],
    "web_frameworks": [
        "react", "react.js", "reactjs", "angular", "angularjs", "vue", "vue.js",
        "vuejs", "svelte", "next.js", "nextjs", "nuxt", "gatsby", "django",
        "flask", "fastapi", "express", "express.js", "node.js", "nodejs",
        "spring", "spring boot", "spring mvc", "rails", "ruby on rails",
        "laravel", "symfony", "asp.net", ".net", "fastify", "hapi", "koa",
        "aiohttp", "tornado", "bottle", "pyramid", "starlette",
        "streamlit", "gradio", "chainlit", "panel",
    ],
    "ml_ai": [
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
        "xgboost", "lightgbm", "catboost", "hugging face", "transformers",
        "langchain", "llamaindex", "llama index", "openai", "anthropic",
        "sentence transformers", "spacy", "nltk", "gensim", "fasttext",
        "bert", "gpt", "llm", "nlp", "computer vision", "cv",
        "deep learning", "machine learning", "neural networks",
        "reinforcement learning", "generative ai", "diffusion models",
        "stable diffusion", "yolo", "object detection", "image segmentation",
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
        "jupyter", "mlflow", "wandb", "weights & biases", "dvc",
        "feature engineering", "model deployment", "mlops",
        "recommendation systems", "a/b testing", "causal inference",
        "time series", "anomaly detection", "clustering", "classification",
        "regression", "random forest", "gradient boosting", "svm",
        "pca", "dimensionality reduction", "embeddings", "faiss",
        "sentence transformers", "sentence-transformers",
        "lora", "qlora", "peft", "lo-ra",
        "retrieval augmented generation", "retrieval-augmented generation",
        "document q&a", "text generation", "text classification",
        "named entity recognition", "ner", "pos tagging",
        "bleu", "rouge", "perplexity",
        "distilgpt", "distilbert", "roberta", "llama", "mistral",
        "data mining", "data processing", "optimization", "collaboration",
        "research", "experimentation", "prototyping", "scalability",
        "algorithms", "algorithmic",
        "vector database", "rag", "fine-tuning", "lora", "qlora",
    ],
    "data_engineering": [
        "sql", "mysql", "postgresql", "postgres", "sqlite", "oracle",
        "sql server", "mssql", "mariadb", "mongodb", "redis", "cassandra",
        "dynamodb", "elasticsearch", "opensearch", "neo4j", "influxdb",
        "couchdb", "firebase", "supabase", "bigquery", "snowflake",
        "redshift", "databricks", "spark", "pyspark", "hadoop", "hive",
        "kafka", "rabbitmq", "celery", "airflow", "dbt", "dagster",
        "prefect", "luigi", "flink", "nifi", "beam",
    ],
    "devops_cloud": [
        "docker", "kubernetes", "k8s", "helm", "terraform", "ansible",
        "aws", "amazon web services", "gcp", "google cloud", "azure",
        "microsoft azure", "ci/cd", "github actions", "gitlab ci",
        "jenkins", "circleci", "travis ci", "argocd", "gitops",
        "nginx", "apache", "linux", "ubuntu", "centos",
        "prometheus", "grafana", "elk stack", "datadog", "newrelic",
        "cloudformation", "cdk", "pulumi", "vagrant", "virtualbox",
        "istio", "envoy", "consul", "vault", "packer",
    ],
    "api_architecture": [
        "rest", "restful", "graphql", "grpc", "websocket", "microservices",
        "api gateway", "oauth", "jwt", "openapi", "swagger",
        "event-driven", "pub/sub", "message queue", "cqrs",
        "domain-driven design", "ddd", "clean architecture",
        "solid principles", "design patterns",
    ],
    "tools_version_control": [
        "git", "github", "gitlab", "bitbucket", "svn", "mercurial",
        "jira", "confluence", "notion", "trello", "asana", "linear",
        "figma", "adobe xd", "photoshop", "illustrator",
        "postman", "insomnia", "swagger ui",
        "vscode", "intellij", "pycharm", "vim", "emacs",
    ],
    "soft_skills": [
        "agile", "scrum", "kanban", "sprint", "product management",
        "technical leadership", "team lead", "mentoring", "code review",
        "system design", "architecture", "scalability", "performance",
        "problem solving", "communication", "collaboration",
    ],
}

# Flat lookup for fast O(1) membership test
_ALL_SKILLS_FLAT: set[str] = set()
_SKILL_TO_CATEGORY: dict[str, str] = {}

for _cat, _skills in SKILL_TAXONOMY.items():
    for _s in _skills:
        _ALL_SKILLS_FLAT.add(_s.lower())
        _SKILL_TO_CATEGORY[_s.lower()] = _cat

# Canonical display form (title-case for most, uppercase for acronyms)
_ACRONYMS = {
    "sql", "aws", "gcp", "ci/cd", "api", "rest", "grpc", "nlp",
    "cv", "llm", "rag", "ml", "ai", "jwt", "html", "css", "php",
    "vba", "svm", "pca", "ddd", "cqrs", "k8s", "elk", "cdk",
}


def _canonical(skill: str) -> str:
    """Return a nicely-formatted display version of a skill name."""
    lower = skill.lower()
    if lower in _ACRONYMS:
        return lower.upper()
    # Known mixed-case spellings
    known_cases = {
        "javascript": "JavaScript",
        "typescript": "TypeScript",
        "tensorflow": "TensorFlow",
        "pytorch": "PyTorch",
        "scikit-learn": "scikit-learn",
        "sklearn": "scikit-learn",
        "spacy": "spaCy",
        "react.js": "React",
        "reactjs": "React",
        "vue.js": "Vue.js",
        "vuejs": "Vue.js",
        "next.js": "Next.js",
        "nextjs": "Next.js",
        "node.js": "Node.js",
        "nodejs": "Node.js",
        "spring boot": "Spring Boot",
        "github": "GitHub",
        "gitlab": "GitLab",
        "postgresql": "PostgreSQL",
        "mysql": "MySQL",
        "mongodb": "MongoDB",
        "redis": "Redis",
        "dynamodb": "DynamoDB",
        "elasticsearch": "Elasticsearch",
        "fastapi": "FastAPI",
        "openai": "OpenAI",
        "hugging face": "Hugging Face",
        "langchain": "LangChain",
        "peft": "PEFT",
        "lora": "LoRA",
        "qlora": "QLoRA",
        "streamlit": "Streamlit",
        "gradio": "Gradio",
        "sentence transformers": "Sentence Transformers",
        "sentence-transformers": "Sentence Transformers",
        "distilgpt": "DistilGPT",
        "distilbert": "DistilBERT",
        "data mining": "Data Mining",
        "optimization": "Optimization",
        "research": "Research",
        "collaboration": "Collaboration",
        "prototyping": "Prototyping",
        "algorithmic": "Algorithmic Foundations",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "pyspark": "PySpark",
        "github actions": "GitHub Actions",
        "gitlab ci": "GitLab CI",
        "graphql": "GraphQL",
        "websocket": "WebSocket",
        "postgresql": "PostgreSQL",
    }
    if lower in known_cases:
        return known_cases[lower]
    return skill.title()


# ── spaCy loader ─────────────────────────────────────────────────────────────
_nlp = None


def _get_nlp():
    """Lazy-load spaCy model. Falls back gracefully if not installed."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model 'en_core_web_sm' loaded.")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm\n"
                "Falling back to rule-based extraction only."
            )
            _nlp = None
    except ImportError:
        logger.warning("spaCy not installed. Using rule-based extraction only.")
        _nlp = None
    return _nlp


# ── Core extraction logic ────────────────────────────────────────────────────

def _extract_skills_regex(text: str) -> set[str]:
    """Fast regex-based skill extraction from the curated taxonomy."""
    found = set()
    text_lower = text.lower()

    # Sort by length descending so "spring boot" matches before "spring"
    sorted_skills = sorted(_ALL_SKILLS_FLAT, key=len, reverse=True)

    for skill in sorted_skills:
        # Word-boundary match to avoid "java" matching inside "javascript"
        # Exception: handle skills with special chars like "c++", "c#"
        if skill in ("c++", "c#", "c"):
            if skill == "c":
                pattern = r"\bc\b"
            else:
                pattern = re.escape(skill)
        else:
            pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text_lower):
            found.add(skill)

    # Post-process: remove "c" if "c++" or "c#" found
    if "c++" in found or "c#" in found:
        found.discard("c")

    # Remove "sklearn" and keep "scikit-learn"
    if "scikit-learn" in found:
        found.discard("sklearn")

    # Normalize react variants
    for alias in ("react.js", "reactjs"):
        if alias in found and "react" in found:
            found.discard(alias)

    return found


def _extract_skills_spacy(text: str, nlp) -> set[str]:
    """
    Use spaCy NER + noun chunks to catch skills the regex might miss
    (e.g., company-specific tool names, newer frameworks).
    """
    found = set()
    doc = nlp(text[:100_000])  # cap to avoid OOM on huge texts

    # Extract noun chunks and check against taxonomy
    for chunk in doc.noun_chunks:
        chunk_lower = chunk.text.lower().strip()
        if chunk_lower in _ALL_SKILLS_FLAT:
            found.add(chunk_lower)

    # Extract individual tokens labelled as technical terms (heuristic)
    tech_pos = {"PROPN", "NOUN"}
    for token in doc:
        tok_lower = token.text.lower().strip()
        if tok_lower in _ALL_SKILLS_FLAT and token.pos_ in tech_pos:
            found.add(tok_lower)

    return found


def extract_skills(text: str) -> dict:
    """
    Full skill extraction pipeline.

    Returns:
        {
          "all_skills": ["Python", "FastAPI", ...],       # sorted, canonical
          "by_category": {
              "programming_languages": ["Python", ...],
              "ml_ai": ["TensorFlow", ...],
              ...
          },
          "raw_matches": ["python", "fastapi", ...]       # lowercase originals
        }
    """
    if not text or not text.strip():
        return {"all_skills": [], "by_category": {}, "raw_matches": []}

    # Step 1: Regex extraction (primary)
    raw_skills = _extract_skills_regex(text)

    # Step 2: spaCy augmentation (supplementary)
    nlp = _get_nlp()
    if nlp:
        spacy_skills = _extract_skills_spacy(text, nlp)
        raw_skills = raw_skills.union(spacy_skills)

    # Step 3: Categorize and canonicalize
    by_category: dict[str, list[str]] = {}
    for skill in raw_skills:
        cat = _SKILL_TO_CATEGORY.get(skill, "other")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(_canonical(skill))

    # Sort within each category
    for cat in by_category:
        by_category[cat] = sorted(set(by_category[cat]))

    all_skills = sorted(
        set(item for sublist in by_category.values() for item in sublist)
    )

    logger.debug(f"Extracted {len(all_skills)} skills from {len(text)} chars of text")

    return {
        "all_skills": all_skills,
        "by_category": by_category,
        "raw_matches": sorted(raw_skills),
    }


def extract_skills_from_sections(sections: dict) -> dict:
    """
    Extract skills using multiple text sections for better coverage.
    Prioritizes the 'skills' section but also scans experience and projects.
    """
    # Weight: skills section gets full text, others get partial scan
    combined_parts = []

    if sections.get("skills"):
        combined_parts.append(sections["skills"])
    if sections.get("experience"):
        combined_parts.append(sections["experience"])
    if sections.get("projects"):
        combined_parts.append(sections["projects"])
    if sections.get("summary"):
        combined_parts.append(sections["summary"])

    combined_text = "\n".join(combined_parts)

    # If no structured sections, fall back to full text
    if not combined_text.strip() and sections.get("full_text"):
        combined_text = sections["full_text"]

    return extract_skills(combined_text)


def compare_skills(resume_skills: list[str], jd_skills: list[str]) -> dict:
    """
    Compare resume skills against job description skills.

    Returns:
        {
          "matched": [...],
          "missing": [...],
          "extra": [...],       # skills in resume but NOT in JD
          "match_pct": 72.5,    # percentage of JD skills found in resume
        }
    """
    resume_lower = {s.lower() for s in resume_skills}
    jd_lower = {s.lower() for s in jd_skills}

    # Build display-form lookups
    resume_display = {s.lower(): s for s in resume_skills}
    jd_display = {s.lower(): s for s in jd_skills}

    matched_lower = resume_lower & jd_lower
    missing_lower = jd_lower - resume_lower
    extra_lower = resume_lower - jd_lower

    matched = sorted([jd_display.get(s, resume_display.get(s, s.title())) for s in matched_lower])
    missing = sorted([jd_display.get(s, s.title()) for s in missing_lower])
    extra = sorted([resume_display.get(s, s.title()) for s in extra_lower])

    match_pct = (len(matched) / len(jd_lower) * 100) if jd_lower else 0.0

    return {
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "match_pct": round(match_pct, 1),
        "total_jd_skills": len(jd_lower),
        "total_resume_skills": len(resume_lower),
        "total_matched": len(matched),
        "total_missing": len(missing),
    }