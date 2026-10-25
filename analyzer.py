"""Resume / job-description fit analysis.

Deliberately free of Streamlit imports so the scoring can be exercised directly
by eval_analyzer.py.

Why this is not a single whole-document cosine similarity: all-MiniLM-L6-v2
truncates at 256 word pieces, and a real two-page resume runs 800-1500. A
document-level embedding therefore never sees most of the resume, and squeezing
two long documents into one vector each collapses the score into a narrow band
(~0.45-0.70) where a strong and a mediocre candidate are indistinguishable.

Instead: the JD is broken into individual requirements, the resume into
overlapping chunks, and each requirement is matched against its best-matching
chunk. That keeps every input under the token limit and yields a per-requirement
score with the resume line that justifies it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from skills_db import SKILL_DB

# --- Tunables. Calibrated against eval_analyzer.py; see TUNING.md notes there.
COVERED_T = 0.50          # cosine at or above this counts as evidenced
PARTIAL_T = 0.34          # between PARTIAL_T and COVERED_T = related but thin
SIM_FLOOR = 0.10          # cosine mapped to 0% credit
SIM_CEIL = 0.55           # cosine mapped to 100% credit; real matches top out near
                          # 0.58 sentence-to-sentence, so a higher ceiling would
                          # make even a perfect candidate look mediocre.
W_SKILLS = 0.55           # weight of explicit skill coverage in the final score
W_REQS = 0.45             # weight of semantic requirement coverage
PREFERRED_WEIGHT = 0.40   # a "nice to have" counts this much against a "must have"
MAX_REQUIREMENTS = 24
CHUNK_WORDS = 45
CHUNK_OVERLAP = 12

_PUNCT = re.compile(r"[^a-z0-9+#./\- ]+")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase and strip punctuation, preserving the characters that carry
    meaning in technology names: ``c++``, ``c#``, ``.net``, ``ci/cd``, ``node.js``."""
    t = text.lower().replace("–", "-").replace("—", "-").replace("’", "'")
    t = _PUNCT.sub(" ", t)
    return _WS.sub(" ", t).strip()


def _alias_pattern(aliases: list[str]) -> re.Pattern:
    # Longest first so "machine learning" wins over a hypothetical "machine".
    parts = sorted((re.escape(a) for a in aliases), key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9+#])(?:" + "|".join(parts) + r")(?![a-z0-9+#])")


_SKILLS = [(name, cat, _alias_pattern(aliases), url) for name, cat, aliases, url in SKILL_DB]


def find_skills(normalized: str) -> dict[str, tuple[str, str | None]]:
    """Return {canonical skill: (category, learn_url)} present in normalized text."""
    return {
        name: (cat, url)
        for name, cat, pattern, url in _SKILLS
        if pattern.search(normalized)
    }


# --- Job description parsing -------------------------------------------------

_BULLET = re.compile(r"^\s*(?:[-*•▪◦‣·o]|\d+[.)])\s+")
_SENTENCE = re.compile(r"(?<=[.;!?])\s+")

_REQ_CUE = re.compile(
    r"\b(experience[d]?|proficien\w*|knowledge|familiar\w*|skill\w*|abilit\w*|able to|"
    r"degree|years?|bachelor\w*|master\w*|phd|understand\w*|expertise|background|"
    r"competen\w*|you will|you'll|responsible|work with|build|develop|design|deploy|"
    r"maintain|collaborat\w*|strong|solid|hands.on|demonstrated|track record|"
    r"required|preferred|must)\b"
)
_REQUIRED_CUE = re.compile(
    r"\b(must have|must|required|require[sd]?|essential|mandatory|minimum|at least|"
    r"strong|solid|proven|demonstrated)\b"
)
_PREFERRED_CUE = re.compile(
    r"\b(preferred|nice to have|nice-to-have|plus|bonus|desirable|ideally|advantage|"
    r"familiarity|would be great|good to have)\b"
)
_SECTION_PREFERRED = re.compile(
    r"^\s*(preferred|nice to have|bonus|desirable|good to have|pluses)\b[^.]{0,40}:?\s*$"
)
_SECTION_REQUIRED = re.compile(
    r"^\s*(requirements?|qualifications?|must have|what you.{0,4}ll need|"
    r"responsibilit\w+|who you are|about you)\b[^.]{0,40}:?\s*$"
)


@dataclass
class Requirement:
    text: str
    priority: str          # "required" | "preferred"
    score: float = 0.0     # best cosine against any resume chunk
    evidence: str = ""     # the resume chunk that best matched
    status: str = "missing"  # "covered" | "partial" | "missing"
    skills_present: list[str] = field(default_factory=list)
    skills_missing: list[str] = field(default_factory=list)

    @property
    def semantic_credit(self) -> float:
        return float(np.clip((self.score - SIM_FLOOR) / (SIM_CEIL - SIM_FLOOR), 0.0, 1.0))

    @property
    def lexical_credit(self) -> float | None:
        """Share of the named technologies this requirement asks for that the
        resume actually contains, or None when it names none."""
        total = len(self.skills_present) + len(self.skills_missing)
        return (len(self.skills_present) / total) if total else None

    @property
    def credit(self) -> float:
        """Hybrid of semantic and literal evidence.

        A skills list embeds poorly against prose: "Hands-on experience
        deploying services with Docker on AWS" scores only ~0.27 against a
        resume whose skills line plainly reads "Docker, Kubernetes, AWS".
        Literal evidence of the named technology is the stronger signal, so
        whichever source is more convincing wins.
        """
        lexical = self.lexical_credit
        if lexical is None:
            return self.semantic_credit
        return max(self.semantic_credit, lexical)


def split_requirements(jd_text: str) -> tuple[list[Requirement], str, str]:
    """Pull individual requirements out of a job description.

    Also returns (required_blob, preferred_blob): the normalized text of each
    priority band, used to decide whether a skill is a must-have or a bonus.
    """
    requirements: list[Requirement] = []
    required_lines: list[str] = []
    preferred_lines: list[str] = []
    seen: set[str] = set()
    section = "required"

    for raw_line in jd_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        low = line.lower()
        if _SECTION_PREFERRED.match(low):
            section = "preferred"
            continue
        if _SECTION_REQUIRED.match(low):
            section = "required"
            continue

        is_bullet = bool(_BULLET.match(line))
        body = _BULLET.sub("", line).strip()

        # A paragraph can hold several requirements; a bullet is already one.
        pieces = [body] if (is_bullet or len(body.split()) <= 35) else _SENTENCE.split(body)

        for piece in pieces:
            piece = piece.strip(" \t;:-•")
            words = piece.split()
            if not (4 <= len(words) <= 60):
                continue

            low_piece = piece.lower()
            if not (is_bullet or _REQ_CUE.search(low_piece)):
                continue

            if _PREFERRED_CUE.search(low_piece):
                priority = "preferred"
            elif _REQUIRED_CUE.search(low_piece):
                priority = "required"
            else:
                priority = section

            (preferred_lines if priority == "preferred" else required_lines).append(piece)

            key = normalize(piece)
            if key in seen:
                continue
            seen.add(key)
            requirements.append(Requirement(text=piece, priority=priority))

    # Fall back to sentence splitting when the JD has no line structure at all.
    if not requirements:
        for piece in _SENTENCE.split(jd_text.replace("\n", " ")):
            piece = piece.strip()
            if 4 <= len(piece.split()) <= 60:
                requirements.append(Requirement(text=piece, priority="required"))
                required_lines.append(piece)

    return (
        requirements[:MAX_REQUIREMENTS],
        normalize(" ".join(required_lines)),
        normalize(" ".join(preferred_lines)),
    )


# --- Resume parsing ----------------------------------------------------------

def chunk_resume(resume_text: str, max_words: int = CHUNK_WORDS) -> list[str]:
    """Split a resume into overlapping chunks that stay well under the model's
    256 word-piece limit, so nothing past the first page is silently dropped."""
    units: list[str] = []
    for raw_line in resume_text.splitlines():
        line = _BULLET.sub("", raw_line.strip()).strip()
        if len(line.split()) < 3:
            continue
        if len(line.split()) > max_words:
            units.extend(s.strip() for s in _SENTENCE.split(line) if s.strip())
        else:
            units.append(line)

    chunks: list[str] = []
    buffer: list[str] = []
    for unit in units:
        buffer.extend(unit.split())
        if len(buffer) >= max_words:
            chunks.append(" ".join(buffer[:max_words]))
            buffer = buffer[max_words - CHUNK_OVERLAP:]
    if len(buffer) >= 3:
        chunks.append(" ".join(buffer))

    return chunks or [resume_text[:500]]


# --- Experience ---------------------------------------------------------------

_JD_YEARS = re.compile(r"(\d{1,2})\s*\+?\s*(?:-|to|–)?\s*(?:\d{1,2})?\s*\+?\s*years?")
_DATE_RANGE = re.compile(
    r"((?:19|20)\d{2})\s*(?:-|–|—|to)\s*((?:19|20)\d{2}|present|current|now)",
    re.I,
)


def required_years(jd_text: str) -> int | None:
    values = [int(m) for m in _JD_YEARS.findall(jd_text.lower())]
    values = [v for v in values if 0 < v <= 20]
    return max(values) if values else None


def estimated_years(resume_text: str) -> float | None:
    """Rough span of employment implied by date ranges. An estimate only:
    overlapping roles are merged, undated work is invisible."""
    this_year = date.today().year
    spans: list[tuple[int, int]] = []
    for start, end in _DATE_RANGE.findall(resume_text):
        s = int(start)
        e = this_year if end.lower() in {"present", "current", "now"} else int(end)
        if 1960 < s <= e <= this_year:
            spans.append((s, e))
    if not spans:
        return None

    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return float(sum(e - s for s, e in merged))


# --- ATS checks ---------------------------------------------------------------

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
_SECTION_WORDS = {
    "experience": r"\b(experience|employment|work history)\b",
    "education": r"\b(education|academic|university|college|b\.?e\.?|b\.?tech|degree)\b",
    "skills": r"\b(skills?|technologies|technical)\b",
}


def ats_report(resume_text: str) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    words = len(resume_text.split())
    low = resume_text.lower()

    if words < 50:
        checks.append((False, f"Only {words} words extracted - the PDF is likely image-based or heavily formatted, which ATS parsers cannot read."))
    elif words > 1200:
        checks.append((False, f"{words} words is long for a resume; most screeners expect under about 800."))
    else:
        checks.append((True, f"{words} words extracted cleanly."))

    checks.append((bool(_EMAIL.search(resume_text)), "Email address found." if _EMAIL.search(resume_text) else "No email address detected - parsers often key on it."))
    checks.append((bool(_PHONE.search(resume_text)), "Phone number found." if _PHONE.search(resume_text) else "No phone number detected."))

    missing = [name for name, pattern in _SECTION_WORDS.items() if not re.search(pattern, low)]
    if missing:
        checks.append((False, "Missing standard section heading(s): " + ", ".join(sorted(missing)) + "."))
    else:
        checks.append((True, "Standard sections present (experience, education, skills)."))

    return checks


# --- Result types -------------------------------------------------------------

@dataclass
class SkillGap:
    name: str
    category: str
    priority: str
    in_resume: bool
    learn_url: str | None = None
    related: list[str] = field(default_factory=list)

    @property
    def weight(self) -> float:
        return 1.0 if self.priority == "required" else PREFERRED_WEIGHT


@dataclass
class Analysis:
    overall: int
    verdict: str
    skill_percent: int
    requirement_percent: int
    skills: list[SkillGap]
    requirements: list[Requirement]
    category_coverage: dict[str, tuple[int, int]]
    ats: list[tuple[bool, str]]
    years_required: int | None
    years_estimated: float | None
    resume_chunks: int

    @property
    def missing_required(self) -> list[SkillGap]:
        return [s for s in self.skills if not s.in_resume and s.priority == "required"]

    @property
    def missing_preferred(self) -> list[SkillGap]:
        return [s for s in self.skills if not s.in_resume and s.priority == "preferred"]

    @property
    def matched(self) -> list[SkillGap]:
        return [s for s in self.skills if s.in_resume]


def _verdict(score: int) -> str:
    if score >= 75:
        return "Strong match"
    if score >= 58:
        return "Good match"
    if score >= 40:
        return "Partial match"
    return "Weak match"


def analyze(resume_text: str, jd_text: str, model) -> Analysis:
    """Score a resume against a job description.

    ``model`` is any object exposing SentenceTransformer's
    ``encode(list[str], normalize_embeddings=True) -> np.ndarray``.
    """
    norm_resume = normalize(resume_text)
    requirements, required_blob, preferred_blob = split_requirements(jd_text)
    chunks = chunk_resume(resume_text)

    # --- explicit skill coverage
    jd_skills = find_skills(normalize(jd_text))
    resume_skills = find_skills(norm_resume)

    # --- requirement coverage: semantic similarity plus literal skill evidence
    if requirements:
        req_vecs = model.encode([r.text for r in requirements], normalize_embeddings=True)
        chunk_vecs = model.encode(chunks, normalize_embeddings=True)
        sims = np.asarray(req_vecs) @ np.asarray(chunk_vecs).T

        for i, requirement in enumerate(requirements):
            best = int(np.argmax(sims[i]))
            requirement.score = float(sims[i][best])
            requirement.evidence = chunks[best]

            named = find_skills(normalize(requirement.text))
            requirement.skills_present = sorted(n for n in named if n in resume_skills)
            requirement.skills_missing = sorted(n for n in named if n not in resume_skills)

            lexical = requirement.lexical_credit
            if (lexical is not None and lexical >= 0.75) or requirement.score >= COVERED_T:
                requirement.status = "covered"
            elif (lexical is not None and lexical > 0.0) or requirement.score >= PARTIAL_T:
                requirement.status = "partial"
            else:
                requirement.status = "missing"

        weights = np.array([1.0 if r.priority == "required" else PREFERRED_WEIGHT for r in requirements])
        credits = np.array([r.credit for r in requirements])
        requirement_score = float((weights * credits).sum() / weights.sum())
    else:
        requirement_score = 0.0

    skills: list[SkillGap] = []
    for name, (category, url) in sorted(jd_skills.items()):
        pattern = next(p for n, _c, p, _u in _SKILLS if n == name)
        # A skill only mentioned in the "nice to have" band is weighted down.
        priority = "preferred" if (
            not pattern.search(required_blob) and pattern.search(preferred_blob)
        ) else "required"
        in_resume = name in resume_skills
        related = (
            []
            if in_resume
            else [n for n in sorted(resume_skills) if resume_skills[n][0] == category][:3]
        )
        skills.append(SkillGap(name, category, priority, in_resume, url, related))

    if skills:
        total_w = sum(s.weight for s in skills)
        skill_score = sum(s.weight for s in skills if s.in_resume) / total_w
        overall = W_SKILLS * skill_score + W_REQS * requirement_score
    else:
        skill_score = 0.0
        overall = requirement_score

    # --- per-category coverage, for the radar chart
    category_coverage: dict[str, tuple[int, int]] = {}
    for skill in skills:
        hit, total = category_coverage.get(skill.category, (0, 0))
        category_coverage[skill.category] = (hit + int(skill.in_resume), total + 1)

    score = int(round(np.clip(overall, 0.0, 1.0) * 100))
    return Analysis(
        overall=score,
        verdict=_verdict(score),
        skill_percent=int(round(skill_score * 100)),
        requirement_percent=int(round(requirement_score * 100)),
        skills=skills,
        requirements=requirements,
        category_coverage=category_coverage,
        ats=ats_report(resume_text),
        years_required=required_years(jd_text),
        years_estimated=estimated_years(resume_text),
        resume_chunks=len(chunks),
    )
