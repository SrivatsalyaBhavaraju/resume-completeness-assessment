"""
Rule-based validation.

Two roles:

1. Deterministic checks that do not need a neural model
   (contact details are fixed patterns: e-mail, phone, profile URL).

2. A PLAIN-TEXT RULE BASELINE for every deficiency label. This is what a
   keyword / regex system would do with only the plain text of a section.
   The fine-tuned DistilBERT must beat this baseline on a label for us to
   justify using the model for that label (see the hybrid policy in
   12_assess_resume.py / docs/METHODOLOGY.md).
"""

import re

from rca.config import EXP_MIN_DESCRIPTION_WORDS, SUMMARY_MIN_WORDS, SKILLS_MIN_ITEMS
from rca.labels import has_quantity, word_count, skill_items

# --------------------------------------------------
# 1. CONTACT INFORMATION
# --------------------------------------------------

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Any run of digits, spaces, dots, dashes and brackets, optionally with a
# leading "+". It counts as a phone number if it holds 10-13 digits, which
# covers Indian (+91 98450 12345), US ((555) 123-4567) and other formats.
PHONE_CANDIDATE = re.compile(r"\+?\(?\d[\d\s().-]{7,}\d")
PROFILE_URL = re.compile(r"(linkedin\.com/\S+|github\.com/\S+|https?://\S+)", re.I)


def has_phone(text):
    for match in PHONE_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if 10 <= len(digits) <= 13 and not DATE_RANGE.fullmatch(match.group(0).strip()):
            return True
    return False


def check_contact(text):
    """Return which contact items are present in the resume text."""
    return {
        "email": bool(EMAIL.search(text)),
        "phone": has_phone(text),
        "profile_link": bool(PROFILE_URL.search(text)),
    }


# --------------------------------------------------
# 2. PLAIN-TEXT RULE BASELINE
# --------------------------------------------------

MONTHS = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE = re.compile(
    rf"\b({MONTHS}\s*\d{{4}}|\d{{1,2}}/\d{{2,4}}|(19|20)\d{{2}}|current|present)\b", re.I
)
DATE_RANGE = re.compile(
    rf"({MONTHS}\s*)?(\d{{1,2}}/)?(19|20)\d{{2}}\s*(to|-|–|—)\s*"
    rf"(({MONTHS}\s*)?(\d{{1,2}}/)?(19|20)\d{{2}}|current|present)",
    re.I,
)

DEGREE_KEYWORDS = re.compile(
    r"\b(bachelor|master|associate|doctor|ph\.?\s?d|mba|m\.?b\.?a|b\.?\s?s\.?c?|m\.?\s?s\.?c?|"
    r"b\.?\s?a|m\.?\s?a|b\.?\s?tech|m\.?\s?tech|b\.?\s?e|m\.?\s?e|bba|bcom|mcom|diploma|"
    r"degree|ged|high school|certificate|certification|a\.?\s?a\.?s|j\.?d|m\.?d|"
    r"r\.?n|b\.?s\.?n|coursework)\b",
    re.I,
)
INSTITUTION_KEYWORDS = re.compile(
    r"\b(university|college|institute|school|academy|polytechnic|"
    r"universidad|iit|nit|community college)\b",
    re.I,
)
TITLE_KEYWORDS = re.compile(
    r"\b(manager|engineer|assistant|associate|specialist|analyst|director|"
    r"coordinator|consultant|representative|officer|administrator|supervisor|"
    r"technician|developer|designer|teacher|intern|chef|cook|nurse|accountant|"
    r"clerk|lead|executive|president|agent|advisor|adviser|instructor|trainer|"
    r"owner|founder|head|architect|programmer|attorney|lawyer|paralegal|"
    r"server|cashier|driver|mechanic|pilot|editor|writer|producer|artist|"
    r"auditor|banker|teller|planner|buyer|recruiter|generalist|volunteer|"
    r"member|operator|worker|staff|tutor|professor|faculty)s?\b",
    re.I,
)


BULLET = re.compile(r"^\s*[-*•·▪◦]\s*")


def project_blocks(text):
    """
    Split a Projects section into projects: a new project starts at a
    non-bullet line (the project title) that follows bullet / description
    lines, or after a blank line.
    """
    blocks, current, seen_description = [], [], False
    for line in text.split("\n"):
        if not line.strip():
            if current:
                blocks.append(current)
            current, seen_description = [], False
            continue
        is_bullet = bool(BULLET.match(line))
        if not is_bullet and seen_description and current:
            blocks.append(current)
            current, seen_description = [], False
        current.append(line)
        seen_description = seen_description or is_bullet or len(line.split()) > 8
    if current:
        blocks.append(current)
    return blocks


def project_rule(label, text):
    """
    Projects are absent from the training data, so the model is not
    trusted there; the label definitions are applied directly.
    """
    blocks = project_blocks(text) or [[text]]
    if label == "exp_thin_description":
        # Title line excluded, description words counted per project
        return int(any(word_count(" ".join(block[1:] if len(block) > 1 else block))
                       < EXP_MIN_DESCRIPTION_WORDS for block in blocks))
    if label == "exp_no_quantified_impact":
        return int(not has_quantity(DATE.sub(" ", text)))
    raise ValueError(f"No project rule for: {label}")


def rule_predict(label, text):
    """Plain-text rule prediction (1 = deficiency) for one label."""

    if label == "edu_missing_degree":
        return int(not DEGREE_KEYWORDS.search(text))

    if label == "edu_missing_institution":
        return int(not INSTITUTION_KEYWORDS.search(text))

    if label == "edu_missing_date":
        return int(not DATE.search(text))

    if label == "exp_missing_dates":
        return int(not DATE_RANGE.search(text) and not DATE.search(text))

    if label == "exp_missing_title":
        return int(not TITLE_KEYWORDS.search(text))

    if label == "exp_thin_description":
        roles = max(1, len(DATE_RANGE.findall(text)))
        # ~8 words per role are title / company / location / dates
        description_words = word_count(text) - 8 * roles
        return int(description_words / roles < EXP_MIN_DESCRIPTION_WORDS)

    if label == "exp_no_quantified_impact":
        without_dates = DATE_RANGE.sub(" ", text)
        without_dates = DATE.sub(" ", without_dates)
        return int(not has_quantity(without_dates))

    if label == "summary_too_brief":
        return int(word_count(text) < SUMMARY_MIN_WORDS)

    if label == "skills_too_few":
        return int(len(skill_items(text)) < SKILLS_MIN_ITEMS)

    raise ValueError(f"Unknown label: {label}")
