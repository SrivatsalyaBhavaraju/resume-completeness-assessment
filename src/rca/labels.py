"""
Derive section-level deficiency labels from the STRUCTURED HTML fields.

Why this is not circular
------------------------
The labels are computed from the resume builder's field markup
(<span class="degree">, <span class="jobdates">, <span class="jobline"> ...).
The model never sees this markup: at training and inference time it only
receives the PLAIN TEXT of the section (headings removed). So the model
has to learn, from raw text, what the builder recorded structurally.
This is distant supervision: an external, objective signal that is
available for our dataset but not for a real uploaded resume.

Unknown vs. negative
--------------------
If an entry has no structured fields at all (free-text block), we cannot
know whether e.g. a degree is missing, so the label is None ("unknown")
and is masked out of the loss - it is NOT treated as 0.
"""

import re

from rca.config import (
    LABELS,
    PLACEHOLDER_VALUES,
    EXP_MIN_DESCRIPTION_WORDS,
    SUMMARY_MIN_WORDS,
    SKILLS_MIN_ITEMS,
)

YEAR = re.compile(r"^(19|20)\d{2}$")
NUMBER_WORDS = re.compile(r"\b(hundred|thousand|million|billion|dozen)s?\b", re.I)
TOKEN_WITH_DIGIT = re.compile(r"[$€£]?\d[\d,.]*\s*(%|percent|k\b|m\b|\+)?", re.I)
SKILL_SPLIT = re.compile(r"[\n,;|•·]+|\s{2,}")


def is_filled(values):
    """True if the field list has at least one non-placeholder value."""
    return any(v.strip().lower() not in PLACEHOLDER_VALUES for v in values)


def word_count(text):
    return len(re.findall(r"[A-Za-z][A-Za-z'+#.-]*", text or ""))


def has_quantity(text):
    """
    True if the text contains a quantity that is not just a year.
    Examples counted: "15%", "$2M", "120 clients", "3 teams", "thousands".
    Examples not counted: "2014", "Jan 2015".
    """
    if NUMBER_WORDS.search(text or ""):
        return True
    for match in TOKEN_WITH_DIGIT.finditer(text or ""):
        token = match.group(0).strip().rstrip(".,")
        digits = re.sub(r"[^\d]", "", token)
        if not digits:
            continue
        if YEAR.match(digits) and token == digits:
            continue
        return True
    return False


def skill_items(text):
    items = {
        item.strip(" -*:.").lower()
        for item in SKILL_SPLIT.split(text or "")
    }
    return {item for item in items if item and word_count(item) <= 8}


# --------------------------------------------------
# PER-ENTRY CHECKS
# --------------------------------------------------

def education_entry(fields):
    """Return per-entry flags (1 / 0 / None) for one education entry."""
    structured = any(fields[k] for k in ("degree", "companyname_educ", "jobdates", "programline"))
    if not structured:
        return {"edu_missing_degree": None, "edu_missing_institution": None, "edu_missing_date": None}

    return {
        "edu_missing_degree": int(not is_filled(fields["degree"])),
        "edu_missing_institution": int(not is_filled(fields["companyname_educ"])),
        "edu_missing_date": int(not is_filled(fields["jobdates"])),
    }


def experience_entry(fields):
    """Return per-entry flags for one experience entry (one role)."""
    structured = any(fields[k] for k in ("jobtitle", "jobdates", "companyname"))
    description = " ".join(fields["jobline"])

    flags = {
        "exp_thin_description": int(word_count(description) < EXP_MIN_DESCRIPTION_WORDS),
        "has_quantity": int(has_quantity(description)),
        "exp_missing_dates": None,
        "exp_missing_title": None,
    }
    if structured:
        flags["exp_missing_dates"] = int(not is_filled(fields["jobdates"]))
        flags["exp_missing_title"] = int(not is_filled(fields["jobtitle"]))
    return flags


# --------------------------------------------------
# COMBINE ENTRIES -> SECTION LABELS
# --------------------------------------------------

def combine_any(values):
    known = [v for v in values if v is not None]
    return int(any(known)) if known else None


def section_labels(section_type, entries, text):
    """
    Labels for a section (or for a chunk made of whole entries).
    Returns {label_name: 1 | 0 | None} for labels of this section type only.
    """
    labels = {}

    if section_type == "education":
        per_entry = [education_entry(e["fields"]) for e in entries]
        for name in ("edu_missing_degree", "edu_missing_institution", "edu_missing_date"):
            labels[name] = combine_any([p[name] for p in per_entry])

    elif section_type == "experience":
        per_entry = [experience_entry(e["fields"]) for e in entries]
        for name in ("exp_missing_dates", "exp_missing_title", "exp_thin_description"):
            labels[name] = combine_any([p[name] for p in per_entry])
        labels["exp_no_quantified_impact"] = int(not any(p["has_quantity"] for p in per_entry))

    elif section_type == "summary":
        labels["summary_too_brief"] = int(word_count(text) < SUMMARY_MIN_WORDS)

    elif section_type == "skills":
        labels["skills_too_few"] = int(len(skill_items(text)) < SKILLS_MIN_ITEMS)

    assert all(name in LABELS for name in labels)
    return labels
