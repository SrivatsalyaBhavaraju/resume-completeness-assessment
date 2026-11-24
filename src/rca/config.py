"""
Central configuration: every path, section mapping, label definition and
threshold used by the pipeline lives here so it can be cited and defended
in one place.
"""

import os

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "input", "Resume.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

SECTIONS_FILE = os.path.join(PROCESSED_DIR, "sections.jsonl")
UNITS_FILE = os.path.join(PROCESSED_DIR, "training_units.jsonl")
SPLITS_FILE = os.path.join(PROCESSED_DIR, "splits.json")

FINETUNED_DIR = os.path.join(MODELS_DIR, "distilbert_rca")

MODEL_NAME = "distilbert/distilbert-base-uncased"
MAX_LENGTH = 512
RANDOM_SEED = 42

# --------------------------------------------------
# 2. SECTION TAXONOMY
# --------------------------------------------------
# The HTML of every resume tags each section with a 4-letter code
# (id="SECTION_EDUC..."). These codes are assigned by the resume
# builder, independently of the heading text the candidate typed,
# so they are a reliable ground truth for "which section is this".
#
# Codes are mapped onto the section types used by our project.
# Codes not listed fall into "other".

SECTION_CODE_TO_TYPE = {
    "SUMM": "summary",
    "PRFL": "summary",
    "OBJC": "summary",
    "EDUC": "education",
    "EXPR": "experience",
    "WRKH": "experience",
    "EXFU": "experience",
    "EEXP": "experience",
    "MILI": "experience",
    "SKLL": "skills",
    "HILT": "skills",
    "TSKL": "skills",
    "CERT": "certifications",
}

# NAME holds only the (anonymised) name / headline and PRIN the
# anonymised personal information, so they carry no learnable content.
IGNORED_SECTION_CODES = {"NAME", "PRIN", "PICT", "RELO", "RTTL", "ERTL"}

SECTION_TYPES = [
    "summary",
    "education",
    "experience",
    "skills",
    "certifications",
    "other",
]

# Required / optional sections as defined in the project PPT.
# Contact information and projects are handled by rules (see rules.py):
# the dataset is anonymised (no e-mails / phone numbers) and contains
# almost no "Projects" sections, so neither can be learned from it.
REQUIRED_SECTIONS = ["contact", "summary", "education", "experience", "skills"]
OPTIONAL_SECTIONS = ["projects", "certifications"]

# --------------------------------------------------
# 3. DEFICIENCY LABELS (MULTI-LABEL TARGETS)
# --------------------------------------------------
# Each label is a binary "deficiency flag": 1 = the problem is present.
# A label only applies to one section type; for other section types it
# is masked out of the loss (see model.py).
#
# "aggregate" states how chunk-level predictions are combined into a
# section-level prediction when a long section is split into chunks.
# It follows the logical quantifier in the label definition:
#   any  : flag if ANY entry has the problem          -> max over chunks
#   all  : flag only if NO entry anywhere avoids it   -> min over chunks
#   mean : property of the section as a whole         -> mean over chunks

LABELS = {
    "edu_missing_degree": {
        "section": "education",
        "aggregate": "any",
        "definition": "At least one education entry does not state a degree / qualification.",
    },
    "edu_missing_institution": {
        "section": "education",
        "aggregate": "any",
        "definition": "At least one education entry does not name the institution.",
    },
    "edu_missing_date": {
        "section": "education",
        "aggregate": "any",
        "definition": "At least one education entry has no date (graduation / attendance year).",
    },
    "exp_missing_dates": {
        "section": "experience",
        "aggregate": "any",
        "definition": "At least one role has no employment dates.",
    },
    "exp_missing_title": {
        "section": "experience",
        "aggregate": "any",
        "definition": "At least one role does not state the job title.",
    },
    "exp_thin_description": {
        "section": "experience",
        "aggregate": "any",
        "definition": "At least one role is described in fewer than "
                      "EXP_MIN_DESCRIPTION_WORDS words (responsibilities / contributions missing).",
    },
    "exp_no_quantified_impact": {
        "section": "experience",
        "aggregate": "all",
        "definition": "No role description contains a quantified result "
                      "(number, percentage, amount) - dates excluded.",
    },
    "summary_too_brief": {
        "section": "summary",
        "aggregate": "mean",
        "definition": "Summary has fewer than SUMMARY_MIN_WORDS words.",
    },
    "skills_too_few": {
        "section": "skills",
        "aggregate": "mean",
        "definition": "Skills section lists fewer than SKILLS_MIN_ITEMS distinct items.",
    },
}

LABEL_NAMES = list(LABELS.keys())

# --------------------------------------------------
# 4. THRESHOLDS
# --------------------------------------------------
# Thresholds were chosen from the empirical distribution of the
# dataset (see results/label_statistics.json produced by script 06)
# together with common resume-writing guidance. They are explicit
# design decisions, documented in docs/METHODOLOGY.md.

EXP_MIN_DESCRIPTION_WORDS = 15
SUMMARY_MIN_WORDS = 20
SKILLS_MIN_ITEMS = 5

# Placeholders the resume builder leaves in unfilled fields.
PLACEHOLDER_VALUES = {"select one", "select  one", ""}
