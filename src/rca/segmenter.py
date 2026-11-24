"""
Rule-based segmentation of PLAIN resume text into sections.

A real uploaded resume has no HTML tags, so the system must find section
boundaries in plain text. We detect headings using a lexicon that is
LEARNED FROM THE TRAINING SPLIT: every heading string the candidates wrote
(<div class="sectiontitle">) is mapped to the section type of the HTML
section code it belongs to. This is data-driven rather than a hand-written
keyword list, and because it is built only from training resumes the
segmentation accuracy we report on test resumes is not inflated.

Works on both newline-formatted text (real resumes) and the dataset's
flattened Resume_str (headings separated by runs of spaces).
"""

import json
import re

# Headings that matter for our project but are (almost) absent from the
# dataset. Added manually and documented as such.
MANUAL_HEADINGS = {
    "projects": "projects",
    "project": "projects",
    "academic projects": "projects",
    "personal projects": "projects",
    "key projects": "projects",
    "project experience": "projects",
    "contact": "contact",
    "contact information": "contact",
    "contact details": "contact",
    "personal details": "contact",
    "objective": "summary",
    "career objective": "summary",
    "about me": "summary",
    "internships": "experience",
    "internship": "experience",
    "certifications": "certifications",
    "certificates": "certifications",
    "licenses and certifications": "certifications",
}

FRAGMENT_SPLIT = re.compile(r"\n|\s{2,}")


def normalise_heading(text):
    text = text.replace("&amp;", "&").lower()
    text = re.sub(r"[:\-–—_*#|]+$", "", text.strip())
    text = re.sub(r"[^a-z&/ ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_lexicon(title_type_pairs, min_count=3):
    """
    title_type_pairs: iterable of (heading text, section type) from the
    training split. A heading is kept if it occurs at least `min_count`
    times; it maps to its most frequent section type.
    """
    counts = {}
    for title, section_type in title_type_pairs:
        key = normalise_heading(title)
        if not key:
            continue
        counts.setdefault(key, {})
        counts[key][section_type] = counts[key].get(section_type, 0) + 1

    lexicon = {}
    for key, by_type in counts.items():
        total = sum(by_type.values())
        if total >= min_count:
            lexicon[key] = max(by_type, key=by_type.get)

    # Manual headings override: "projects" and "contact" do not exist as
    # section types in the dataset (its 3 "Projects" headings sit inside
    # generic custom sections), so the learned mapping would be wrong.
    lexicon.update(MANUAL_HEADINGS)
    return lexicon


def load_lexicon(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def segment(text, lexicon, max_heading_words=6):
    """
    Split plain text into sections.

    Returns a list of dicts:
        {"heading": str | None, "type": str | None, "text": str}
    The first element (heading None) is the preamble before the first
    heading - normally the name, headline and contact details.
    """
    pieces = []
    position = 0
    for match in FRAGMENT_SPLIT.finditer(text):
        pieces.append((position, match.start()))
        position = match.end()
    pieces.append((position, len(text)))

    headings = []
    for start, end in pieces:
        fragment = text[start:end].strip()
        if not fragment or len(fragment.split()) > max_heading_words:
            continue
        key = normalise_heading(fragment)
        if key in lexicon:
            headings.append((start, end, fragment, lexicon[key]))

    sections = []
    preamble_end = headings[0][0] if headings else len(text)
    sections.append({"heading": None, "type": None, "text": text[:preamble_end].strip(),
                     "heading_start": None, "start": 0, "end": preamble_end})

    for index, (start, end, fragment, section_type) in enumerate(headings):
        next_start = headings[index + 1][0] if index + 1 < len(headings) else len(text)
        body = text[end:next_start].strip()
        sections.append({"heading": fragment, "type": section_type, "text": body,
                         "heading_start": start, "start": end, "end": next_start})

    return sections
