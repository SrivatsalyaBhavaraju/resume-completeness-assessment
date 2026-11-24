"""
Parse the Resume_html column into sections, entries and structured fields.

Structure of the builder HTML (verified on the dataset):

    <div class="section" id="SECTION_EXPR992636660">
        <div class="sectiontitle"> Experience </div>
        <div class="paragraph"> ...one entry (one job / one degree)...
            <span class="jobtitle">  ...  </span>
            <span class="jobdates">  ...  </span>
            <span class="companyname"> ... </span>
            <span class="jobline"> <ul><li> bullet </li> ... </ul> </span>
        </div>
        ...
    </div>

Education entries use: degree, programline, companyname_educ, jobdates.
"""

import re
from bs4 import BeautifulSoup

from rca.config import SECTION_CODE_TO_TYPE, IGNORED_SECTION_CODES

SECTION_ID_PATTERN = re.compile(r"^SECTION_([A-Z]+)\d*")
NOT_SKILLS_TITLE = re.compile(r"achievement|accomplishment|award|honou?r|recognition", re.I)

# Field classes we extract from each entry
FIELD_CLASSES = [
    "jobtitle",
    "jobdates",
    "companyname",
    "companyname_educ",
    "degree",
    "programline",
    "jobline",
]


def clean(text):
    """Collapse all whitespace runs into single spaces."""
    return re.sub(r"\s+", " ", text or "").strip()


def block_text(element):
    """
    Text of an element with line breaks kept at block boundaries
    (bullets, paragraphs, line breaks) so the reconstructed section
    reads like real resume text rather than one long line.
    """
    for br in element.find_all("br"):
        br.replace_with("\n")
    for tag in element.find_all(["li", "p", "div"]):
        tag.insert_before("\n")
        tag.insert_after("\n")
    for tag in element.find_all("span", class_="paddedline"):
        tag.insert_after("\n")

    lines = [clean(line) for line in element.get_text().split("\n")]
    return "\n".join(line for line in lines if line)


def entry_fields(paragraph):
    """Return {field_class: [text, ...]} for one entry (div.paragraph)."""
    fields = {name: [] for name in FIELD_CLASSES}

    for span in paragraph.find_all("span"):
        classes = span.get("class") or []
        for name in FIELD_CLASSES:
            if name in classes:
                # companyname_educ spans also carry the plain "companyname" class
                if name == "companyname" and "companyname_educ" in classes:
                    continue
                fields[name].append(clean(span.get_text(" ")))

    # Free-text descriptions sometimes sit in a generic "field" span
    # (e.g. bullet lists without a jobline wrapper, or next to an
    # empty jobline span).
    if not any(fields["jobline"]):
        fields["jobline"] = []
        for span in paragraph.find_all("span", class_="field"):
            text = clean(span.get_text(" "))
            if text:
                fields["jobline"].append(text)

    return fields


def parse_resume_html(html):
    """
    Parse one resume. Returns a list of sections:

        {
          "code": "EXPR",
          "type": "experience",
          "title": "Professional Experience",
          "entries": [ {"text": ..., "fields": {...}}, ... ],
          "text": full section text WITHOUT the heading
        }
    """
    soup = BeautifulSoup(html or "", "html.parser")
    sections = []

    for div in soup.find_all("div", id=SECTION_ID_PATTERN):
        code = SECTION_ID_PATTERN.match(div["id"]).group(1)
        if code in IGNORED_SECTION_CODES:
            continue

        title_div = div.find("div", class_="sectiontitle")
        title = clean(title_div.get_text(" ")) if title_div else ""
        heading = div.find("div", class_="heading")
        if heading is not None:
            heading.decompose()

        entries = []
        for paragraph in div.find_all("div", class_="paragraph"):
            fields = entry_fields(paragraph)
            text = block_text(paragraph)
            if text:
                entries.append({"text": text, "fields": fields})

        text = "\n".join(entry["text"] for entry in entries)
        if not text:
            continue

        section_type = SECTION_CODE_TO_TYPE.get(code, "other")
        # The HILT ("highlights") block is a skills list for most resumes,
        # but some candidates retitle it "Achievements" etc.
        if code == "HILT" and NOT_SKILLS_TITLE.search(title):
            section_type = "other"

        sections.append({
            "code": code,
            "type": section_type,
            "title": title,
            "entries": entries,
            "text": text,
        })

    return sections
