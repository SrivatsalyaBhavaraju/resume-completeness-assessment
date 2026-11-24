"""
End-to-end assessment of a PLAIN-TEXT resume.

    resume text
        -> rule-based segmentation (heading lexicon)        [segmenter.py]
        -> each section split into <=510-token chunks (line-aligned)
        -> fine-tuned DistilBERT: section type + deficiency probabilities
        -> chunk predictions aggregated to section level (max / min / mean)
        -> hybrid decision per label: model or rule (chosen on validation)
        -> rule-based checks: contact details, required / optional presence
        -> section-wise feedback                             [feedback.py]
"""

import json
import os
import re

import numpy as np
import torch
from transformers import AutoTokenizer

from rca import config, feedback
from rca.model import ResumeCompletenessModel
from rca.rules import DATE_RANGE, MONTHS, check_contact, project_rule, rule_predict
from rca.segmenter import load_lexicon, segment

# Month-year dates ("Jul 2010", "07/2010") used as role anchors. Bare
# years are excluded: they also appear inside descriptions.
MONTH_DATE = re.compile(rf"\b({MONTHS}\s*(19|20)\d{{2}}|\d{{1,2}}/(19|20)\d{{2}})\b", re.I)
SEPARATOR = re.compile(r"\n|\s{2,}")
BLANK_LINE = re.compile(r"\n[ \t]*\n\s*")

AGGREGATE = {"any": np.max, "all": np.min, "mean": np.mean}

DISPLAY_TITLE = {
    "summary": "Summary",
    "education": "Education",
    "experience": "Experience",
    "skills": "Skills",
    "certifications": "Certifications",
    "projects": "Projects",
}

# Projects sections are absent from the dataset, so no projects-specific
# label could be trained. A project description is assessed with the
# experience-description labels (same nature: what you did, with what,
# with what result). This transfer is NOT validated on data.
PROJECT_LABELS = ["exp_thin_description", "exp_no_quantified_impact"]


class ResumeAssessor:

    def __init__(self, model_dir=config.FINETUNED_DIR, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.meta = ResumeCompletenessModel.load(model_dir, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.lexicon = load_lexicon(os.path.join(config.PROCESSED_DIR, "heading_lexicon.json"))
        with open(os.path.join(model_dir, "decision_policy.json"), "r", encoding="utf-8") as file:
            policy = json.load(file)
        self.thresholds = policy["thresholds"]
        self.source = policy["source"]  # label -> "model" | "rule"
        # Packing whole roles beat one-role-per-chunk on validation
        # (mean experience F1 0.617 vs 0.531), so packing is the default.
        self.one_role_per_chunk = policy.get("one_role_per_chunk", False)

    # --------------------------------------------------
    # CHUNKING
    # --------------------------------------------------

    def pieces(self, text, budget):
        """
        Lines of the section; a line longer than the budget (flattened
        text has no line breaks) is split on runs of spaces, and as a last
        resort into fixed word windows.
        """
        result = []
        for line in text.split("\n"):
            if not line.strip():
                continue
            if len(self.tokenizer.tokenize(line)) <= budget:
                result.append(line)
                continue
            for fragment in re.split(r"\s{2,}", line):
                if len(self.tokenizer.tokenize(fragment)) <= budget:
                    result.append(fragment)
                else:
                    words = fragment.split()
                    result += [" ".join(words[k:k + 250]) for k in range(0, len(words), 250)]
        return result

    @staticmethod
    def role_starts(text):
        """
        Character offsets where a new role (job / degree) begins.

        Every dated role contains a date anchor ("01/2014 to Current",
        "Jul 2010"). The role starts either at the anchor (date-first
        layout) or at the short title fragment just before it
        (title-first layout: "Sous Chef   Jul 2010").
        """
        anchors = [m.start() for m in DATE_RANGE.finditer(text)]
        covered = [(m.start(), m.end()) for m in DATE_RANGE.finditer(text)]
        for m in MONTH_DATE.finditer(text):
            if not any(a <= m.start() < b for a, b in covered):
                anchors.append(m.start())

        starts = set()
        for anchor in sorted(anchors):
            before = text[:anchor]
            separators = list(SEPARATOR.finditer(before))
            if len(separators) >= 1:
                fragment_start = separators[-1].end()
                fragment = before[fragment_start:].strip()
                if not fragment:
                    # Title sits one fragment further back (or at the very start)
                    fragment_start = separators[-2].end() if len(separators) >= 2 else 0
                    fragment = before[fragment_start:separators[-1].start()].strip()
                if fragment and len(fragment.split()) <= 8 and not fragment.endswith("."):
                    starts.add(fragment_start)
                    continue
            starts.add(anchor)

        # A blank line almost always separates two roles in a real
        # resume, and it is the only boundary an UNDATED role has.
        # (Dataset text has no blank lines, so evaluation is unaffected.)
        for match in BLANK_LINE.finditer(text):
            starts.add(match.end())
        starts = sorted(s for s in starts if 0 < s < len(text))

        # Drop a boundary that would leave a fragment holding only a title
        # or date line, i.e. a blank line placed between a title and its body.
        merged = []
        for start in starts:
            if merged and len(text[merged[-1]:start].split()) <= 4:
                continue
            merged.append(start)
        return merged

    def chunks(self, text, budget=config.MAX_LENGTH - 2, role_aligned=False):
        """
        Pack text into chunks of at most `budget` tokens.

        role_aligned=True (experience / projects): split the text into
        roles first and pack WHOLE roles, mirroring how the training
        chunks were built (whole entries). Otherwise whole lines are packed.
        """
        if role_aligned:
            bounds = [0] + self.role_starts(text) + [len(text)]
            roles = [text[a:b].strip() for a, b in zip(bounds, bounds[1:]) if text[a:b].strip()]
            if self.one_role_per_chunk:
                result = []
                for role in roles:
                    result += self.chunks(role, budget) if len(self.tokenizer.tokenize(role)) > budget else [role]
                return result or [text]
            lines = []
            for role in roles:
                lines += self.pieces(role, budget) if len(self.tokenizer.tokenize(role)) > budget else [role]
        else:
            lines = self.pieces(text, budget)
        result, current, used = [], [], 0
        for line in lines:
            n = len(self.tokenizer.tokenize(line))
            if current and used + n > budget:
                result.append("\n".join(current))
                current, used = [], 0
            current.append(line)
            used += n
        if current:
            result.append("\n".join(current))
        return result or [text]

    @torch.no_grad()
    def predict_chunks(self, texts):
        inputs = self.tokenizer(texts, truncation=True, max_length=config.MAX_LENGTH,
                                padding=True, return_tensors="pt").to(self.device)
        type_logits, label_logits = self.model(inputs["input_ids"], inputs["attention_mask"])
        return (torch.softmax(type_logits, -1).cpu().numpy(),
                torch.sigmoid(label_logits).cpu().numpy())

    # --------------------------------------------------
    # SECTION-LEVEL PREDICTION
    # --------------------------------------------------

    def assess_section(self, text, heading_type):
        role_aligned = heading_type in ("experience", "projects")
        chunk_texts = self.chunks(text, role_aligned=role_aligned)
        type_probs, label_probs = self.predict_chunks(chunk_texts)
        mean_type = type_probs.mean(0)
        model_type = config.SECTION_TYPES[int(mean_type.argmax())]

        # The heading decides the type when it is known; the model's type
        # prediction is used when there is no recognised heading.
        section_type = heading_type or model_type
        if section_type == "projects":
            applicable = PROJECT_LABELS
        else:
            applicable = [n for n, spec in config.LABELS.items() if spec["section"] == section_type]

        decisions = {}
        for name in applicable:
            j = config.LABEL_NAMES.index(name)
            probability = float(AGGREGATE[config.LABELS[name]["aggregate"]](label_probs[:, j]))
            model_flag = probability >= self.thresholds[name]
            if section_type == "projects":
                # No project sections in the training data: the label
                # definition is applied by rule; the model's probability is
                # reported as advisory only.
                decisions[name] = {
                    "flagged": bool(project_rule(name, text)),
                    "model_probability": round(probability, 3),
                    "decided_by": "rule (model advisory)",
                }
                continue
            source = self.source[name]
            rule_flag = bool(rule_predict(name, text)) if source != "model" else None
            flagged = {
                "model": model_flag,
                "rule": rule_flag,
                "and": model_flag and rule_flag,
                "or": model_flag or rule_flag,
            }[source]
            decisions[name] = {
                "flagged": bool(flagged),
                "model_probability": round(probability, 3),
                "decided_by": {"model": "model", "rule": "rule",
                               "and": "model AND rule", "or": "model OR rule"}[source],
            }

        return {
            "type": section_type,
            "model_type": model_type,
            "model_type_confidence": round(float(mean_type.max()), 3),
            "num_chunks": len(chunk_texts),
            "labels": decisions,
        }

    # --------------------------------------------------
    # FULL RESUME
    # --------------------------------------------------

    def assess(self, resume_text, preamble_min_words=25, no_heading_confidence=0.80):
        segments = segment(resume_text, self.lexicon)
        sections = []

        for seg in segments:
            if not seg["text"].strip():
                continue
            if seg["heading"] is None:
                # Preamble: name, headline, contact details. If it contains
                # substantial content, let the model say what it is.
                if len(seg["text"].split()) < preamble_min_words:
                    continue
                result = self.assess_section(seg["text"], None)
                if result["model_type"] == "other" or result["model_type_confidence"] < no_heading_confidence:
                    continue
                result["heading"] = None
                result["no_heading"] = True
            elif seg["type"] == "contact":
                continue
            else:
                result = self.assess_section(seg["text"], seg["type"])
                result["heading"] = seg["heading"]
                result["no_heading"] = False
            result["preview"] = seg["text"][:160].replace("\n", " ")
            result["span"] = {"heading_start": seg["heading_start"], "start": seg["start"], "end": seg["end"]}
            sections.append(result)

        present = {s["type"] for s in sections}
        contact = check_contact(resume_text)

        return {
            "contact": contact,
            "sections": sections,
            "present": sorted(present),
            "missing_required": [t for t in config.REQUIRED_SECTIONS
                                 if t != "contact" and t not in present],
            "missing_optional": [t for t in config.OPTIONAL_SECTIONS if t not in present],
            "feedback": self.feedback(sections, present, contact),
        }

    # --------------------------------------------------
    # FEEDBACK
    # --------------------------------------------------

    def feedback(self, sections, present, contact):
        items = []

        if not contact["email"] and not contact["phone"]:
            items.append({"section": "contact", "severity": "missing", "message": feedback.MISSING_REQUIRED["contact"]})
        else:
            for key in ("email", "phone"):
                if not contact[key]:
                    items.append({"section": "contact", "severity": "weak", "message": feedback.CONTACT_ITEMS[key]})
        if not contact["profile_link"]:
            items.append({"section": "contact", "severity": "suggestion", "message": feedback.CONTACT_ITEMS["profile_link"]})

        for section_type in config.REQUIRED_SECTIONS:
            if section_type != "contact" and section_type not in present:
                items.append({"section": section_type, "severity": "missing",
                              "message": feedback.MISSING_REQUIRED[section_type]})

        for section in sections:
            if section.get("no_heading"):
                items.append({"section": section["type"], "severity": "weak",
                              "message": feedback.NO_HEADING.format(
                                  type=section["type"], title=DISPLAY_TITLE.get(section["type"], section["type"]))})
            for name, decision in section["labels"].items():
                if decision["flagged"]:
                    items.append({"section": section["type"], "severity": "weak", "label": name,
                                  "confidence": decision["model_probability"],
                                  "decided_by": decision["decided_by"],
                                  "message": feedback.PROJECT_MESSAGES.get(name, feedback.MESSAGES[name])
                                  if section["type"] == "projects" else feedback.MESSAGES[name]})

        for section_type in config.OPTIONAL_SECTIONS:
            if section_type not in present:
                items.append({"section": section_type, "severity": "suggestion",
                              "message": feedback.MISSING_OPTIONAL[section_type]})
        return items
