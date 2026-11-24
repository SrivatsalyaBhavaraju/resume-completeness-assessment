
import os
import sys
import json

import pandas as pd

from rca import config
from rca.html_sections import parse_resume_html
from rca.segmenter import load_lexicon, segment

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------
# Evaluates the rule-based segmenter on the PLAIN TEXT (Resume_str) of the
# held-out test resumes. Ground truth = the section types present in the
# resume's HTML. This measures the "section presence" part of the system.

EVALUATED_TYPES = ["summary", "education", "experience", "skills", "certifications"]

print("=" * 65)
print("SECTION SEGMENTATION EVALUATION (PLAIN TEXT, TEST SPLIT)")
print("=" * 65)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(config.DATASET_PATH)
with open(config.SPLITS_FILE, "r", encoding="utf-8") as file:
    test_ids = set(json.load(file)["test"])
lexicon = load_lexicon(os.path.join(config.PROCESSED_DIR, "heading_lexicon.json"))

test_df = df[df["ID"].isin(test_ids)]
print("\nTest resumes:", len(test_df))

# --------------------------------------------------
# 3. PRESENCE DETECTION PER SECTION TYPE
# --------------------------------------------------

counts = {t: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for t in EVALUATED_TYPES}
errors = []

for row in test_df.itertuples():
    gold = {s["type"] for s in parse_resume_html(row.Resume_html)}
    predicted = {s["type"] for s in segment(str(row.Resume_str), lexicon) if s["type"]}

    for section_type in EVALUATED_TYPES:
        g, p = section_type in gold, section_type in predicted
        key = "tp" if g and p else "fp" if p else "fn" if g else "tn"
        counts[section_type][key] += 1
        if key in ("fp", "fn") and len(errors) < 40:
            errors.append({"resume_id": int(row.ID), "type": section_type, "error": key})

# --------------------------------------------------
# 4. METRICS
# --------------------------------------------------

results = {}
print(f"\n{'Section':16s}{'Precision':>10s}{'Recall':>10s}{'F1':>10s}{'Support':>10s}")
print("-" * 56)
for section_type, c in counts.items():
    precision = c["tp"] / max(1, c["tp"] + c["fp"])
    recall = c["tp"] / max(1, c["tp"] + c["fn"])
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    results[section_type] = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "support": c["tp"] + c["fn"],
        **c,
    }
    print(f"{section_type:16s}{precision:10.3f}{recall:10.3f}{f1:10.3f}{c['tp'] + c['fn']:10d}")

with open(os.path.join(config.RESULTS_DIR, "segmenter_evaluation.json"), "w", encoding="utf-8") as file:
    json.dump({"per_type": results, "sample_errors": errors}, file, indent=4)

print("\nSaved: results/segmenter_evaluation.json")
print("\n" + "=" * 65)
print("SEGMENTATION EVALUATION COMPLETED")
print("=" * 65)
