
import os
import json
import random

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from rca import config
from rca.html_sections import parse_resume_html
from rca.labels import section_labels
from rca.inference import ResumeAssessor
from rca.text_utils import replace_placeholders

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------
# The whole system on raw PLAIN TEXT of the held-out test resumes:
#   segmentation -> chunking -> fine-tuned model -> aggregation -> hybrid
# Compared with the truth derived from each resume's HTML.
#
# Resume-level label for a section type T and label L:
#   1 if ANY section of type T in the resume has deficiency L.
# (A resume can have e.g. both a "Skills" and a "Highlights" block.)

EVALUATED_TYPES = ["summary", "education", "experience", "skills"]

print("=" * 65)
print("END-TO-END SYSTEM EVALUATION (PLAIN TEXT, TEST RESUMES)")
print("=" * 65)

df = pd.read_csv(config.DATASET_PATH)
with open(config.SPLITS_FILE, "r", encoding="utf-8") as file:
    test_ids = set(json.load(file)["test"])
test_df = df[df["ID"].isin(test_ids)]

assessor = ResumeAssessor()
print("Test resumes:", len(test_df), "| policy:", assessor.source)

# --------------------------------------------------
# 2. RUN THE SYSTEM
# --------------------------------------------------

gold = {name: [] for name in config.LABEL_NAMES}
pred = {name: [] for name in config.LABEL_NAMES}
presence = {t: {"gold": [], "pred": []} for t in EVALUATED_TYPES}

for count, row in enumerate(test_df.itertuples(), start=1):
    html_sections = parse_resume_html(row.Resume_html)
    text = replace_placeholders(str(row.Resume_str), random.Random(int(row.ID)))
    report = assessor.assess(text)

    for section_type in EVALUATED_TYPES:
        presence[section_type]["gold"].append(int(any(s["type"] == section_type for s in html_sections)))
        presence[section_type]["pred"].append(int(section_type in report["present"]))

    for name, spec in config.LABELS.items():
        section_type = spec["section"]
        gold_values = [
            section_labels(s["type"], s["entries"], s["text"]).get(name)
            for s in html_sections if s["type"] == section_type
        ]
        gold_values = [v for v in gold_values if v is not None]
        predicted = [s for s in report["sections"] if s["type"] == section_type and name in s["labels"]]
        # Only score when the section exists in BOTH (presence errors are
        # measured separately above).
        if gold_values and predicted:
            gold[name].append(int(any(gold_values)))
            pred[name].append(int(any(s["labels"][name]["flagged"] for s in predicted)))

    if count % 50 == 0:
        print(f"  assessed {count}/{len(test_df)} resumes")

# --------------------------------------------------
# 3. METRICS
# --------------------------------------------------

results = {"presence": {}, "labels": {}}

print(f"\n{'Section presence':18s}{'Precision':>10s}{'Recall':>10s}{'F1':>8s}")
for section_type, values in presence.items():
    g, p = values["gold"], values["pred"]
    results["presence"][section_type] = {
        "precision": round(precision_score(g, p, zero_division=0), 4),
        "recall": round(recall_score(g, p, zero_division=0), 4),
        "f1": round(f1_score(g, p, zero_division=0), 4),
    }
    r = results["presence"][section_type]
    print(f"{section_type:18s}{r['precision']:10.3f}{r['recall']:10.3f}{r['f1']:8.3f}")

print(f"\n{'Deficiency label':28s}{'Precision':>10s}{'Recall':>10s}{'F1':>8s}{'Resumes':>9s}{'Source':>8s}")
for name in config.LABEL_NAMES:
    g, p = gold[name], pred[name]
    results["labels"][name] = {
        "precision": round(precision_score(g, p, zero_division=0), 4),
        "recall": round(recall_score(g, p, zero_division=0), 4),
        "f1": round(f1_score(g, p, zero_division=0), 4),
        "resumes": len(g),
        "positives": int(sum(g)),
        "decided_by": assessor.source[name],
    }
    r = results["labels"][name]
    print(f"{name:28s}{r['precision']:10.3f}{r['recall']:10.3f}{r['f1']:8.3f}{r['resumes']:9d}{r['decided_by']:>8s}")

results["macro_f1"] = round(float(np.mean([v["f1"] for v in results["labels"].values()])), 4)
print("\nEnd-to-end macro-F1:", results["macro_f1"])

with open(os.path.join(config.RESULTS_DIR, "end_to_end_results.json"), "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nSaved: results/end_to_end_results.json")
print("\n" + "=" * 65)
print("END-TO-END EVALUATION COMPLETED")
print("=" * 65)
