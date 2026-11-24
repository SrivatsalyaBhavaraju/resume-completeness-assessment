
import os
import sys
import json
import random

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from rca import config
from rca.data import load_units

# --------------------------------------------------
# HUMAN VALIDATION OF THE DERIVED LABELS
# --------------------------------------------------
# Our training labels are derived automatically from the HTML structure.
# This script checks them against HUMAN judgement.
#
#   python src/13_human_validation.py export
#       -> data/annotation/annotation_sheet.csv  (blank human columns)
#
#   (team members fill the sheet following docs/ANNOTATION_GUIDELINES.md;
#    ideally two annotators fill human_a_* and human_b_* independently)
#
#   python src/13_human_validation.py score
#       -> results/human_validation.json
#          * agreement between annotators (Cohen's kappa)
#          * agreement between derived labels and humans
#          * fine-tuned model vs humans, including the "vague" judgement,
#            which the HTML cannot provide

ANNOTATION_DIR = os.path.join(config.PROJECT_ROOT, "data", "annotation")
SHEET = os.path.join(ANNOTATION_DIR, "annotation_sheet.csv")
PER_TYPE = {"experience": 60, "education": 40, "summary": 30, "skills": 30}

mode = sys.argv[1] if len(sys.argv) > 1 else "export"

if mode == "export":
    os.makedirs(ANNOTATION_DIR, exist_ok=True)
    rng = random.Random(config.RANDOM_SEED)
    test_units = [u for u in load_units("test") if u["num_chunks"] == 1]
    rows = []
    for section_type, n in PER_TYPE.items():
        candidates = [u for u in test_units if u["type"] == section_type]
        # Over-sample flagged sections so that positives are not too rare
        flagged = [u for u in candidates if any(v == 1 for v in u["labels"].values())]
        clean = [u for u in candidates if u not in flagged]
        chosen = rng.sample(flagged, min(len(flagged), n // 2)) + rng.sample(clean, n - min(len(flagged), n // 2))
        for unit in chosen:
            row = {"unit_id": f"{unit['resume_id']}_{unit['section_index']}_{unit['chunk_index']}",
                   "section_type": section_type, "text": unit["text"]}
            for name, spec in config.LABELS.items():
                if spec["section"] == section_type:
                    row[f"human_a_{name}"] = ""
                    row[f"human_b_{name}"] = ""
            row["human_a_vague"] = ""
            row["human_b_vague"] = ""
            rows.append(row)
    rng.shuffle(rows)
    pd.DataFrame(rows).to_csv(SHEET, index=False, encoding="utf-8-sig")
    print(f"Exported {len(rows)} sections to {SHEET}")
    print("Derived labels are deliberately NOT included, so annotators are not biased.")

elif mode == "score":
    import torch
    from transformers import AutoTokenizer
    from rca.model import ResumeCompletenessModel

    sheet = pd.read_csv(SHEET, encoding="utf-8-sig")
    units = {f"{u['resume_id']}_{u['section_index']}_{u['chunk_index']}": u for u in load_units("test")}
    with open(os.path.join(config.FINETUNED_DIR, "decision_policy.json"), "r", encoding="utf-8") as file:
        thresholds = json.load(file)["thresholds"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = ResumeCompletenessModel.load(config.FINETUNED_DIR, device)
    tokenizer = AutoTokenizer.from_pretrained(config.FINETUNED_DIR)

    @torch.no_grad()
    def probabilities(text):
        inputs = tokenizer(text, truncation=True, max_length=config.MAX_LENGTH, return_tensors="pt").to(device)
        return torch.sigmoid(model(inputs["input_ids"], inputs["attention_mask"])[1])[0].cpu().numpy()

    results = {}
    for name in config.LABEL_NAMES + ["vague"]:
        col_a, col_b = f"human_a_{name}", f"human_b_{name}"
        if col_a not in sheet:
            continue
        rows = sheet[sheet[col_a].notna()]
        if rows.empty:
            continue
        human_a = rows[col_a].astype(int).values
        entry = {"annotated": int(len(rows))}
        if col_b in rows and rows[col_b].notna().all():
            entry["kappa_annotator_a_vs_b"] = round(cohen_kappa_score(human_a, rows[col_b].astype(int)), 4)
        if name in config.LABELS:
            derived = np.array([units[u]["labels"].get(name, -1) for u in rows["unit_id"]])
            known = derived >= 0
            entry["kappa_derived_vs_human"] = round(cohen_kappa_score(human_a[known], derived[known]), 4)
            j = config.LABEL_NAMES.index(name)
            model_pred = np.array([probabilities(t)[j] >= thresholds[name] for t in rows["text"]]).astype(int)
            entry["model_f1_vs_human"] = round(f1_score(human_a, model_pred, zero_division=0), 4)
        else:
            # "vague" is not a trained label: test whether the model's
            # thin-description / summary-too-brief probability tracks it.
            proxy = []
            for text, section_type in zip(rows["text"], rows["section_type"]):
                p = probabilities(text)
                key = "exp_thin_description" if section_type == "experience" else "summary_too_brief"
                proxy.append(p[config.LABEL_NAMES.index(key)])
            if 0 < human_a.sum() < len(human_a):
                from sklearn.metrics import roc_auc_score
                entry["auroc_of_model_proxy_vs_human_vague"] = round(roc_auc_score(human_a, proxy), 4)
        results[name] = entry
        print(name, entry)

    with open(os.path.join(config.RESULTS_DIR, "human_validation.json"), "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)
    print("Saved: results/human_validation.json")
