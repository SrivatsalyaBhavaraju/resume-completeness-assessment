
import os
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from transformers import AutoTokenizer

from rca import config
from rca.data import load_units, to_arrays
from rca.evaluation import tune_thresholds, label_metrics, type_metrics
from rca.model import ResumeCompletenessModel
from rca.rules import rule_predict

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AGGREGATE = {"any": np.max, "all": np.min, "mean": np.mean}

print("=" * 65)
print("EVALUATING THE FINE-TUNED MODEL")
print("=" * 65)

model, meta = ResumeCompletenessModel.load(config.FINETUNED_DIR, DEVICE)
tokenizer = AutoTokenizer.from_pretrained(config.FINETUNED_DIR)
print("Loaded model from best epoch:", meta.get("best_epoch"))

# --------------------------------------------------
# 2. PREDICT ON VALIDATION AND TEST UNITS
# --------------------------------------------------


@torch.no_grad()
def predict(texts, batch_size=32):
    order = np.argsort([len(t) for t in texts])
    type_probs = np.zeros((len(texts), len(config.SECTION_TYPES)), dtype=np.float32)
    label_probs = np.zeros((len(texts), len(config.LABEL_NAMES)), dtype=np.float32)
    for start in range(0, len(texts), batch_size):
        index = order[start:start + batch_size]
        inputs = tokenizer([texts[i] for i in index], truncation=True, max_length=config.MAX_LENGTH,
                           padding=True, return_tensors="pt").to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            type_logits, label_logits = model(inputs["input_ids"], inputs["attention_mask"])
        type_probs[index] = torch.softmax(type_logits.float(), -1).cpu().numpy()
        label_probs[index] = torch.sigmoid(label_logits.float()).cpu().numpy()
    return type_probs, label_probs


units = {name: load_units(name) for name in ("val", "test")}
arrays = {name: to_arrays(u) for name, u in units.items()}
predictions = {name: predict(arrays[name][0]) for name in units}

val_texts, val_types, val_Y, val_M = arrays["val"]
test_texts, test_types, test_Y, test_M = arrays["test"]
val_type_probs, val_P = predictions["val"]
test_type_probs, test_P = predictions["test"]

# --------------------------------------------------
# 3. THRESHOLDS (VALIDATION) -> TEST METRICS
# --------------------------------------------------

thresholds = tune_thresholds(val_P, val_Y, val_M)
finetuned = {
    "labels": label_metrics((test_P >= thresholds).astype(int), test_Y, test_M, test_P),
    "section_type": type_metrics(test_type_probs.argmax(1), test_types),
    "thresholds": dict(zip(config.LABEL_NAMES, thresholds.round(2).tolist())),
}
print("\nFine-tuned test macro-F1:", finetuned["labels"]["macro_f1"],
      "| macro-AUROC:", finetuned["labels"]["macro_auroc"],
      "| section-type accuracy:", finetuned["section_type"]["accuracy"])

type_confusion = confusion_matrix(test_types, test_type_probs.argmax(1)).tolist()

# --------------------------------------------------
# 4. HYBRID DECISION POLICY (CHOSEN ON VALIDATION)
# --------------------------------------------------
# For each label we compare, ON VALIDATION, the F1 of the fine-tuned model
# and of the plain-text rule, and use whichever is better. This is how the
# PPT's "DistilBERT + rule-based validation" is made concrete and
# evidence-based instead of arbitrary.


def rule_matrix(texts, M):
    R = np.zeros_like(M)
    for i, text in enumerate(texts):
        for j, name in enumerate(config.LABEL_NAMES):
            if M[i, j]:
                R[i, j] = rule_predict(name, text)
    return R.astype(int)


val_R, test_R = rule_matrix(val_texts, val_M), rule_matrix(test_texts, test_M)
policy = {"thresholds": {}, "source": {}, "validation_f1": {}}
hybrid_test = np.zeros_like(test_Y, dtype=int)

# Four candidate decision sources per label:
#   model : fine-tuned DistilBERT alone
#   rule  : plain-text rule alone
#   and   : flag only if BOTH agree   (rule acts as a veto -> precision)
#   or    : flag if EITHER fires      (rule as a safety net -> recall)
# Ties are broken in this order, so the model is preferred when equal.


def combine(source, model_flags, rule_flags):
    if source == "model":
        return model_flags
    if source == "rule":
        return rule_flags
    if source == "and":
        return model_flags & rule_flags
    return model_flags | rule_flags


for j, name in enumerate(config.LABEL_NAMES):
    rows = val_M[:, j] == 1
    val_model = (val_P[rows, j] >= thresholds[j]).astype(int)
    scores = {
        source: f1_score(val_Y[rows, j], combine(source, val_model, val_R[rows, j]), zero_division=0)
        for source in ("model", "rule", "and", "or")
    }
    source = max(scores, key=scores.get)
    policy["thresholds"][name] = float(round(thresholds[j], 2))
    policy["source"][name] = source
    policy["validation_f1"][name] = {k: round(v, 4) for k, v in scores.items()}
    hybrid_test[:, j] = combine(source, (test_P[:, j] >= thresholds[j]).astype(int), test_R[:, j])

with open(os.path.join(config.FINETUNED_DIR, "decision_policy.json"), "w", encoding="utf-8") as file:
    json.dump(policy, file, indent=4)

hybrid = {"labels": label_metrics(hybrid_test, test_Y, test_M), "policy": policy["source"]}
print("Hybrid test macro-F1:", hybrid["labels"]["macro_f1"], "| policy:", policy["source"])

# --------------------------------------------------
# 5. SECTION-LEVEL EVALUATION (CHUNKS AGGREGATED)
# --------------------------------------------------
# Long sections were split into chunks. Here chunk probabilities are
# combined into one section prediction with the label's quantifier
# (any -> max, all -> min, mean -> mean) and compared with the
# section-level truth (any -> OR of chunk labels, all -> AND).

groups = {}
for i, unit in enumerate(units["test"]):
    groups.setdefault((unit["resume_id"], unit["section_index"]), []).append(i)

section_true, section_pred, section_mask = [], [], []
for rows in groups.values():
    y_row, p_row, m_row = np.zeros(len(config.LABEL_NAMES)), np.zeros(len(config.LABEL_NAMES)), np.zeros(len(config.LABEL_NAMES))
    for j, name in enumerate(config.LABEL_NAMES):
        known = [r for r in rows if test_M[r, j]]
        if not known:
            continue
        rule = config.LABELS[name]["aggregate"]
        m_row[j] = 1
        y_row[j] = max(test_Y[r, j] for r in known) if rule == "any" else \
            min(test_Y[r, j] for r in known) if rule == "all" else test_Y[known[0], j]
        p_row[j] = AGGREGATE[rule]([test_P[r, j] for r in known])
    section_true.append(y_row)
    section_pred.append(p_row)
    section_mask.append(m_row)

S_Y, S_P, S_M = map(np.array, (section_true, section_pred, section_mask))
section_level = label_metrics((S_P >= thresholds).astype(int), S_Y, S_M, S_P)
print("Section-level (aggregated) test macro-F1:", section_level["macro_f1"])

# --------------------------------------------------
# 6. BEHAVIOURAL TESTS (CHECKLIST-STYLE)
# --------------------------------------------------
# Hand-written sections with a known expected outcome. Not a benchmark -
# a sanity check that the model reacts to the RIGHT content, e.g. that it
# flags "Worked on software development." as too thin.

with open(os.path.join(os.path.dirname(__file__), "rca", "behavioural_tests.json"), "r", encoding="utf-8") as file:
    tests = json.load(file)

_, test_case_probs = predict([case["text"] for case in tests])
behaviour = []
passed_model = passed_rule = total = 0
for case, probs in zip(tests, test_case_probs):
    outcome = {"name": case["name"], "checks": []}
    for name, expected in case["expect"].items():
        j = config.LABEL_NAMES.index(name)
        model_says = int(probs[j] >= thresholds[j])
        rule_says = rule_predict(name, case["text"])
        passed_model += int(model_says == expected)
        passed_rule += int(rule_says == expected)
        total += 1
        outcome["checks"].append({
            "label": name, "expected": expected,
            "model": model_says, "model_probability": round(float(probs[j]), 3),
            "rule": rule_says,
        })
    behaviour.append(outcome)

print(f"Behavioural tests: model {passed_model}/{total}, rules {passed_rule}/{total}")

# --------------------------------------------------
# 7. COMPARISON WITH BASELINES
# --------------------------------------------------

with open(os.path.join(config.RESULTS_DIR, "baseline_results.json"), "r", encoding="utf-8") as file:
    baselines = json.load(file)

methods = {
    "Plain-text rules": baselines["rules"],
    "TF-IDF + LogReg": baselines["tfidf_logreg"],
    "Frozen DistilBERT CLS + LogReg": baselines["frozen_probe"]["cls"]["Layer 6"],
    "Frozen DistilBERT Mean + LogReg": baselines["frozen_probe"]["mean"]["Layer 6"],
    "Fine-tuned DistilBERT": finetuned,
    "Hybrid (fine-tuned + rules)": hybrid,
}

lines = [
    "| Method | Deficiency macro-F1 | Deficiency macro-AUROC | Section-type accuracy |",
    "|---|---:|---:|---:|",
]
for method, result in methods.items():
    auroc = result["labels"].get("macro_auroc")
    type_acc = result.get("section_type", {}).get("accuracy")
    lines.append(f"| {method} | {result['labels']['macro_f1']:.3f} | "
                 f"{auroc if auroc is None else f'{auroc:.3f}'} | "
                 f"{type_acc if type_acc is None else f'{type_acc:.3f}'} |")

lines += ["", "Per-label test F1:", "",
          "| Label | " + " | ".join(methods) + " |",
          "|---|" + "---:|" * len(methods)]
for name in config.LABEL_NAMES:
    lines.append(f"| {name} | " + " | ".join(
        f"{methods[m]['labels']['per_label'][name]['f1']:.3f}" for m in methods) + " |")

comparison_md = "\n".join(lines).replace("None", "n/a")
with open(os.path.join(config.RESULTS_DIR, "model_comparison.md"), "w", encoding="utf-8") as file:
    file.write(comparison_md + "\n")
print("\n" + comparison_md)

# Per-label F1 chart
figure, axis = plt.subplots(figsize=(13, 5))
width = 0.8 / len(methods)
x = np.arange(len(config.LABEL_NAMES))
for k, (method, result) in enumerate(methods.items()):
    axis.bar(x + k * width, [result["labels"]["per_label"][n]["f1"] for n in config.LABEL_NAMES], width, label=method)
axis.set_xticks(x + width * (len(methods) - 1) / 2)
axis.set_xticklabels(config.LABEL_NAMES, rotation=25, ha="right")
axis.set_ylabel("Test F1")
axis.set_title("Per-label F1 on held-out test resumes")
axis.legend(fontsize=8)
axis.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, "per_label_f1_comparison.png"), dpi=200)
plt.close()

# Layer-wise probing chart (frozen) vs fine-tuned
layers = list(baselines["frozen_probe"]["mean"].keys())
figure, axis = plt.subplots(figsize=(9, 4.5))
for pooling in ("cls", "mean"):
    axis.plot(layers, [baselines["frozen_probe"][pooling][l]["labels"]["macro_auroc"] for l in layers],
              marker="o", label=f"Frozen, {pooling.upper()} pooling")
axis.axhline(finetuned["labels"]["macro_auroc"], color="black", linestyle="--", label="Fine-tuned (mean pooling)")
axis.set_ylabel("Test macro-AUROC (deficiency labels)")
axis.set_title("Linear probe per layer (frozen) vs fine-tuned")
axis.legend()
axis.grid(True, alpha=0.3)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, "layerwise_probe_vs_finetuned.png"), dpi=200)
plt.close()

# --------------------------------------------------
# 8. SAVE
# --------------------------------------------------

with open(os.path.join(config.RESULTS_DIR, "finetuned_results.json"), "w", encoding="utf-8") as file:
    json.dump({
        "finetuned_chunk_level": finetuned,
        "finetuned_section_level": section_level,
        "hybrid": hybrid,
        "decision_policy": policy,
        "section_type_confusion": {"order": config.SECTION_TYPES, "matrix": type_confusion},
        "behavioural_tests": {"model_passed": passed_model, "rule_passed": passed_rule,
                              "total_checks": total, "cases": behaviour},
    }, file, indent=4)

print("\nSaved: results/finetuned_results.json, results/model_comparison.md,")
print("       results/per_label_f1_comparison.png, results/layerwise_probe_vs_finetuned.png,")
print("       models/distilbert_rca/decision_policy.json")
print("\n" + "=" * 65)
print("EVALUATION COMPLETED")
print("=" * 65)
