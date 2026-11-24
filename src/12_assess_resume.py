
import os
import sys
import json

from rca import config
from rca.inference import ResumeAssessor

# --------------------------------------------------
# 1. INPUT
# --------------------------------------------------
# Usage:
#   python src/12_assess_resume.py                       (uses sample_resume.txt)
#   python src/12_assess_resume.py path/to/resume.txt

input_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    config.PROJECT_ROOT, "data", "input", "sample_resume.txt")

with open(input_path, "r", encoding="utf-8") as file:
    resume_text = file.read()

print("=" * 65)
print("RESUME COMPLETENESS ASSESSMENT")
print("=" * 65)
print("Input:", input_path)

# --------------------------------------------------
# 2. ASSESS
# --------------------------------------------------

assessor = ResumeAssessor()
report = assessor.assess(resume_text)

# --------------------------------------------------
# 3. DISPLAY
# --------------------------------------------------

print("\nCONTACT DETAILS (rule-based)")
print("-" * 50)
for item, found in report["contact"].items():
    print(f"  {item:14s}: {'found' if found else 'NOT FOUND'}")

print("\nSECTIONS DETECTED")
print("-" * 50)
for section in report["sections"]:
    heading = section["heading"] or "(no heading)"
    print(f"\n  [{section['type'].upper()}]  heading: {heading}")
    print(f"    model reads it as: {section['model_type']} "
          f"(confidence {section['model_type_confidence']:.2f}), chunks: {section['num_chunks']}")
    for name, decision in section["labels"].items():
        status = "WEAK" if decision["flagged"] else "ok"
        print(f"    {name:28s} {status:5s} P(deficiency)={decision['model_probability']:.2f}"
              f"  [decided by {decision['decided_by']}]")

print("\nREQUIRED SECTIONS MISSING:", report["missing_required"] or "none")
print("OPTIONAL SECTIONS MISSING:", report["missing_optional"] or "none")

print("\nFEEDBACK")
print("-" * 50)
order = {"missing": 0, "weak": 1, "suggestion": 2}
for item in sorted(report["feedback"], key=lambda f: order[f["severity"]]):
    print(f"  [{item['severity'].upper():10s}] {item['section']:14s} {item['message']}")

# --------------------------------------------------
# 4. SAVE
# --------------------------------------------------

name = os.path.splitext(os.path.basename(input_path))[0]
output_path = os.path.join(config.RESULTS_DIR, f"assessment_{name}.json")
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\nSaved:", output_path)
print("\n" + "=" * 65)
print("ASSESSMENT COMPLETED")
print("=" * 65)
