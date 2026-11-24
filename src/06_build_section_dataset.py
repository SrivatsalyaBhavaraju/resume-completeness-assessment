
import os
import sys
import json
import random
import collections

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from rca import config
from rca.html_sections import parse_resume_html
from rca.labels import section_labels
from rca.text_utils import replace_placeholders, normalise_whitespace
from rca.segmenter import build_lexicon

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------
# Chunk budget: 512 positions minus [CLS] and [SEP].
CHUNK_TOKENS = config.MAX_LENGTH - 2

os.makedirs(config.PROCESSED_DIR, exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

print("=" * 65)
print("BUILDING THE SECTION-LEVEL COMPLETENESS DATASET")
print("=" * 65)

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(config.DATASET_PATH)
print("\nResumes:", len(df))

# --------------------------------------------------
# 3. RESUME-LEVEL TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------
# Split by RESUME, not by section: all sections of one resume go to the
# same split, so the model is never tested on a resume it has partly
# seen. Stratified by job category so every category appears in each
# split. 70 / 15 / 15.

train_ids, temp_ids = train_test_split(
    df["ID"], test_size=0.30, stratify=df["Category"], random_state=config.RANDOM_SEED
)
temp_categories = df.set_index("ID").loc[temp_ids, "Category"]
val_ids, test_ids = train_test_split(
    temp_ids, test_size=0.50, stratify=temp_categories, random_state=config.RANDOM_SEED
)

split_of = {}
for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
    for resume_id in ids:
        split_of[int(resume_id)] = name

with open(config.SPLITS_FILE, "w", encoding="utf-8") as file:
    json.dump({
        "train": sorted(int(i) for i in train_ids),
        "val": sorted(int(i) for i in val_ids),
        "test": sorted(int(i) for i in test_ids),
    }, file)

print("Train / Val / Test resumes:", len(train_ids), "/", len(val_ids), "/", len(test_ids))

# --------------------------------------------------
# 4. PARSE HTML INTO SECTIONS
# --------------------------------------------------

print("\nParsing HTML sections...")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

parsed = {}
for row in df.itertuples():
    parsed[int(row.ID)] = parse_resume_html(row.Resume_html)

# --------------------------------------------------
# 5. HEADING LEXICON (TRAINING SPLIT ONLY)
# --------------------------------------------------

title_pairs = [
    (section["title"], section["type"])
    for resume_id, sections in parsed.items()
    if split_of[resume_id] == "train"
    for section in sections
]
lexicon = build_lexicon(title_pairs)

with open(os.path.join(config.PROCESSED_DIR, "heading_lexicon.json"), "w", encoding="utf-8") as file:
    json.dump(lexicon, file, indent=2, sort_keys=True)

print("Heading lexicon entries:", len(lexicon))

# --------------------------------------------------
# 6. BUILD TRAINING UNITS (ENTRY-ALIGNED CHUNKS)
# --------------------------------------------------
# A section longer than 510 tokens is split into chunks made of WHOLE
# entries (whole jobs / whole degrees). Labels are then recomputed on
# exactly the entries inside each chunk, so a chunk is never given a
# label caused by text it does not contain.


def entry_tokens(text):
    return len(tokenizer.tokenize(text))


units = []
truncated_entries = 0

for resume_id, sections in parsed.items():
    rng = random.Random(resume_id)

    for section_index, section in enumerate(sections):
        # Replace anonymisation placeholders once per entry
        entries = []
        for entry in section["entries"]:
            text = normalise_whitespace(replace_placeholders(entry["text"], rng))
            entries.append({"text": text, "fields": entry["fields"], "tokens": entry_tokens(text)})

        chunks, current, current_tokens = [], [], 0
        for entry in entries:
            if current and current_tokens + entry["tokens"] > CHUNK_TOKENS:
                chunks.append(current)
                current, current_tokens = [], 0
            current.append(entry)
            current_tokens += entry["tokens"]
            if entry["tokens"] > CHUNK_TOKENS:
                truncated_entries += 1
        if current:
            chunks.append(current)

        for chunk_index, chunk in enumerate(chunks):
            text = "\n".join(entry["text"] for entry in chunk)
            labels = section_labels(section["type"], chunk, text)
            units.append({
                "resume_id": resume_id,
                "split": split_of[resume_id],
                "section_index": section_index,
                "chunk_index": chunk_index,
                "num_chunks": len(chunks),
                "code": section["code"],
                "type": section["type"],
                "title": section["title"],
                "text": text,
                "tokens": sum(entry["tokens"] for entry in chunk),
                "labels": labels,
            })

with open(config.UNITS_FILE, "w", encoding="utf-8") as file:
    for unit in units:
        file.write(json.dumps(unit, ensure_ascii=False) + "\n")

print("\nTraining units (chunks):", len(units))
print("Entries longer than one chunk (truncated):", truncated_entries)

# --------------------------------------------------
# 7. DATASET STATISTICS
# --------------------------------------------------

stats = {"units_per_split": {}, "types_per_split": {}, "labels": {}, "token_length": {}}

for split in ("train", "val", "test"):
    subset = [u for u in units if u["split"] == split]
    stats["units_per_split"][split] = len(subset)
    stats["types_per_split"][split] = dict(collections.Counter(u["type"] for u in subset))

for name in config.LABEL_NAMES:
    stats["labels"][name] = {}
    for split in ("train", "val", "test"):
        values = [u["labels"][name] for u in units if u["split"] == split and name in u["labels"]]
        known = [v for v in values if v is not None]
        stats["labels"][name][split] = {
            "applicable_units": len(values),
            "unknown": len(values) - len(known),
            "positive": int(sum(known)),
            "positive_rate": round(float(np.mean(known)), 4) if known else None,
        }

lengths = np.array([u["tokens"] for u in units])
stats["token_length"] = {
    "mean": float(lengths.mean()),
    "median": float(np.median(lengths)),
    "p95": float(np.percentile(lengths, 95)),
    "max": int(lengths.max()),
    "over_510": int((lengths > CHUNK_TOKENS).sum()),
}
stats["sections_split_into_multiple_chunks"] = sum(
    1 for u in units if u["chunk_index"] == 0 and u["num_chunks"] > 1
)

with open(os.path.join(config.RESULTS_DIR, "label_statistics.json"), "w", encoding="utf-8") as file:
    json.dump(stats, file, indent=4)

print("\nUnits per split:", stats["units_per_split"])
print("Section types (train):", stats["types_per_split"]["train"])
print("\nLabel prevalence (train):")
for name in config.LABEL_NAMES:
    s = stats["labels"][name]["train"]
    print(f"  {name:26s} positive {s['positive']:5d} / {s['applicable_units'] - s['unknown']:5d}"
          f"  ({s['positive_rate']:.1%})")
print("\nToken length:", stats["token_length"])
print("Sections needing >1 chunk:", stats["sections_split_into_multiple_chunks"])

print("\n" + "=" * 65)
print("DATASET BUILT SUCCESSFULLY")
print("=" * 65)
