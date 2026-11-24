
import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer

MODEL_NAME = "distilbert/distilbert-base-uncased"
DATASET_PATH = "data/input/Resume.csv"
OUTPUT_FILE = "results/tokenization_output.json"
MAX_LENGTH = 512

print("=" * 60)
print("DISTILBERT RESUME TOKENIZATION")
print("=" * 60)

# 1. Load dataset
print("\nLoading resume dataset...")
df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Total resumes:", len(df))
print("Available categories:", df["Category"].nunique())

# 2. Select one actual resume
row = df.iloc[0]
resume_id = int(row["ID"])
resume_category = row["Category"]
resume_text = str(row["Resume_str"])

print("\nSelected Resume ID:", resume_id)
print("Resume Category:", resume_category)
print("Resume character count:", len(resume_text))

# 3. Load tokenizer
print("\nLoading DistilBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer loaded successfully!")

# 4. Tokenize resume
tokens = tokenizer.tokenize(resume_text)

print("\n" + "=" * 60)
print("TOKENIZATION OUTPUT")
print("=" * 60)

print("\nFirst 50 generated tokens:")
print(tokens[:50])
print("\nTotal tokens before truncation:", len(tokens))

# 5. Convert text into model inputs
encoded_input = tokenizer(
    resume_text,
    add_special_tokens=True,
    max_length=MAX_LENGTH,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt"
)

input_ids = encoded_input["input_ids"]
attention_mask = encoded_input["attention_mask"]

# 6. Display model input information
valid_length = int(attention_mask.sum().item())

valid_ids = input_ids[0][:valid_length].tolist()
encoded_tokens = tokenizer.convert_ids_to_tokens(valid_ids)

print("\n" + "=" * 60)
print("SPECIAL TOKENS")
print("=" * 60)

print("CLS:", tokenizer.cls_token)
print("SEP:", tokenizer.sep_token)
print("PAD:", tokenizer.pad_token)
print("UNK:", tokenizer.unk_token)

print("\n" + "=" * 60)
print("MODEL INPUT INFORMATION")
print("=" * 60)

print("Input IDs shape:", input_ids.shape)
print("Attention Mask shape:", attention_mask.shape)
print("Actual tokens after encoding:", valid_length)
print("Padding tokens:", MAX_LENGTH - valid_length)

print("\nFirst 30 input IDs:")
print(input_ids[0][:30].tolist())

print("\nFirst 30 attention mask values:")
print(attention_mask[0][:30].tolist())

print("\nFirst 50 encoded tokens:")
print(encoded_tokens[:50])

# 7. Decode tokens for verification
decoded_text = tokenizer.decode(
    valid_ids,
    skip_special_tokens=True
)

print("\nDecoded text preview:")
print(decoded_text[:500])

# 8. Save results
os.makedirs("results", exist_ok=True)

output_data = {
    "resume_id": resume_id,
    "resume_category": resume_category,
    "original_text": resume_text,
    "tokens": encoded_tokens,
    "input_ids": valid_ids,
    "attention_mask": attention_mask[0].tolist(),
    "total_tokens_before_truncation": len(tokens),
    "actual_tokens_after_encoding": valid_length,
    "padding_tokens": MAX_LENGTH - valid_length,
    "input_shape": list(input_ids.shape),
    "attention_mask_shape": list(attention_mask.shape),
    "decoded_text": decoded_text
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=4)

print("\nResults saved to:", OUTPUT_FILE)

print("\n" + "=" * 60)
print("TOKENIZATION COMPLETED SUCCESSFULLY")
print("=" * 60)
