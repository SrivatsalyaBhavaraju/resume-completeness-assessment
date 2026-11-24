
import os
import json
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModel

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "distilbert/distilbert-base-uncased"
DATASET_PATH = "data/input/Resume.csv"
OUTPUT_DIR = "results"
MAX_LENGTH = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("PRE-TRAINED DISTILBERT INFERENCE")
print("=" * 65)

# --------------------------------------------------
# 2. LOAD ACTUAL RESUME DATASET
# --------------------------------------------------

print("\nLoading resume dataset...")

df = pd.read_csv(DATASET_PATH)

resume = df.iloc[0]

resume_id = int(resume["ID"])
resume_category = resume["Category"]
resume_text = str(resume["Resume_str"])

print("Dataset loaded successfully!")
print("Resume ID:", resume_id)
print("Resume Category:", resume_category)
print("Resume characters:", len(resume_text))

# --------------------------------------------------
# 3. LOAD PRE-TRAINED TOKENIZER
# --------------------------------------------------

print("\nLoading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded successfully!")

# --------------------------------------------------
# 4. ENCODE RESUME TEXT
# --------------------------------------------------

print("\nEncoding resume text...")

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

print("Input IDs shape:", input_ids.shape)
print("Attention Mask shape:", attention_mask.shape)

# --------------------------------------------------
# 5. LOAD ORIGINAL PRE-TRAINED DISTILBERT
# --------------------------------------------------

print("\nLoading original pre-trained DistilBERT...")

model = AutoModel.from_pretrained(MODEL_NAME)

# Freeze all parameters: no training or fine-tuning
for parameter in model.parameters():
    parameter.requires_grad = False

model.eval()

print("Pre-trained DistilBERT loaded successfully!")
print("Model is in evaluation mode.")
print("All model parameters are frozen.")

# --------------------------------------------------
# 6. PASS INPUT THROUGH TRANSFORMER
# --------------------------------------------------

print("\nPassing resume through Transformer layers...")

with torch.inference_mode():

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

last_hidden_state = outputs.last_hidden_state

print("\nTransformer processing completed!")

print("Last Hidden State Shape:", last_hidden_state.shape)

# --------------------------------------------------
# 7. EXTRACT CLS TOKEN EMBEDDING
# --------------------------------------------------

cls_embedding = last_hidden_state[:, 0, :]

print("\nCLS TOKEN EMBEDDING")
print("CLS Embedding Shape:", cls_embedding.shape)

print("\nFirst 10 CLS embedding values:")
print(cls_embedding[0][:10].tolist())

# --------------------------------------------------
# 8. EXTRACT MEAN-POOLED EMBEDDING
# --------------------------------------------------

mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)

sum_embeddings = (last_hidden_state * mask).sum(dim=1)

token_counts = mask.sum(dim=1).clamp(min=1)

mean_embedding = sum_embeddings / token_counts

print("\nMEAN-POOLED EMBEDDING")
print("Mean Embedding Shape:", mean_embedding.shape)

print("\nFirst 10 Mean Embedding Values:")
print(mean_embedding[0][:10].tolist())

# --------------------------------------------------
# 9. SAVE EMBEDDINGS
# --------------------------------------------------

torch.save(
    {
        "resume_id": resume_id,
        "resume_category": resume_category,
        "last_hidden_state": last_hidden_state.cpu(),
        "cls_embedding": cls_embedding.cpu(),
        "mean_embedding": mean_embedding.cpu()
    },
    os.path.join(OUTPUT_DIR, "pretrained_embeddings.pt")
)

# Save readable summary
summary = {
    "model_name": MODEL_NAME,
    "resume_id": resume_id,
    "resume_category": resume_category,
    "input_shape": list(input_ids.shape),
    "attention_mask_shape": list(attention_mask.shape),
    "last_hidden_state_shape": list(last_hidden_state.shape),
    "cls_embedding_shape": list(cls_embedding.shape),
    "mean_embedding_shape": list(mean_embedding.shape),
    "first_10_cls_values": cls_embedding[0][:10].tolist(),
    "first_10_mean_values": mean_embedding[0][:10].tolist(),
    "training_performed": False,
    "fine_tuning_performed": False
}

with open(
    os.path.join(OUTPUT_DIR, "inference_summary.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(summary, file, indent=4)

print("\nEmbeddings saved successfully!")
print("File: results/pretrained_embeddings.pt")
print("File: results/inference_summary.json")

print("\n" + "=" * 65)
print("PRE-TRAINED INFERENCE COMPLETED SUCCESSFULLY")
print("=" * 65)
