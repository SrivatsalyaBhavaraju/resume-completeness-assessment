import torch
import json
import os
from transformers import AutoTokenizer, AutoModel

# --------------------------------------------------
# 1. MODEL SELECTION
# --------------------------------------------------

MODEL_NAME = "distilbert/distilbert-base-uncased"

print("=" * 60)
print("PRE-TRAINED DISTILBERT MODEL IMPLEMENTATION")
print("=" * 60)

# --------------------------------------------------
# 2. LOAD PRE-TRAINED TOKENIZER
# --------------------------------------------------

print("\nLoading DistilBERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded successfully!")

# --------------------------------------------------
# 3. LOAD PRE-TRAINED MODEL
# --------------------------------------------------

print("\nLoading original pre-trained DistilBERT model...")

model = AutoModel.from_pretrained(MODEL_NAME)

# Set model to evaluation mode
model.eval()

print("Pre-trained DistilBERT loaded successfully!")

# --------------------------------------------------
# 4. EXTRACT MODEL CONFIGURATION
# --------------------------------------------------

config = model.config

print("\n" + "=" * 60)
print("DISTILBERT ARCHITECTURE DETAILS")
print("=" * 60)

architecture_details = {
    "Model Name": MODEL_NAME,
    "Model Type": config.model_type,
    "Number of Transformer Layers": config.n_layers,
    "Number of Attention Heads": config.n_heads,
    "Hidden Dimension": config.dim,
    "Feed Forward Dimension": config.hidden_dim,
    "Vocabulary Size": config.vocab_size,
    "Maximum Sequence Length": config.max_position_embeddings,
    "Number of Labels": config.num_labels
}

for key, value in architecture_details.items():
    print(f"{key}: {value}")

# --------------------------------------------------
# 5. CALCULATE TOTAL MODEL PARAMETERS
# --------------------------------------------------

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\n" + "=" * 60)
print("MODEL PARAMETER INFORMATION")
print("=" * 60)

print(f"Total Parameters: {total_parameters:,}")
print(f"Trainable Parameters: {trainable_parameters:,}")
print(f"Model Size: {total_parameters * 4 / (1024 ** 2):.2f} MB (FP32 estimate)")

# --------------------------------------------------
# 6. TOKENIZER INFORMATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("TOKENIZER INFORMATION")
print("=" * 60)

print("Tokenizer Vocabulary:", tokenizer.vocab_size)
print("Padding Token:", tokenizer.pad_token)
print("Unknown Token:", tokenizer.unk_token)
print("Classification Token:", tokenizer.cls_token)
print("Separator Token:", tokenizer.sep_token)

# --------------------------------------------------
# 7. DISPLAY MODEL ARCHITECTURE
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)

print(model)

# --------------------------------------------------
# 8. SAVE MODEL CONFIGURATION
# --------------------------------------------------

os.makedirs("results", exist_ok=True)

with open(
    "results/model_configuration.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(architecture_details, file, indent=4)

print("\nModel configuration saved successfully!")

print("\n" + "=" * 60)
print("MODEL INSPECTION COMPLETED")
print("=" * 60)