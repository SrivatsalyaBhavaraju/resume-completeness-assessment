
import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModel

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "distilbert/distilbert-base-uncased"
DATASET_PATH = "data/input/Resume.csv"
MAX_LENGTH = 512

OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("DISTILBERT LAYER-WISE ANALYSIS")
print("=" * 65)

# --------------------------------------------------
# 2. LOAD RESUME
# --------------------------------------------------

print("\nLoading resume dataset...")

df = pd.read_csv(DATASET_PATH)
resume = df.iloc[0]

resume_text = str(resume["Resume_str"])

print("Resume ID:", resume["ID"])
print("Resume Category:", resume["Category"])

# --------------------------------------------------
# 3. LOAD TOKENIZER AND MODEL
# --------------------------------------------------

print("\nLoading pre-trained DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Freeze parameters: no training
for parameter in model.parameters():
    parameter.requires_grad = False

model.eval()

print("Model loaded successfully!")
print("Training disabled.")

# --------------------------------------------------
# 4. TOKENIZE RESUME
# --------------------------------------------------

encoded_input = tokenizer(
    resume_text,
    max_length=MAX_LENGTH,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt"
)

input_ids = encoded_input["input_ids"]
attention_mask = encoded_input["attention_mask"]

# --------------------------------------------------
# 5. EXTRACT OUTPUT FROM EVERY LAYER
# --------------------------------------------------

print("\nExecuting all Transformer layers...")

with torch.inference_mode():

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )

hidden_states = outputs.hidden_states

print("\nLayer-wise outputs extracted successfully!")
print("Number of hidden-state outputs:", len(hidden_states))

# --------------------------------------------------
# 6. ANALYZE EACH LAYER
# --------------------------------------------------

layer_names = ["Embedding Output"]
layer_norms = []
layer_means = []
layer_stds = []

mask = attention_mask.unsqueeze(-1).float()
valid_tokens = mask.sum()

for index, hidden_state in enumerate(hidden_states):

    if index > 0:
        layer_names.append(f"Transformer Layer {index}")

    # Calculate vector magnitude for each token
    vector_norms = torch.linalg.vector_norm(
        hidden_state,
        dim=-1
    )

    average_norm = (
        (vector_norms * attention_mask).sum() / valid_tokens
    ).item()

    average_value = hidden_state.mean().item()
    standard_deviation = hidden_state.std().item()

    layer_norms.append(average_norm)
    layer_means.append(average_value)
    layer_stds.append(standard_deviation)

    print("\n" + "-" * 50)
    print(layer_names[index])
    print("Output shape:", tuple(hidden_state.shape))
    print("Average vector norm:", round(average_norm, 4))
    print("Mean value:", round(average_value, 4))
    print("Standard deviation:", round(standard_deviation, 4))

# --------------------------------------------------
# 7. VISUALIZE LAYER-WISE OUTPUT
# --------------------------------------------------

plt.figure(figsize=(11, 6))

plt.plot(
    layer_names,
    layer_norms,
    marker="o"
)

plt.title("DistilBERT Layer-wise Embedding Norm")
plt.xlabel("Model Layer")
plt.ylabel("Average Token Vector Norm")

plt.xticks(rotation=25)
plt.grid(True)
plt.tight_layout()

chart_path = os.path.join(
    OUTPUT_DIR,
    "layerwise_embedding_analysis.png"
)

plt.savefig(chart_path, dpi=300)
plt.close()

print("\nVisualization saved:", chart_path)

# --------------------------------------------------
# 8. SAVE ANALYSIS RESULTS
# --------------------------------------------------

analysis = {
    "model": MODEL_NAME,
    "resume_id": int(resume["ID"]),
    "resume_category": str(resume["Category"]),
    "number_of_transformer_layers": model.config.n_layers,
    "number_of_hidden_state_outputs": len(hidden_states),
    "hidden_state_shapes": [
        list(state.shape) for state in hidden_states
    ],
    "layer_names": layer_names,
    "average_vector_norms": layer_norms,
    "mean_values": layer_means,
    "standard_deviations": layer_stds,
    "training_performed": False,
    "fine_tuning_performed": False
}

json_path = os.path.join(
    OUTPUT_DIR,
    "layerwise_analysis.json"
)

with open(json_path, "w", encoding="utf-8") as file:
    json.dump(analysis, file, indent=4)

print("Analysis saved:", json_path)

print("\n" + "=" * 65)
print("LAYER-WISE ANALYSIS COMPLETED")
print("=" * 65)
