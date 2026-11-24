
import os
import json
import torch

from transformers import AutoModel

RESULTS_DIR = "results"
MODEL_NAME = "distilbert/distilbert-base-uncased"

print("=" * 65)
print("CIA1 COMPONENT 2 - PRE-TRAINED DISTILBERT RESULTS")
print("=" * 65)

# --------------------------------------------------
# 1. LOAD SAVED CONFIGURATION
# --------------------------------------------------

with open(
    os.path.join(RESULTS_DIR, "model_configuration.json"),
    "r",
    encoding="utf-8"
) as file:
    model_config = json.load(file)

# --------------------------------------------------
# 2. LOAD INFERENCE SUMMARY
# --------------------------------------------------

with open(
    os.path.join(RESULTS_DIR, "inference_summary.json"),
    "r",
    encoding="utf-8"
) as file:
    inference = json.load(file)

# --------------------------------------------------
# 3. LOAD LAYER-WISE ANALYSIS
# --------------------------------------------------

with open(
    os.path.join(RESULTS_DIR, "layerwise_analysis.json"),
    "r",
    encoding="utf-8"
) as file:
    layer_analysis = json.load(file)

# --------------------------------------------------
# 4. CALCULATE PARAMETER COUNT
# --------------------------------------------------

print("\nCalculating model parameters...")

model = AutoModel.from_pretrained(MODEL_NAME)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

del model

# --------------------------------------------------
# 5. DISPLAY MODEL INFORMATION
# --------------------------------------------------

print("\nMODEL INFORMATION")
print("-" * 50)

print("Model:", model_config["Model Name"])
print("Transformer layers:",
      model_config["Number of Transformer Layers"])
print("Attention heads:",
      model_config["Number of Attention Heads"])
print("Hidden dimension:",
      model_config["Hidden Dimension"])
print("Feed-forward dimension:",
      model_config["Feed Forward Dimension"])
print("Vocabulary size:",
      model_config["Vocabulary Size"])
print("Total parameters:", total_parameters)
print("Model size (FP32):",
      round(total_parameters * 4 / (1024 ** 2), 2), "MB")

# --------------------------------------------------
# 6. DISPLAY DATASET INFORMATION
# --------------------------------------------------

print("\nDATASET INFORMATION")
print("-" * 50)

print("Resume ID:", inference["resume_id"])
print("Resume category:", inference["resume_category"])
print("Input shape:", inference["input_shape"])

# --------------------------------------------------
# 7. DISPLAY MODEL OUTPUTS
# --------------------------------------------------

print("\nPRE-TRAINED MODEL OUTPUTS")
print("-" * 50)

print("Last hidden state:",
      inference["last_hidden_state_shape"])
print("CLS embedding:",
      inference["cls_embedding_shape"])
print("Mean-pooled embedding:",
      inference["mean_embedding_shape"])

# --------------------------------------------------
# 8. DISPLAY LAYER-WISE ANALYSIS
# --------------------------------------------------

print("\nLAYER-WISE ANALYSIS")
print("-" * 50)

print("Transformer layers:",
      layer_analysis["number_of_transformer_layers"])

print("Hidden-state outputs:",
      layer_analysis["number_of_hidden_state_outputs"])

for name, shape in zip(
    layer_analysis["layer_names"],
    layer_analysis["hidden_state_shapes"]
):
    print(name, ":", shape)

# --------------------------------------------------
# 9. CREATE FINAL RESULTS REPORT
# --------------------------------------------------

report = f"""
CIA1 COMPONENT 2
PRE-TRAINED DISTILBERT IMPLEMENTATION RESULTS

1. MODEL DETAILS

Model: {MODEL_NAME}
Architecture: DistilBERT
Transformer layers: {model_config["Number of Transformer Layers"]}
Attention heads: {model_config["Number of Attention Heads"]}
Hidden dimension: {model_config["Hidden Dimension"]}
Feed-forward dimension: {model_config["Feed Forward Dimension"]}
Vocabulary size: {model_config["Vocabulary Size"]}
Maximum sequence length: {model_config["Maximum Sequence Length"]}
Total parameters: {total_parameters}
Trainable parameters before freezing: {trainable_parameters}

2. DATASET DETAILS

Dataset: Resume.csv
Selected Resume ID: {inference["resume_id"]}
Resume Category: {inference["resume_category"]}

3. INPUT ENCODING

Input IDs shape: {inference["input_shape"]}
Attention mask shape: {inference["attention_mask_shape"]}

4. PRE-TRAINED INFERENCE

Last hidden state shape:
{inference["last_hidden_state_shape"]}

CLS embedding shape:
{inference["cls_embedding_shape"]}

Mean-pooled embedding shape:
{inference["mean_embedding_shape"]}

5. TRANSFORMER LAYER ANALYSIS

Number of Transformer layers:
{layer_analysis["number_of_transformer_layers"]}

Number of hidden-state outputs:
{layer_analysis["number_of_hidden_state_outputs"]}

6. TRAINING STATUS

Training performed: No
Fine-tuning performed: No
Original pre-trained weights used: Yes
Inference mode: Evaluation
Parameters frozen during inference: Yes

7. CONCLUSION

The original pre-trained DistilBERT model was successfully
loaded and used to process resume text from Resume.csv.

The resume was tokenized and converted into numerical inputs.
These inputs were passed through all six Transformer layers.

The model generated contextual representations in the form
of the last hidden state, CLS embedding and mean-pooled
embedding.

This implementation demonstrates pre-trained language
representation without task-specific fine-tuning or
resume-completeness classification.
"""

report_path = os.path.join(
    RESULTS_DIR,
    "CIA1_pretrained_results.txt"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report)

# --------------------------------------------------
# 10. VERIFY RESULT FILES
# --------------------------------------------------

required_files = [
    "model_configuration.json",
    "tokenization_output.json",
    "pretrained_embeddings.pt",
    "inference_summary.json",
    "layerwise_analysis.json",
    "layerwise_embedding_analysis.png",
    "CIA1_pretrained_results.txt"
]

print("\nOUTPUT FILE VERIFICATION")
print("-" * 50)

for filename in required_files:

    path = os.path.join(RESULTS_DIR, filename)

    if os.path.exists(path):
        print("[SUCCESS]", filename)
    else:
        print("[MISSING]", filename)

print("\nFinal report saved:", report_path)

print("\n" + "=" * 65)
print("CIA1 RESULTS CONSOLIDATION COMPLETED")
print("=" * 65)
