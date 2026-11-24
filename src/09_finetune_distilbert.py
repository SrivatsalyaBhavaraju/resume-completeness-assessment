
import os
import json
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score

from rca import config
from rca.data import load_units, to_arrays
from rca.model import ResumeCompletenessModel, masked_bce

# --------------------------------------------------
# 1. TRAINING CONFIGURATION
# --------------------------------------------------
# Values follow the fine-tuning recommendations of the BERT paper
# (Devlin et al., 2019: lr in {2e-5, 3e-5, 5e-5}, 2-4 epochs, warm-up,
# AdamW with weight decay) and the memory limit of a 4 GB GPU.

HPARAMS = {
    "pooling": "mean",           # chosen from the frozen-probe experiment (08)
    "learning_rate": 3e-5,
    "weight_decay": 0.01,
    "epochs": 4,
    "batch_size": 8,
    "grad_accumulation": 2,      # effective batch size 16
    "warmup_ratio": 0.06,
    "max_grad_norm": 1.0,
    "dropout": 0.1,
    "max_length": config.MAX_LENGTH,
    "mixed_precision": True,
    "seed": config.RANDOM_SEED,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(HPARAMS["seed"])
np.random.seed(HPARAMS["seed"])
torch.manual_seed(HPARAMS["seed"])

print("=" * 65)
print("FINE-TUNING DISTILBERT FOR SECTION COMPLETENESS")
print("=" * 65)
print("Device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

# --------------------------------------------------
# 2. DATA
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

train_units, val_units = load_units("train"), load_units("val")
train_texts, train_types, train_Y, train_M = to_arrays(train_units)
val_texts, val_types, val_Y, val_M = to_arrays(val_units)
print("Train units:", len(train_units), "| Val units:", len(val_units))


def length_bucketed_batches(lengths, batch_size, shuffle=True):
    """
    Group units of similar length into the same batch so that little
    compute is wasted on padding (sections vary from 5 to 510 tokens).
    Batches are still shuffled, so training order stays random.
    """
    indices = np.random.permutation(len(lengths)) if shuffle else np.arange(len(lengths))
    pool = batch_size * 50
    batches = []
    for start in range(0, len(indices), pool):
        chunk = sorted(indices[start:start + pool], key=lambda i: lengths[i])
        batches += [chunk[k:k + batch_size] for k in range(0, len(chunk), batch_size)]
    if shuffle:
        random.shuffle(batches)
    return batches


def encode(texts):
    return tokenizer(texts, truncation=True, max_length=HPARAMS["max_length"],
                     padding=True, return_tensors="pt").to(DEVICE)


train_lengths = [u["tokens"] for u in train_units]
val_lengths = [u["tokens"] for u in val_units]

# --------------------------------------------------
# 3. MODEL, OPTIMISER, SCHEDULE
# --------------------------------------------------

model = ResumeCompletenessModel(pooling=HPARAMS["pooling"], dropout=HPARAMS["dropout"]).to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,} (full fine-tuning: encoder + both heads)")

# No weight decay on biases and LayerNorm weights (standard practice)
decay, no_decay = [], []
for name, parameter in model.named_parameters():
    (no_decay if "bias" in name or "LayerNorm" in name or "layer_norm" in name else decay).append(parameter)

optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": HPARAMS["weight_decay"]},
     {"params": no_decay, "weight_decay": 0.0}],
    lr=HPARAMS["learning_rate"],
)

steps_per_epoch = int(np.ceil(len(train_units) / HPARAMS["batch_size"] / HPARAMS["grad_accumulation"]))
total_steps = steps_per_epoch * HPARAMS["epochs"]
scheduler = get_linear_schedule_with_warmup(optimizer, int(HPARAMS["warmup_ratio"] * total_steps), total_steps)
scaler = torch.amp.GradScaler(enabled=HPARAMS["mixed_precision"] and DEVICE == "cuda")

# --------------------------------------------------
# 4. VALIDATION
# --------------------------------------------------


@torch.no_grad()
def evaluate():
    model.eval()
    type_logits_all = np.zeros((len(val_texts), len(config.SECTION_TYPES)))
    label_probs = np.zeros_like(val_Y)
    total_loss, batches = 0.0, 0
    for batch in length_bucketed_batches(val_lengths, 32, shuffle=False):
        inputs = encode([val_texts[i] for i in batch])
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=HPARAMS["mixed_precision"] and DEVICE == "cuda"):
            type_logits, label_logits = model(inputs["input_ids"], inputs["attention_mask"])
        type_logits, label_logits = type_logits.float(), label_logits.float()
        y = torch.tensor(val_Y[batch], device=DEVICE)
        m = torch.tensor(val_M[batch], device=DEVICE)
        t = torch.tensor(val_types[batch], device=DEVICE)
        total_loss += (F.cross_entropy(type_logits, t) + masked_bce(label_logits, y, m)).item()
        batches += 1
        type_logits_all[batch] = type_logits.cpu().numpy()
        label_probs[batch] = torch.sigmoid(label_logits).cpu().numpy()

    aucs = []
    for j in range(val_Y.shape[1]):
        rows = val_M[:, j] == 1
        aucs.append(roc_auc_score(val_Y[rows, j], label_probs[rows, j]))
    type_acc = float((type_logits_all.argmax(1) == val_types).mean())
    model.train()
    return total_loss / batches, float(np.mean(aucs)), type_acc


# --------------------------------------------------
# 5. TRAINING LOOP
# --------------------------------------------------

history = []
best_auroc = -1.0
model.train()
start_time = time.time()

for epoch in range(1, HPARAMS["epochs"] + 1):
    running, seen = 0.0, 0
    batches = length_bucketed_batches(train_lengths, HPARAMS["batch_size"])

    for step, batch in enumerate(batches, start=1):
        inputs = encode([train_texts[i] for i in batch])
        y = torch.tensor(train_Y[batch], device=DEVICE)
        m = torch.tensor(train_M[batch], device=DEVICE)
        t = torch.tensor(train_types[batch], device=DEVICE)

        with torch.autocast(DEVICE, dtype=torch.float16, enabled=HPARAMS["mixed_precision"] and DEVICE == "cuda"):
            type_logits, label_logits = model(inputs["input_ids"], inputs["attention_mask"])
        # Losses computed in float32 for numerical stability
        loss = F.cross_entropy(type_logits.float(), t) + masked_bce(label_logits.float(), y, m)

        scaler.scale(loss / HPARAMS["grad_accumulation"]).backward()

        if step % HPARAMS["grad_accumulation"] == 0 or step == len(batches):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), HPARAMS["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        running += loss.item()
        seen += 1
        if step % 200 == 0:
            print(f"  epoch {epoch} step {step}/{len(batches)} train loss {running / seen:.4f}")

    val_loss, val_auroc, val_type_acc = evaluate()
    record = {
        "epoch": epoch,
        "train_loss": round(running / seen, 4),
        "val_loss": round(val_loss, 4),
        "val_macro_auroc": round(val_auroc, 4),
        "val_type_accuracy": round(val_type_acc, 4),
        "elapsed_min": round((time.time() - start_time) / 60, 1),
    }
    history.append(record)
    print(f"Epoch {epoch}: {record}")

    # Model selection on a threshold-free validation metric
    if val_auroc > best_auroc:
        best_auroc = val_auroc
        model.save(config.FINETUNED_DIR, extra={"hparams": HPARAMS, "best_epoch": epoch})
        tokenizer.save_pretrained(config.FINETUNED_DIR)
        print(f"  -> new best model saved (val macro-AUROC {val_auroc:.4f})")

# --------------------------------------------------
# 6. SAVE HISTORY AND TRAINING CURVES
# --------------------------------------------------

with open(os.path.join(config.RESULTS_DIR, "training_history.json"), "w", encoding="utf-8") as file:
    json.dump({"hparams": HPARAMS, "history": history, "best_val_macro_auroc": best_auroc}, file, indent=4)

epochs = [h["epoch"] for h in history]
figure, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
axes[0].plot(epochs, [h["val_loss"] for h in history], marker="o", label="validation")
axes[0].set_title("Loss (CE type + masked BCE deficiencies)")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(True)
axes[1].plot(epochs, [h["val_macro_auroc"] for h in history], marker="o", label="deficiency macro-AUROC")
axes[1].plot(epochs, [h["val_type_accuracy"] for h in history], marker="o", label="section-type accuracy")
axes[1].set_title("Validation metrics")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(True)
plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, "training_curves.png"), dpi=200)
plt.close()

print("\nSaved: models/distilbert_rca/, results/training_history.json, results/training_curves.png")
print("\n" + "=" * 65)
print("FINE-TUNING COMPLETED")
print("=" * 65)
