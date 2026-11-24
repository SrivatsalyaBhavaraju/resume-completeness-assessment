
import os
import json
import time

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

from rca import config
from rca.data import load_units, to_arrays
from rca.evaluation import tune_thresholds, label_metrics, type_metrics
from rca.model import mean_pool
from rca.rules import rule_predict

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------
# Three baselines, evaluated exactly like the fine-tuned model:
#   A. plain-text rules                       (no learning)
#   B. TF-IDF + logistic regression           (classical ML, as in the
#                                              surveyed literature)
#   C. FROZEN pre-trained DistilBERT features + logistic regression
#      (= our Review-1 implementation, used as a feature extractor),
#      for every one of the 7 hidden-state outputs, CLS and mean pooling.

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)

print("=" * 65)
print("BASELINES: RULES, TF-IDF, FROZEN DISTILBERT PROBES")
print("=" * 65)

# --------------------------------------------------
# 2. LOAD UNITS
# --------------------------------------------------

splits = {name: load_units(name) for name in ("train", "val", "test")}
arrays = {name: to_arrays(units) for name, units in splits.items()}
for name, units in splits.items():
    print(f"{name:5s} units: {len(units)}")

results = {}


def make_classifier(X_train):
    """
    Dense DistilBERT features are standardised before logistic regression.
    Sparse TF-IDF features are already length-normalised and must NOT be
    standardised (that would blow up rare n-grams), so they go in as-is.
    """
    if hasattr(X_train, "tocsr"):
        return LogisticRegression(C=1.0, max_iter=3000)
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))


def fit_and_score(features):
    """
    One logistic regression per label (trained only on units where the
    label applies), plus one multinomial LR for section type.
    Thresholds tuned on validation, metrics reported on test.
    """
    X = features
    _, type_train, Y_train, M_train = arrays["train"]
    _, type_val, Y_val, M_val = arrays["val"]
    _, type_test, Y_test, M_test = arrays["test"]

    P_val = np.zeros_like(Y_val)
    P_test = np.zeros_like(Y_test)

    for j in range(len(config.LABEL_NAMES)):
        rows = M_train[:, j] == 1
        classifier = make_classifier(X["train"])
        classifier.fit(X["train"][rows], Y_train[rows, j])
        P_val[:, j] = classifier.predict_proba(X["val"])[:, 1]
        P_test[:, j] = classifier.predict_proba(X["test"])[:, 1]

    thresholds = tune_thresholds(P_val, Y_val, M_val)
    labels = label_metrics((P_test >= thresholds).astype(int), Y_test, M_test, P_test)

    type_classifier = make_classifier(X["train"])
    type_classifier.fit(X["train"], type_train)
    types = type_metrics(type_classifier.predict(X["test"]), type_test)

    return {"labels": labels, "section_type": types, "thresholds": thresholds.round(2).tolist()}


# --------------------------------------------------
# 3. BASELINE A - PLAIN-TEXT RULES
# --------------------------------------------------

print("\n[A] Plain-text rules...")
texts_test, _, Y_test, M_test = arrays["test"]
R = np.zeros_like(Y_test)
for i, text in enumerate(texts_test):
    for j, name in enumerate(config.LABEL_NAMES):
        if M_test[i, j]:
            R[i, j] = rule_predict(name, text)
results["rules"] = {"labels": label_metrics(R.astype(int), Y_test, M_test)}
print("    macro-F1:", results["rules"]["labels"]["macro_f1"])

# --------------------------------------------------
# 4. BASELINE B - TF-IDF + LOGISTIC REGRESSION
# --------------------------------------------------

print("\n[B] TF-IDF + logistic regression...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
tfidf = {"train": vectorizer.fit_transform(arrays["train"][0])}
tfidf["val"] = vectorizer.transform(arrays["val"][0])
tfidf["test"] = vectorizer.transform(arrays["test"][0])
results["tfidf_logreg"] = fit_and_score(tfidf)
print("    macro-F1:", results["tfidf_logreg"]["labels"]["macro_f1"],
      "| type acc:", results["tfidf_logreg"]["section_type"]["accuracy"])

# --------------------------------------------------
# 5. BASELINE C - FROZEN DISTILBERT FEATURES (ALL LAYERS)
# --------------------------------------------------

print("\n[C] Extracting frozen DistilBERT features on", DEVICE, "...")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
encoder = AutoModel.from_pretrained(config.MODEL_NAME).to(DEVICE).eval()
for parameter in encoder.parameters():
    parameter.requires_grad = False

NUM_STATES = encoder.config.n_layers + 1  # embeddings + 6 layers = 7


def extract(texts):
    """Returns cls[7, N, 768] and mean[7, N, 768] (float16)."""
    order = np.argsort([len(t) for t in texts])
    cls = np.zeros((NUM_STATES, len(texts), 768), dtype=np.float16)
    mean = np.zeros_like(cls)
    with torch.inference_mode(), torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
        for start in range(0, len(texts), BATCH_SIZE):
            index = order[start:start + BATCH_SIZE]
            batch = tokenizer([texts[i] for i in index], truncation=True,
                              max_length=config.MAX_LENGTH, padding=True, return_tensors="pt").to(DEVICE)
            states = encoder(**batch, output_hidden_states=True).hidden_states
            for layer, state in enumerate(states):
                cls[layer, index] = state[:, 0].float().cpu().numpy()
                mean[layer, index] = mean_pool(state.float(), batch["attention_mask"]).cpu().numpy()
    return cls, mean


features = {}
start_time = time.time()
for name in ("train", "val", "test"):
    features[name] = extract(arrays[name][0])
print(f"    feature extraction took {time.time() - start_time:.0f}s")

layer_names = ["Embeddings"] + [f"Layer {i}" for i in range(1, NUM_STATES)]
results["frozen_probe"] = {"cls": {}, "mean": {}}

for pool_index, pooling in enumerate(("cls", "mean")):
    for layer in range(NUM_STATES):
        X = {name: features[name][pool_index][layer].astype(np.float32) for name in features}
        scores = fit_and_score(X)
        results["frozen_probe"][pooling][layer_names[layer]] = scores
        print(f"    {pooling:4s} {layer_names[layer]:10s} macro-F1 {scores['labels']['macro_f1']:.4f}"
              f" | macro-AUROC {scores['labels']['macro_auroc']:.4f}"
              f" | type acc {scores['section_type']['accuracy']:.4f}")

# --------------------------------------------------
# 6. SAVE
# --------------------------------------------------

with open(os.path.join(config.RESULTS_DIR, "baseline_results.json"), "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nSaved: results/baseline_results.json")
print("\n" + "=" * 65)
print("BASELINES COMPLETED")
print("=" * 65)
