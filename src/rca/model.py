"""
DistilBERT Resume Completeness model.

                 section text (heading removed, <= 512 tokens)
                                   |
                       DistilBERT encoder (6 layers)
                                   |
                  last hidden state  [batch, seq, 768]
                                   |
               attention-mask-aware mean pooling  [batch, 768]
                          /                    \\
          Section-type head                 Deficiency head
     Linear(768 -> 6), softmax       Linear(768 -> 9), sigmoid per label
     (summary, education, ...)       (multi-label: edu_missing_degree, ...)
       Cross-Entropy loss              MASKED Binary Cross-Entropy loss

Total loss = CE(type) + BCE(deficiencies), BCE averaged only over the
(unit, label) pairs where the label applies to that section type and is
known (mask M from rca/data.py).

Why one shared encoder with two heads (multi-task learning):
  * both tasks need the same skill - reading resume content in context;
  * the type head lets the system recognise a section from its content
    when the heading is missing or non-standard;
  * one encoder = one forward pass at inference time.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from rca import config


def mean_pool(last_hidden_state, attention_mask):
    """Average token vectors, excluding padding positions."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


class ResumeCompletenessModel(nn.Module):

    def __init__(self, model_name=config.MODEL_NAME, pooling="mean", dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        hidden = self.encoder.config.dim
        self.dropout = nn.Dropout(dropout)
        self.type_head = nn.Linear(hidden, len(config.SECTION_TYPES))
        self.label_head = nn.Linear(hidden, len(config.LABEL_NAMES))

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            pooled = mean_pool(hidden, attention_mask)
        pooled = self.dropout(pooled)
        return self.type_head(pooled), self.label_head(pooled)

    # --------------------------------------------------
    # SAVE / LOAD
    # --------------------------------------------------

    def save(self, directory, extra=None):
        os.makedirs(directory, exist_ok=True)
        self.encoder.save_pretrained(directory)
        torch.save(
            {"type_head": self.type_head.state_dict(), "label_head": self.label_head.state_dict()},
            os.path.join(directory, "heads.pt"),
        )
        meta = {
            "pooling": self.pooling,
            "section_types": config.SECTION_TYPES,
            "label_names": config.LABEL_NAMES,
        }
        meta.update(extra or {})
        with open(os.path.join(directory, "rca_config.json"), "w", encoding="utf-8") as file:
            json.dump(meta, file, indent=4)

    @classmethod
    def load(cls, directory, device="cpu"):
        with open(os.path.join(directory, "rca_config.json"), "r", encoding="utf-8") as file:
            meta = json.load(file)
        model = cls(model_name=directory, pooling=meta["pooling"])
        heads = torch.load(os.path.join(directory, "heads.pt"), map_location=device)
        model.type_head.load_state_dict(heads["type_head"])
        model.label_head.load_state_dict(heads["label_head"])
        return model.to(device).eval(), meta


def masked_bce(logits, targets, mask):
    """Binary cross-entropy averaged over applicable, known labels only."""
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)
