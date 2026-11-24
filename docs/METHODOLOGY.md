# Methodology: Fine-tuning DistilBERT for Section-wise Resume Completeness Assessment

This document explains **what** we built after Review 1, **why** each
decision was made, **what alternatives exist**, and **what the evidence
shows**. Every number below comes from a script in `src/` and a file in
`results/`. Nothing is quoted from the original DistilBERT paper as if it
were ours.

---

## 0. Where Review 1 ended, and what was added

| Review 1 (PPT) | Now |
|---|---|
| Frozen pre-trained DistilBERT, one resume, CLS / mean embeddings | Frozen DistilBERT kept as a **measured baseline**, probed at all 7 layers on 16,771 sections |
| "Section-level labels must be prepared" | 9 deficiency labels derived from the dataset's HTML structure (**distant supervision**) |
| "Add a classification head" | Section-type head (6 classes) + **multi-label deficiency head** (9 labels) |
| "Fine-tune with BCE" | Full fine-tuning, **masked** BCE + cross-entropy, 4 epochs |
| "Handle long resumes (chunking)" | Entry-aligned chunks for training; role-aligned chunks + quantifier-aware aggregation at inference |
| "DistilBERT + rule-based validation" | A **per-label hybrid policy chosen on validation data** |
| "Generate feedback" | Template-based feedback, each message traceable to one prediction |

### Improvements over the PPT proposal (and why)

1. **Labels without manual annotation of 2,484 resumes.** See §2. This was
   the project's main blocker.
2. **A deficiency-type label space instead of a single complete / weak flag
   per section.** "Experience: weak" is not actionable. "Experience: dates
   missing + no quantified results" is. The labels map one-to-one to
   feedback messages.
3. **Section-type head (multi-task).** Lets the system recognise a section
   from its *content* when the heading is missing or unusual.
4. **The hybrid is evidence-based.** Rather than deciding by intuition
   which checks are "rules" and which are "model", we measure both on
   validation and pick the better one per label (§7).
5. **Frozen-vs-fine-tuned and CLS-vs-mean are now experiments, not claims**
   (§6.2).

---

## 1. What the dataset actually contains (findings)

`Resume.csv`: 2,484 resumes, 24 job categories, columns `ID`,
`Resume_str`, `Resume_html`, `Category`.

Key findings from inspecting it:

1. **`Resume_html` is structured.** Every section is a
   `<div id="SECTION_EDUC…">` with a builder-assigned code (EDUC, EXPR,
   SKLL, SUMM, HILT, CERT…). Inside, every entry tags its fields:
   `degree`, `programline`, `companyname_educ`, `jobdates`, `jobtitle`,
   `companyname`, and `jobline` (description bullets).
2. **Empty fields are visible.** A blank degree field or the builder
   placeholder "Select One" means the candidate did not fill it in. That
   is an objective fact about completeness.
3. **The data is anonymised.** Every employer is "Company Name" and every
   location is "City , State". There are no e-mails or phone numbers.
4. **Almost no Projects sections** (3 of 2,484). These are experienced US
   professionals, not students.
5. **Resumes are long.** Mean 6,295 characters ≈ 1,300 tokens, far above
   512, so whole-resume input would be truncated (§4).

Consequences: contact information **cannot be learned** from this data, so
it is checked by rules (regex). Projects **cannot be learned** either, so
they reuse the experience-description labels (an untested transfer,
stated as such).

---

## 2. Labelling strategy: distant supervision from HTML structure

### 2.1 The problem
Supervised fine-tuning needs labels. `Category` is a **job** label, not a
completeness label (PPT §8). Options considered:

| Option | Verdict |
|---|---|
| Manually label ~10,000 sections | Gold standard, but infeasible for a 4-person team in the time available |
| Keyword rules on the text, then train on them | **Circular**: the model only re-learns the rules on its own input, and can never beat them |
| LLM-generated labels | Costly, hard to audit, and still needs human validation |
| **Labels from the HTML field markup** ✅ | Objective, free, available for every resume; the model never sees the markup |

### 2.2 Why it is not circular
- The **label source** is the structured HTML (field spans).
- The **model input** is the **plain text** of the section, with the
  heading removed.

At inference time a real resume has no HTML. The model must therefore
learn, from raw text alone, what the builder recorded structurally. For
example, it has to tell that an entry has no degree when the only cue is
the absence of a qualification phrase among a school name, dates and
grades. A keyword rule cannot do this well: the plain-text rule scores
F1 0.09 on `edu_missing_degree`, against 0.62 for the fine-tuned model
(§6). This is the standard *distant supervision* idea (Mintz et al., 2009):
an external structured source labels raw text.

### 2.3 The nine labels (`src/rca/config.py`, `src/rca/labels.py`)

1 = deficiency present. Each label applies to one section type.

| Label | Section | Definition | Quantifier |
|---|---|---|---|
| `edu_missing_degree` | Education | some entry has no degree / qualification | any |
| `edu_missing_institution` | Education | some entry has no institution | any |
| `edu_missing_date` | Education | some entry has no date | any |
| `exp_missing_dates` | Experience | some role has no employment dates | any |
| `exp_missing_title` | Experience | some role has no job title | any |
| `exp_thin_description` | Experience | some role's description < 15 words | any |
| `exp_no_quantified_impact` | Experience | no number / % / amount in any description (years excluded) | all |
| `summary_too_brief` | Summary | < 20 words | – |
| `skills_too_few` | Skills | < 5 distinct items | – |

**Unknown is not negative.** If an entry has no structured fields at all,
its labels are `None` and **masked out of the loss** rather than treated
as 0. Treating unknown as negative would teach the model wrong answers.

### 2.4 Thresholds (15 words, 20 words, 5 skills)
Chosen from the dataset's distribution (`results/label_statistics.json`)
together with resume-writing guidance:

| Quantity | 5th pct | 10th pct | median |
|---|---|---|---|
| words per role description | 9 | 23 | 78 |
| summary words | 16 | 21 | 48 |
| skill items | 3 | 5 | 14 |

So "thin" means roughly **the bottom 7–10 % of a corpus of professionally
written resumes**. These are design decisions and can be changed in
`config.py`.

### 2.5 Label quality checks we did
We read flagged examples for every label and fixed two parser bugs found
that way: descriptions in generic `field` spans were missed, and HILT
blocks titled "Achievements" were being treated as skills. Remaining label
noise is real: e.g. a role whose title sits in free text is labelled
"missing title". That noise is the main reason `exp_missing_title` scores
lowest. **Human validation is prepared** (`docs/ANNOTATION_GUIDELINES.md`,
`src/13_human_validation.py`): 160 sections, two annotators, Cohen's κ.

---

## 3. Data preparation (`src/06_build_section_dataset.py`)

| Step | Decision | Why |
|---|---|---|
| Split | 70/15/15 **by resume**, stratified by category (1,738 / 373 / 373) | Sections of one resume must never appear in both train and test (leakage) |
| Unit | One section (or chunk of one), **heading removed** | Our task is section-wise, and removing the heading forces the model to read the content |
| Placeholders | "Company Name", "City , State" → realistic values (seeded) | Otherwise the model learns an anonymisation artifact that never occurs in real resumes |
| Chunking | Pack **whole entries** into ≤ 510 tokens; recompute labels per chunk | A chunk is never labelled with a problem caused by text it doesn't contain |
| Lexicon | Heading → type mapping learned from **training** headings only | Data-driven, and test segmentation scores are not inflated |

Result: **16,771 units** (train 11,698 / val 2,587 / test 2,486). Median
78 tokens, 95th percentile 481. 1,711 sections needed more than one chunk.
254 single entries (1.5 %) exceed 510 tokens and are truncated (a known
limitation).

Train prevalence: 6.1 % (`exp_missing_title`) to 43.8 %
(`edu_missing_date`).

---

## 4. Long resumes

Whole-resume input is not viable: mean ≈ 1,300 tokens versus a 512-token
limit, so most content would be cut off. Working **per section** solves
most of this, because median sections are 78 tokens. For sections that
are still too long:

- **Training:** entry-aligned chunks (above).
- **Inference:** the experience text is split into **roles** using the date
  anchor every dated role carries ("01/2014 to Current", "Jul 2010"),
  with the title fragment before it, and whole roles are packed into
  chunks. Other sections are packed line by line.
- **Aggregation follows the label's logic:** "any entry" labels use the
  **max** probability over chunks, "no entry anywhere" uses the **min**, and
  whole-section labels use the **mean**.

**Evidence that this matters.** With naive chunking (cutting text
anywhere), end-to-end precision on validation for the experience labels
was only 0.12–0.18 for multi-chunk sections, against 0.50–0.62 for
single-chunk ones. A role cut in half looks "undated" or "thin".
Role-aligned chunking, designed and checked **on validation only**, raised
test end-to-end macro-F1 from 0.620 to **0.656**.

We also tested **one role per chunk** (instead of packing whole roles up
to 510 tokens). On validation it was worse (mean experience F1 0.531 vs
0.617): more chunks plus max-aggregation means more false alarms. So
packing whole roles is kept. A blank line is also used as a role boundary,
for undated roles in real resumes; the dataset text has none.

Alternatives: sliding windows with overlap (still cuts roles), and
Longformer/BigBird (longer context, but more memory, and the problem is
the label logic rather than context length).

---

## 5. Model (`src/rca/model.py`, `src/09_finetune_distilbert.py`)

```
section text ──► DistilBERT (6 layers, 66.4 M params, all trainable)
             ──► masked mean pooling  [768]
             ├─► type head   Linear(768→6)  softmax  ─ cross-entropy
             └─► label head  Linear(768→9)  sigmoid  ─ masked BCE
loss = CE(type) + BCE(labels; only applicable & known labels)
```

Trainable parameters: 66,374,415 = 66,362,880 (DistilBERT) + 4,614
(type head) + 6,921 (label head).

| Decision | Choice | Justification |
|---|---|---|
| Sequence vs token classification | **Sequence** (per section) | Completeness is a property of a section, not of individual tokens |
| Multi-label + sigmoid + BCE | yes | Deficiencies co-occur (a role can lack dates *and* numbers); softmax would force exactly one |
| Masked BCE | yes | Labels only apply to their section type; unknown labels carry no signal |
| Pooling | **Mean** (masked) | DistilBERT was pre-trained **without** next-sentence prediction, so [CLS] was never trained as a sentence summary. In our frozen probe, mean beats CLS at the last layer (AUROC 0.801 vs 0.783). Padding is excluded via the attention mask |
| Multi-task (type head) | yes | Same skill (reading resume content); enables heading-free section recognition; one forward pass |
| Full fine-tuning vs frozen | **Full** | Measured: frozen best 0.814 AUROC, fine-tuned 0.908 (§6.2) |
| Head | single linear layer | Minimal head = improvements come from the encoder; less overfitting on 11.7 k units |
| Class imbalance | **validation-tuned thresholds** instead of `pos_weight` | Keeps probabilities calibrated for the feedback confidence and optimises F1 directly |

### Training configuration

| Hyperparameter | Value | Reason |
|---|---|---|
| Optimiser | AdamW, weight decay 0.01 (not on bias / LayerNorm) | Standard for Transformers |
| Learning rate | 3e-5, linear decay, 6 % warm-up | Within BERT's recommended {2e-5, 3e-5, 5e-5}; warm-up stabilises the new heads |
| Epochs | 4 | BERT's recommended 2–4. Validation AUROC rose every epoch (0.851 → 0.891 → 0.904 → 0.911) while validation loss flattened after epoch 2, a sign of growing over-confidence, so we stop at 4 |
| Batch | 8 × 2 gradient accumulation = 16 | Fits a 4 GB RTX 2050 |
| Precision | fp16 autocast + GradScaler, losses in fp32 | Halves memory, ~2× speed |
| Gradient clipping | 1.0 | Stability |
| Length bucketing | similar-length sections per batch | Less padding = faster; batches still shuffled |
| Model selection | best validation macro-AUROC (threshold-free) | Avoids selecting on test |
| Seed | 42 | Reproducibility |

Training time: **12.3 minutes**. Curves: `results/training_curves.png`.

---

## 6. Results (held-out test split, 373 resumes, 2,486 units)

All probabilistic methods use per-label thresholds tuned on **validation**
and applied unchanged to test.

### 6.1 Main comparison (`results/model_comparison.md`)

| Method | Deficiency macro-F1 | macro-AUROC | Section-type acc. |
|---|---:|---:|---:|
| Plain-text rules | 0.497 | – | – |
| TF-IDF + LogReg | 0.382 | 0.748 | 0.927 |
| Frozen DistilBERT CLS + LogReg (Review-1 setup) | 0.481 | 0.783 | 0.924 |
| Frozen DistilBERT Mean + LogReg (Review-1 setup) | 0.512 | 0.801 | 0.922 |
| **Fine-tuned DistilBERT** | **0.665** | **0.908** | **0.955** |
| **Hybrid (fine-tuned + rules)** | **0.721** | – | – |

Per-label test F1:

| Label | Rules | TF-IDF | Frozen Mean | **Fine-tuned** | **Hybrid** |
|---|---:|---:|---:|---:|---:|
| edu_missing_degree | 0.087 | 0.367 | 0.468 | **0.623** | 0.623 |
| edu_missing_institution | 0.152 | 0.407 | 0.412 | **0.605** | 0.607 |
| edu_missing_date | 0.568 | 0.587 | 0.711 | **0.820** | 0.820 |
| exp_missing_dates | 0.235 | 0.093 | 0.265 | **0.608** | 0.608 |
| exp_missing_title | 0.163 | 0.140 | 0.164 | **0.303** | 0.303 |
| exp_thin_description | 0.279 | 0.465 | 0.389 | **0.554** | 0.541 |
| exp_no_quantified_impact | 0.987 | 0.642 | 0.759 | 0.983 | 0.987 |
| summary_too_brief | 1.000 | 0.387 | 0.849 | 0.743 | 1.000 |
| skills_too_few | 1.000 | 0.349 | 0.588 | 0.743 | 1.000 |

**How to read this.**
- Fine-tuning gives **+15.3 F1 points** and **+0.107 AUROC** over the
  Review-1 frozen extractor. This is the core justification for
  fine-tuning.
- The fine-tuned model is the best learned method on **every** label, and
  beats rules by a wide margin on all six **contextual** labels (degree,
  institution, dates, title, thin description).
- On **countable** labels (word count, skill count, presence of a number),
  rules are perfect or near-perfect. That is expected, because those labels
  are simple functions of the text. Using a neural network there adds no
  value, which is exactly why the hybrid exists.

### 6.2 Frozen vs fine-tuned, CLS vs mean (`results/layerwise_probe_vs_finetuned.png`)
A logistic regression was trained on frozen features from **each** of the
7 hidden states:

- **CLS at the embedding layer = AUROC 0.500 (chance).** Before any
  Transformer layer, the [CLS] vector is the same for every input: it is
  the [CLS] token embedding plus position 0. It carries no information
  about the text. Contextualisation is what makes [CLS] informative.
- Frozen features plateau around **0.78–0.81** (best: mean pooling,
  layer 5, 0.814). Pre-trained representations encode useful structure but
  are not organised for our task.
- Fine-tuning reaches **0.908**: the encoder itself had to adapt.

### 6.3 Section-level and segmentation
- Chunk predictions aggregated to whole sections: macro-F1 **0.673**.
- Section-type head accuracy: **0.955** (confusion matrix in
  `results/finetuned_results.json`).
- Rule-based segmentation of plain text (`results/segmenter_evaluation.json`):
  F1 summary 0.989, education 1.000, experience 0.997, skills 0.996,
  certifications 0.923.

### 6.4 End-to-end system (`results/end_to_end_results.json`)
Raw `Resume_str` of the 373 test resumes → segmentation → chunking → model
→ aggregation → hybrid, scored against the HTML truth at resume level:

| Label | P | R | F1 | Decided by |
|---|---:|---:|---:|---|
| edu_missing_degree | 0.537 | 0.717 | 0.614 | model |
| edu_missing_institution | 0.526 | 0.637 | 0.576 | model OR rule |
| edu_missing_date | 0.903 | 0.720 | 0.801 | model |
| exp_missing_dates | 0.350 | 0.656 | 0.457 | model |
| exp_missing_title | 0.211 | 0.367 | 0.268 | model |
| exp_thin_description | 0.438 | 0.629 | 0.517 | model OR rule |
| exp_no_quantified_impact | 0.914 | 0.966 | 0.939 | rule |
| summary_too_brief | 0.778 | 0.933 | 0.849 | rule |
| skills_too_few | 0.915 | 0.860 | 0.887 | rule |
| **macro-F1** | | | **0.656** | |

Section presence F1: summary 0.997, education 1.000, experience 0.997,
skills 0.996.

End-to-end scores are lower than unit-level scores. That is expected:
errors compound across segmentation and chunking, and the plain text has
no clean entry boundaries.

### 6.5 Behavioural tests (`src/rca/behavioural_tests.json`)
Thirteen hand-written sections (32 checks), CheckList-style (Ribeiro
et al., 2020). Model **30/32**. Example: **"Worked on software
development."** is flagged as thin (P = 0.85) and as having no quantified
impact (P = 0.99), while correctly *not* flagged for missing dates
(P = 0.04) or title (P = 0.11), which are present. Rules pass 31/32. The
behavioural suite tests the model's sensitivity to the right content; it
is not where the model's advantage lies (that is §6.1). The 2 failures are a missing job title and a missing institution,
the two weakest labels in §6.1. This is consistent and is reported, not
hidden.

---

## 7. The hybrid policy (`models/distilbert_rca/decision_policy.json`)

For each label, four sources are compared **on validation**: `model`,
`rule`, `model AND rule` (rule as veto) and `model OR rule` (rule as
safety net). The best validation F1 wins; ties go to the model.

Chosen policy: **model** for degree, education date, experience dates and
title; **model OR rule** for institution and thin description; **rule**
for quantified impact, summary length and skills count. Rules also handle
**contact details** (e-mail / phone / profile-URL regex) and
**required / optional section presence**.

This makes the PPT's "semantic understanding + rule-based validation"
concrete: each component is used where the data shows it is better.

---

## 8. How the model is used (`src/12_assess_resume.py`)

```
python src/12_assess_resume.py data/input/sample_resume.txt
```
1. Contact check (rules).
2. Segment into sections (heading lexicon).
3. For each section: chunk → fine-tuned DistilBERT → P(type), P(each
   deficiency) → aggregate → hybrid decision.
4. Required sections missing → "missing" feedback; optional sections
   missing → suggestion.
5. Each flagged label → one feedback message, with its probability and
   the component that decided it.

The same pipeline powers the **web demo** (`run_demo.bat`, `src/14_web_app.py`):
for each section it shows the model's probability next to its decision
threshold and which component decided.

Output on `sample_resume.txt` (abridged):
```
[MISSING] contact     Contact information is missing...
[MISSING] summary     No summary / profile section found...
[WEAK]    experience  Add start and end dates...
[WEAK]    experience  At least one role is described too briefly...
[WEAK]    experience  No measurable results found...
[WEAK]    projects    At least one project is described too briefly...
```

---

## 9. Limitations (state these yourself, before you are asked)

1. **Single-source data / distribution shift.** All resumes come from one
   resume builder, so dates are always standalone fields ("May 2013"). On
   our own `sample_resume.txt`, the model flags **"Graduated in 2025."** as
   a missing date (P = 0.98) and is unsure about **"ABC University"**.
   Prose-style dates and fictional names were never seen in training. Fix:
   format augmentation, plus real student resumes.
2. **Labels are derived, not human-verified yet.** Run the annotation kit
   (§2.5). Until then, our F1 means "agreement with the builder's structure".
3. **`exp_missing_title` is weak** (F1 0.30 unit-level, 0.27 end-to-end):
   it is rare (6 %), and titles that sit in free text produce label noise.
4. **Vagueness is only approximated** by "thin description" and "no
   quantified impact". A true `vague` label needs human labels (the kit
   includes it).
5. **Contact info and Projects are not evaluable** on this anonymised,
   professional-only dataset. Contact uses rules (e-mail, international
   phone formats, profile links). For Projects, the model was never trained
   on project text, so the label definitions (thin description, no
   quantified result) are applied **by rule**, and the model probability is
   shown only as advisory.
6. **Undated roles inside a packed chunk can be missed**: e.g. the
   "Trainee… Worked on software development." role in
   `sample_resume_mixed.txt` is not flagged. This is consistent with
   `exp_missing_dates` being a moderate label (unit F1 0.61).
7. **Input formats.** Plain text and PDF (via the web demo); no DOCX.
8. **Segmentation scores are optimistic** for real-world resumes, because
   headings in this dataset are fairly standard.
9. **Single training run (seed 42).** No variance across seeds is reported.

---

## 10. Likely questions from the panel

**Q: Isn't the model just re-learning your rules?**
No. The labels come from HTML field markup that the model never sees, and
plain-text rules for the same labels score far lower (e.g. 0.09 vs 0.62
on missing degree). Where a label *is* a simple function of the text
(word or skill count), we openly use rules instead.

**Q: Why fine-tune rather than use the frozen embeddings from Review 1?**
Measured on the same test set: frozen best AUROC 0.814, fine-tuned 0.908.
Macro-F1 0.512 → 0.665.

**Q: Why DistilBERT and not BERT or an LLM?**
The task needs contextual reading of short sections (median 78 tokens),
not generation. DistilBERT fine-tunes in 12 minutes on a 4 GB laptop GPU
and returns calibrated per-label probabilities. A generative LLM is
slower, harder to audit, and can state problems the classifier never
detected. Larger encoders (BERT-base, RoBERTa) are a valid ablation for
future work.

**Q: Why mean pooling?**
DistilBERT has no next-sentence-prediction pre-training, so [CLS] was
never trained as a summary vector. Empirically, mean > CLS at the last
frozen layer (0.801 vs 0.783), and frozen CLS at the embedding layer is at
chance (0.500).

**Q: Why BCE and not softmax?**
The labels are not mutually exclusive. A section can have several
deficiencies at once.

**Q: How did you choose thresholds without cheating?**
They were tuned on validation only, then applied unchanged to test. The
same goes for the hybrid policy and the chunking fix.

**Q: How do you handle resumes longer than 512 tokens?**
We work per section; long sections are chunked at entry/role boundaries,
and chunk predictions are combined according to each label's logical
quantifier (max / min / mean).

**Q: How is this different from the surveyed papers?**
They predict job categories, extract entities or rank candidates. We
predict which **information is missing or weak inside each section**, and
generate feedback from that.

---

## 11. Next steps

1. Team annotation → report κ (derived vs human) and model F1 vs human.
2. Format augmentation (prose dates, varied institution names) → fixes
   limitation 1.
3. Train a `vague` label on the human annotations.
4. DOCX ingestion and layout-aware PDF parsing (multi-column resumes).
5. Seeds × 3 and a BERT-base ablation for variance and model-size evidence.

---

## References (new, in addition to the PPT)

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
- Mintz, M., Bills, S., Snow, R., & Jurafsky, D. (2009). Distant supervision for relation extraction without labeled data. ACL.
- Ribeiro, M. T., Wu, T., Guestrin, C., & Singh, S. (2020). Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. ACL.
- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. Biometrics.
- Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization (AdamW). ICLR.
