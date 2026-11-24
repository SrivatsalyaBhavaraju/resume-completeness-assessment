# Intelligent Resume Completeness Assessment

**Section-wise resume completeness assessment using a fine-tuned DistilBERT model plus rule-based validation.**

The system reads a plain-text resume and works out:
1. which sections are present, and which required sections are missing;
2. which present sections are **incomplete or weak** (e.g. an education
   entry with no degree, a job with no dates, a role described in five words);
3. what the candidate should change, as section-wise feedback.

The full justification for every design decision is in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

## Live demo (web app)

Double-click **`run_demo.bat`** (or run `python src/14_web_app.py`). The browser
opens at http://127.0.0.1:8000.

- **Assess a resume**: pick the Weak / Mixed / Strong sample, paste text, or
  drop a `.txt` / `.pdf`. Shows the checks-passed ring, the section checklist,
  prioritised feedback, every model probability against its decision threshold,
  which component decided each check, and the resume colour-coded by detected section.
- **How it works**: the pipeline, where the labels come from, the architecture,
  and the nine checks.
- **Evaluation**: test-set results, comparison with baselines, charts and limitations.

Deep links for presenting: `/?sample=weak`, `/?sample=mixed`, `/?sample=strong`, `/?tab=how`, `/?tab=eval`.
Runs fully offline, using only the Python standard library plus the installed packages.

## Setup

```bash
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu130   # GPU build
.venv\Scripts\python -m pip install -r requirements.txt
```

**Files not stored in git:**

| What | Where to get it | Put it at |
|---|---|---|
| Fine-tuned model (265 MB) | `distilbert_rca.zip` on this repo's **Releases** page | unzip to `models/distilbert_rca/` |
| Dataset `Resume.csv` | Kaggle "Resume Dataset" (Snehaan Bhawal): https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset | `data/input/Resume.csv` |
| `data/processed/training_units.jsonl` | run `python src/06_build_section_dataset.py` (needs the dataset) | (generated) |

The **web demo only needs the model** (the heading lexicon and the split are in
the repo). To retrain instead of downloading the model, get the dataset and run
scripts 06 → 09 (about 15 minutes on a 4 GB GPU).

> After the first run has downloaded DistilBERT, set `HF_HUB_OFFLINE=1`
> (PowerShell: `$env:HF_HUB_OFFLINE=1`) so scripts don't fail when the
> internet connection is unstable.
>
> The old `venv/` folder was created on another computer
> (`C:\Users\ipsai\...`) and cannot run on this one. Use `.venv/`.

## Pipeline (run from the project root, in order)

| Script | Stage | Output |
|---|---|---|
| `src/01`–`05` | **Review 1**: pre-trained DistilBERT inspection, tokenisation, frozen inference, layer-wise analysis | `results/CIA1_*`, `inference_summary.json`, … |
| `src/06_build_section_dataset.py` | Parse HTML → sections → derived labels → entry-aligned chunks; resume-level 70/15/15 split; heading lexicon | `data/processed/`, `results/label_statistics.json` |
| `src/07_evaluate_segmenter.py` | Rule-based section detection on plain text (test split) | `results/segmenter_evaluation.json` |
| `src/08_baselines.py` | Rules, TF-IDF+LogReg, **frozen DistilBERT probes (all 7 layers, CLS vs mean)** | `results/baseline_results.json` |
| `src/09_finetune_distilbert.py` | **Fine-tuning** (multi-task: section type + multi-label deficiencies) | `models/distilbert_rca/`, `results/training_*` |
| `src/10_evaluate_finetuned.py` | Test metrics, hybrid policy (chosen on validation), comparison, behavioural tests | `results/model_comparison.md`, `finetuned_results.json`, plots |
| `src/11_end_to_end_evaluation.py` | Whole system on raw plain text of test resumes | `results/end_to_end_results.json` |
| `src/12_assess_resume.py [file.txt]` | **Use the model**: assess one resume and print feedback | `results/assessment_<name>.json` |
| `src/13_human_validation.py export / score` | Human validation kit (Cohen's κ) | `data/annotation/`, `results/human_validation.json` |
| `src/14_web_app.py` | **Web demo** (front-end in `web/index.html`) | http://127.0.0.1:8000 |

Shared code lives in `src/rca/`.

## Project status

| Component | Status |
|---|---|
| Pre-trained DistilBERT feature extraction (Review 1) | Done |
| Section-level labels | Done: derived from the dataset's HTML field markup (distant supervision) |
| Required / optional section definitions | Done (`src/rca/config.py`) |
| Multi-label classification head | Done (9 deficiency labels, masked BCE) + section-type head |
| Task-specific fine-tuning | Done (full fine-tuning, 4 epochs, RTX 2050, about 12 min) |
| Long-resume handling | Done: entry-aligned chunking (training), role-aligned chunking + quantifier-based aggregation (inference) |
| Rule-based validation | Done: contact details, presence, hybrid decision policy chosen on validation |
| Feedback generation | Done (template-based, traceable to labels) |
| Baselines / comparison | Done |
| Human validation of labels | **Kit ready; annotation by the team pending** |
| `vague` (semantic vagueness) label | **Not trained**: needs human labels (see guidelines) |
| Web demo with PDF / TXT upload | Done (`run_demo.bat`); DOCX not supported |
| Contact-info and Projects assessment on real data | Rule-based / transferred; **not evaluable on this dataset** (anonymised, no projects) |
