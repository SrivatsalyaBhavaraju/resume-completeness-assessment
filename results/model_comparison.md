| Method | Deficiency macro-F1 | Deficiency macro-AUROC | Section-type accuracy |
|---|---:|---:|---:|
| Plain-text rules | 0.497 | n/a | n/a |
| TF-IDF + LogReg | 0.382 | 0.748 | 0.927 |
| Frozen DistilBERT CLS + LogReg | 0.481 | 0.783 | 0.924 |
| Frozen DistilBERT Mean + LogReg | 0.512 | 0.801 | 0.922 |
| Fine-tuned DistilBERT | 0.665 | 0.908 | 0.955 |
| Hybrid (fine-tuned + rules) | 0.721 | n/a | n/a |

Per-label test F1:

| Label | Plain-text rules | TF-IDF + LogReg | Frozen DistilBERT CLS + LogReg | Frozen DistilBERT Mean + LogReg | Fine-tuned DistilBERT | Hybrid (fine-tuned + rules) |
|---|---:|---:|---:|---:|---:|---:|
| edu_missing_degree | 0.087 | 0.367 | 0.438 | 0.468 | 0.623 | 0.623 |
| edu_missing_institution | 0.152 | 0.407 | 0.429 | 0.412 | 0.605 | 0.607 |
| edu_missing_date | 0.568 | 0.587 | 0.717 | 0.711 | 0.820 | 0.820 |
| exp_missing_dates | 0.235 | 0.093 | 0.262 | 0.265 | 0.608 | 0.608 |
| exp_missing_title | 0.163 | 0.140 | 0.182 | 0.164 | 0.303 | 0.303 |
| exp_thin_description | 0.279 | 0.465 | 0.323 | 0.389 | 0.554 | 0.541 |
| exp_no_quantified_impact | 0.987 | 0.642 | 0.694 | 0.759 | 0.983 | 0.987 |
| summary_too_brief | 1.000 | 0.387 | 0.706 | 0.849 | 0.743 | 1.000 |
| skills_too_few | 1.000 | 0.349 | 0.579 | 0.588 | 0.743 | 1.000 |
