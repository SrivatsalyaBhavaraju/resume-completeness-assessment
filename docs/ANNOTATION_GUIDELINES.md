# Annotation Guidelines: Human Validation of Section Completeness Labels

**Purpose.** Our training labels are derived automatically from the HTML
structure of `Resume.csv` (see `docs/METHODOLOGY.md`, §4). This annotation
checks whether those labels agree with human judgement, and adds one label
(`vague`) that the HTML cannot provide.

**Sheet.** `data/annotation/annotation_sheet.csv`, created with
`python src/13_human_validation.py export`. It holds 160 test-split sections:
60 experience, 40 education, 30 summary and 30 skills.

**Process.**
1. Two team members annotate **independently**. Annotator A fills the
   `human_a_*` columns and annotator B fills `human_b_*`. Do not discuss
   individual items until both have finished.
2. Enter `1` if the problem is present and `0` if it is not. Leave the cell
   blank only if it is genuinely impossible to judge.
3. Judge **only the text shown**. The section heading has been removed on purpose.
4. Run `python src/13_human_validation.py score`.

---

## Education sections

| Column | Mark 1 if... | Mark 0 if... |
|---|---|---|
| `edu_missing_degree` | Any entry does not say what qualification was earned or pursued (no "Bachelor of...", "MBA", "High School Diploma", "Certificate in...", etc.). | Every entry states a qualification. |
| `edu_missing_institution` | Any entry does not name the school, college or university. | Every entry names its institution. |
| `edu_missing_date` | Any entry has no year or date. | Every entry has a year or date ("Expected 2026" counts). |

## Experience sections

| Column | Mark 1 if... | Mark 0 if... |
|---|---|---|
| `exp_missing_dates` | Any role has no start/end date or year. | Every role is dated. |
| `exp_missing_title` | Any role does not state a job title. | Every role has a title. |
| `exp_thin_description` | Any role has only a token description (roughly under 15 words of responsibilities or contributions, excluding title, company, place and dates). | Every role describes the work in some detail. |
| `exp_no_quantified_impact` | No role mentions any measurable quantity (numbers, %, money, team size, volumes). Dates do not count. | At least one quantity is mentioned. |

## Summary sections

| Column | Mark 1 if... |
|---|---|
| `summary_too_brief` | The summary is a single short sentence or fragment (under about 20 words). |

## Skills sections

| Column | Mark 1 if... |
|---|---|
| `skills_too_few` | Fewer than 5 distinct, concrete skills are listed. |

## `vague` (experience and summary only)

Mark **1** if a recruiter reading the text would *not* learn **what
specifically** the person did or can do. Typical signs:
- generic phrases with no object ("worked on various tasks", "responsible
  for development");
- no tools, domains, deliverables or outcomes;
- clichés only ("hard-working team player").

Mark **0** if the text names concrete activities, tools, domains or results,
even if it is short.

| Example | vague |
|---|:---:|
| "Worked on software development." | 1 |
| "Responsible for various duties as assigned." | 1 |
| "Built a Flask REST API for the billing team; added PyTest unit tests." | 0 |
| "Dedicated professional with excellent communication skills." | 1 |
| "Registered nurse with 6 years in ICU care, ACLS-certified, experienced with Epic EHR." | 0 |

`vague` is deliberately **not** a training label: it is the judgement we
could not derive automatically. The score script tests whether the model's
`exp_thin_description` / `summary_too_brief` probability tracks it (AUROC).
A high AUROC supports using the model as a vagueness detector. A low AUROC
tells us that vagueness needs its own human-labelled training data, which
is the recommended next step.

## Interpreting Cohen's κ

| κ | Agreement |
|---|---|
| < 0.40 | weak (the label definition needs revising) |
| 0.40 – 0.60 | moderate |
| 0.60 – 0.80 | substantial |
| > 0.80 | almost perfect |

(Landis & Koch, 1977)
