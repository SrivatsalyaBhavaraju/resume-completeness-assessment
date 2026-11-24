"""
Resume Completeness Assessment (RCA) - shared code used by the
numbered pipeline scripts in src/ (06_ onwards).

Modules
-------
config         : paths, section taxonomy, label definitions, thresholds
html_sections  : parse Resume_html into sections / entries / structured fields
labels         : derive section-level deficiency labels from the HTML structure
text_utils     : plain-text reconstruction + anonymisation-placeholder handling
segmenter      : rule-based section segmentation of PLAIN resume text
rules          : rule-based validation (contact info, plain-text baseline rules)
model          : DistilBERT encoder + section-type head + multi-label head
feedback       : turns predictions into section-wise feedback
"""
