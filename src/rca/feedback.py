"""
Feedback generation: turn section-level decisions into actionable,
section-wise suggestions.

Template-based on purpose: every message is traceable to one predicted
label (and its probability), so the feedback can be explained and audited.
A generative model could phrase things more freely but could also state
problems the classifier never detected.
"""

MESSAGES = {
    "edu_missing_degree": "State the degree or qualification for every education entry "
                          "(e.g. 'B.Tech in Computer Science').",
    "edu_missing_institution": "Name the institution (university / college / school) for every education entry.",
    "edu_missing_date": "Add the graduation year (or expected year) for each education entry.",
    "exp_missing_dates": "Add start and end dates (month/year) for every role so the timeline is clear.",
    "exp_missing_title": "State your job title for every role.",
    "exp_thin_description": "At least one role is described too briefly. Add 2-4 bullet points describing "
                            "your responsibilities, the tools you used and what you delivered.",
    "exp_no_quantified_impact": "No measurable results found. Quantify your impact where possible "
                                "(e.g. 'reduced processing time by 30%', 'managed a team of 6', "
                                "'served 200+ customers daily').",
    "summary_too_brief": "Your summary is very short. Expand it to 2-4 sentences covering your role, "
                         "years of experience, core skills and what you are looking for.",
    "skills_too_few": "List more specific skills (aim for at least 5-10), including tools, technologies "
                      "and domain skills, not only soft skills.",
}

PROJECT_MESSAGES = {
    "exp_thin_description": "At least one project is described too briefly. Explain what you built, "
                            "your role, the technologies used and the outcome.",
    "exp_no_quantified_impact": "Quantify project results where possible (e.g. dataset size, accuracy "
                                "achieved, number of users, speed-up).",
}

MISSING_REQUIRED = {
    "contact": "Contact information is missing. Add at least an e-mail address and a phone number.",
    "summary": "No summary / profile section found. Add a 2-4 sentence professional summary at the top.",
    "education": "No education section found. Add your degree, institution and graduation year.",
    "experience": "No experience section found. Add your work experience or internships with dates and "
                  "responsibilities.",
    "skills": "No skills section found. Add a list of your technical and domain skills.",
}

MISSING_OPTIONAL = {
    "projects": "Optional: consider adding a Projects section describing what you built, the technologies "
                "used and the outcome (especially valuable for students and early-career candidates).",
    "certifications": "Optional: add relevant certifications (name, issuing body, year) if you have any.",
}

CONTACT_ITEMS = {
    "email": "Add an e-mail address.",
    "phone": "Add a phone number.",
    "profile_link": "Optional: add a LinkedIn / GitHub / portfolio link.",
}

NO_HEADING = ("This content looks like a {type} section but has no clear heading. "
              "Add a '{title}' heading so readers and parsers can find it.")
