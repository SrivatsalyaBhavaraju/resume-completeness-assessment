"""
Plain-text helpers.

Anonymisation artifact
----------------------
The dataset was anonymised by replacing every employer with the literal
string "Company Name" and every location with "City , State". If we train
on that text, the model can learn "the words 'company name' => this is an
experience entry with an organisation", a cue that never occurs in a real
resume. To remove this shortcut we replace the placeholders with varied,
realistic values (seeded, so results are reproducible). The same function
is applied to dataset text at evaluation time; real resumes need nothing.
"""

import random
import re

COMPANIES = [
    "Infosys", "Accenture", "Deloitte", "Wipro", "Siemens", "Walmart",
    "Bosch", "Oracle", "Target Corporation", "Marriott International",
    "Kaiser Permanente", "Hilton Worldwide", "Tata Consultancy Services",
    "Capgemini", "General Electric", "Wells Fargo", "Johnson & Johnson",
    "Amazon", "Cognizant", "Honeywell", "KPMG", "Pfizer", "Boeing",
    "HDFC Bank", "Mahindra Group", "Lockheed Martin", "Starbucks",
    "Procter & Gamble", "Ernst & Young", "Cisco Systems", "Northrop Grumman",
    "Bright Horizons", "Mayo Clinic", "Ford Motor Company", "Hyatt Hotels",
]

LOCATIONS = [
    ("Austin", "TX"), ("Chicago", "IL"), ("Seattle", "WA"), ("Denver", "CO"),
    ("Atlanta", "GA"), ("Boston", "MA"), ("Phoenix", "AZ"), ("Columbus", "OH"),
    ("San Diego", "CA"), ("Orlando", "FL"), ("Nashville", "TN"), ("Portland", "OR"),
    ("Bengaluru", "Karnataka"), ("Pune", "Maharashtra"), ("Hyderabad", "Telangana"),
    ("Chennai", "Tamil Nadu"), ("Raleigh", "NC"), ("Kansas City", "MO"),
    ("Minneapolis", "MN"), ("Pittsburgh", "PA"), ("Richmond", "VA"),
]

COMPANY_PATTERN = re.compile(r"\bCompany Name\b")
LOCATION_PATTERN = re.compile(r"\bCity\s*,\s*State\b")
CITY_PATTERN = re.compile(r"\bCity\b(?=\s*[,\n－-])")
STATE_PATTERN = re.compile(r"(?<=[,－-])\s*\bState\b")


def replace_placeholders(text, rng=None):
    """Replace anonymisation placeholders with realistic values."""
    rng = rng or random.Random(0)

    def company(_):
        return rng.choice(COMPANIES)

    def location(_):
        city, state = rng.choice(LOCATIONS)
        return f"{city}, {state}"

    text = COMPANY_PATTERN.sub(company, text)
    text = LOCATION_PATTERN.sub(location, text)
    text = CITY_PATTERN.sub(lambda _: rng.choice(LOCATIONS)[0], text)
    text = STATE_PATTERN.sub(lambda _: " " + rng.choice(LOCATIONS)[1], text)
    return text


def normalise_whitespace(text):
    """Collapse spaces inside lines, keep line breaks."""
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)
