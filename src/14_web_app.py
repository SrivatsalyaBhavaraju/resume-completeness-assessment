
import os
import io
import sys
import json
import time
import base64
import threading
import webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # model is cached locally

import torch

from rca import config
from rca.inference import ResumeAssessor

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------
# Local web demo of the complete system. Standard library only (no web
# framework), so it runs offline with the packages already installed.
#
#   python src/14_web_app.py            -> http://127.0.0.1:8000

HOST, PORT = "127.0.0.1", int(os.environ.get("RCA_PORT", 8000))
WEB_DIR = os.path.join(config.PROJECT_ROOT, "web")
INPUT_DIR = os.path.join(config.PROJECT_ROOT, "data", "input")

SAMPLES = {
    "weak": ("Weak resume", "sample_resume.txt"),
    "mixed": ("Mixed resume", "sample_resume_mixed.txt"),
    "strong": ("Strong resume", "sample_resume_strong.txt"),
}

LABEL_TITLES = {
    "edu_missing_degree": "Degree / qualification stated",
    "edu_missing_institution": "Institution named",
    "edu_missing_date": "Graduation date given",
    "exp_missing_dates": "Employment dates given",
    "exp_missing_title": "Job title stated",
    "exp_thin_description": "Work described in enough detail",
    "exp_no_quantified_impact": "Quantified results / impact",
    "summary_too_brief": "Summary long enough",
    "skills_too_few": "Enough specific skills",
}

print("=" * 65)
print("RESUME COMPLETENESS ASSESSMENT - WEB DEMO")
print("=" * 65)

# --------------------------------------------------
# 2. LOAD MODEL ONCE
# --------------------------------------------------

print("\nLoading fine-tuned DistilBERT...")
assessor = ResumeAssessor()
model_lock = threading.Lock()
device_name = torch.cuda.get_device_name(0) if assessor.device == "cuda" else "CPU"
print("Model loaded on:", device_name)

# Warm-up so the first request in the demo is fast
with model_lock:
    assessor.assess("EXPERIENCE\nAnalyst\nJan 2020 to Present\nDid analysis.")


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def model_info():
    with open(os.path.join(config.FINETUNED_DIR, "rca_config.json"), "r", encoding="utf-8") as file:
        meta = json.load(file)
    baselines = read_json(os.path.join(config.RESULTS_DIR, "baseline_results.json")) or {}
    finetuned = read_json(os.path.join(config.RESULTS_DIR, "finetuned_results.json")) or {}

    comparison = []
    if baselines and finetuned:
        rows = [
            ("Plain-text rules", baselines["rules"], "baseline"),
            ("TF-IDF + Logistic Regression", baselines["tfidf_logreg"], "baseline"),
            ("Frozen DistilBERT CLS + LogReg (Review 1)", baselines["frozen_probe"]["cls"]["Layer 6"], "baseline"),
            ("Frozen DistilBERT Mean + LogReg (Review 1)", baselines["frozen_probe"]["mean"]["Layer 6"], "baseline"),
            ("Fine-tuned DistilBERT (ours)", finetuned["finetuned_chunk_level"], "ours"),
            ("Hybrid: fine-tuned + rules (ours)", finetuned["hybrid"], "ours"),
        ]
        for name, result, kind in rows:
            comparison.append({
                "method": name,
                "kind": kind,
                "macro_f1": result["labels"]["macro_f1"],
                "macro_auroc": result["labels"].get("macro_auroc"),
                "type_accuracy": result.get("section_type", {}).get("accuracy"),
                "per_label": {k: v["f1"] for k, v in result["labels"]["per_label"].items()},
            })

    return {
        "model": config.MODEL_NAME,
        "device": device_name,
        "parameters": sum(p.numel() for p in assessor.model.parameters()),
        "best_epoch": meta.get("best_epoch"),
        "hparams": meta.get("hparams"),
        "labels": [{"name": n, "title": LABEL_TITLES[n], "section": s["section"], "definition": s["definition"]}
                   for n, s in config.LABELS.items()],
        "policy": {"source": assessor.source, "thresholds": assessor.thresholds},
        "comparison": comparison,
        "training": read_json(os.path.join(config.RESULTS_DIR, "training_history.json")),
        "end_to_end": read_json(os.path.join(config.RESULTS_DIR, "end_to_end_results.json")),
        "segmenter": read_json(os.path.join(config.RESULTS_DIR, "segmenter_evaluation.json")),
        "label_stats": read_json(os.path.join(config.RESULTS_DIR, "label_statistics.json")),
        "behavioural": (finetuned.get("behavioural_tests") or {}),
        "samples": {key: title for key, (title, _) in SAMPLES.items()},
    }


INFO = model_info()

# --------------------------------------------------
# 3. ASSESSMENT
# --------------------------------------------------


def run_assessment(text):
    started = time.perf_counter()
    with model_lock:
        report = assessor.assess(text)
    elapsed_ms = (time.perf_counter() - started) * 1000

    # "Checks passed": every required section present + every applicable
    # completeness check on present sections not flagged. Shown as a
    # summary of the report, NOT as a model accuracy.
    checks = passed = 0
    contact_ok = report["contact"]["email"] or report["contact"]["phone"]
    checks, passed = checks + 1, passed + int(contact_ok)
    for section_type in config.REQUIRED_SECTIONS:
        if section_type == "contact":
            continue
        checks += 1
        passed += int(section_type in report["present"])
    for section in report["sections"]:
        for name, decision in section["labels"].items():
            decision["title"] = LABEL_TITLES[name]
            decision["threshold"] = assessor.thresholds[name]
            checks += 1
            passed += int(not decision["flagged"])

    report["score"] = {"passed": passed, "checks": checks, "percent": round(100 * passed / max(1, checks))}
    report["timing_ms"] = round(elapsed_ms)
    report["device"] = device_name
    report["text"] = text
    return report


def extract_text(filename, payload):
    raw = base64.b64decode(payload)
    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return raw.decode("utf-8", errors="replace")


# --------------------------------------------------
# 4. HTTP SERVER
# --------------------------------------------------

STATIC_TYPES = {".html": "text/html; charset=utf-8", ".png": "image/png",
                ".css": "text/css", ".js": "application/javascript"}


class Handler(BaseHTTPRequestHandler):

    def send(self, status, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def serve_file(self, path):
        extension = os.path.splitext(path)[1].lower()
        if not os.path.isfile(path) or extension not in STATIC_TYPES:
            return self.send(404, {"error": "not found"})
        with open(path, "rb") as file:
            self.send(200, file.read(), STATIC_TYPES[extension])

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            return self.serve_file(os.path.join(WEB_DIR, "index.html"))
        if url.path == "/api/info":
            return self.send(200, INFO)
        if url.path == "/api/sample":
            key = parse_qs(url.query).get("name", ["weak"])[0]
            if key not in SAMPLES:
                return self.send(404, {"error": "unknown sample"})
            with open(os.path.join(INPUT_DIR, SAMPLES[key][1]), "r", encoding="utf-8") as file:
                return self.send(200, {"name": key, "text": file.read()})
        if url.path.startswith("/results/"):
            name = os.path.basename(url.path)   # no directory traversal
            return self.serve_file(os.path.join(config.RESULTS_DIR, name))
        return self.send(404, {"error": "not found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 15 * 1024 * 1024:
                return self.send(413, {"error": "file too large"})
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/assess":
                text = (body.get("text") or "").strip()
                if len(text.split()) < 5:
                    return self.send(400, {"error": "Please paste or upload a resume first."})
                return self.send(200, run_assessment(text))
            if self.path == "/api/extract":
                text = extract_text(body.get("filename", ""), body.get("data", ""))
                return self.send(200, {"text": text})
            return self.send(404, {"error": "not found"})
        except Exception as error:  # report errors to the page instead of crashing
            return self.send(500, {"error": f"{type(error).__name__}: {error}"})

    def log_message(self, fmt, *args):
        if "/api/assess" in (args[0] if args else ""):
            sys.stdout.write("  %s\n" % (fmt % args))


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    print(f"\nDemo running at {url}   (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
