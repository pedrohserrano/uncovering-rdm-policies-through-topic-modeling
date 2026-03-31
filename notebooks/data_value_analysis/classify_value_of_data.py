#!/usr/bin/env python3
"""
Zero-shot classification of policy chunks for 'value of research data'
using Gemini on Vertex AI (EU region).

Adds three columns to the CSV:
- gemini_class: {"yes","no","unsure"}
- gemini_explanation: brief rationale with minimal evidence
- value_class: semicolon-delimited categories capturing what kind of "value" (can be multiple)

Usage:
  python classify_value_of_data.py \
      --csv ../../data/output/labelling_full_human.csv \
      --project YOUR_GCP_PROJECT \
      --location europe-west4 \
      --model gemini-1.5-pro-002 \
      --dry-run 20
"""

import argparse
import json
import os
import time
from typing import Dict, Any, Tuple, List

import pandas as pd
from dotenv import load_dotenv

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

# ---------- Defaults (env vars override) ----------
DEFAULT_PROJECT = os.getenv("VERTEX_PROJECT", "etui-cloud-architecture")
DEFAULT_LOCATION = os.getenv("VERTEX_LOCATION", "europe-west4")
DEFAULT_MODEL = os.getenv("VERTEX_MODEL", "gemini-1.5-pro-002")

# Guardrail for very long rows
MAX_CHARS = 12000

# Generation config tuned for stability and JSON output
GEN_CONFIG = GenerationConfig(
    temperature=0.1,
    top_p=0.2,
    top_k=32,
    max_output_tokens=640,
    response_mime_type="application/json",
)

PROMPT_TEMPLATE = """You are classifying whether a policy text *talks about the value of research data*.

Return ONLY JSON with keys:
- "label"          : one of yes|no|unsure
- "explanation"    : <= 3 sentences, objective, no speculation
- "evidence"       : array with up to 2 short verbatim spans from the text OR [] if signal is implicit
- "value_categories": array of zero or more categories from the ALLOWED list below
- "value_other"    : optional short free-text label if a relevant category is missing; else empty string

## Task
Given the TEXT, decide if it expresses that *research data has value*. Value can be explicit or implicit.

### Consider "value" broadly, including:
- economic/financial/commercial value (e.g., asset, IP, competitive advantage, monetization)
- intrinsic/scientific value (e.g., enables knowledge creation, reuse, verification, reproducibility)
- societal/public value (e.g., public good, societal impact, innovation, policy-making)
- strategic/organizational value (e.g., strategic asset, institutional advantage, efficiency)
- utility-based/pragmatic value (e.g., practical usefulness, accelerates discovery, improves quality)

### Positive cues (any one is sufficient):
- Direct statements that data *has value*, *is valuable*, or *creates value/impact/benefit*.
- Data framed as an asset/resource with benefits/returns.
- Reuse/sharing/FAIR explicitly linked to value/benefits/impact.
- RDM or curation described as creating *competitive edge*, *innovation*, or *societal impact* via data.

### Implicit cues (count as YES if clearly implied):
- Data described as foundational for research progress, verification, collaboration, **with benefits attributed to the data itself**.

### Negative cues (lean NO if dominant):
- Purely procedural/compliance text (roles, approvals, storage, retention) with no benefit/impact claim.
- Importance of processes without linking to value of the data itself.

### UNSURE:
- Vague language where it’s unclear whether *data itself* is valued.
- Conflicting or insufficient evidence.

### "value_categories" (ALLOWED)
Pick any that apply (can be multiple, or none if label != "yes"):
- "economic_financial_commercial"
- "societal_public"
- "strategic_organizational"
- "intrinsic_scientific"
- "utility_pragmatic"
Optionally propose one extra concise category via "value_other" if needed.

### Output JSON (ONLY):
{{
  "label": "yes|no|unsure",
  "explanation": "max 3 sentences, objective.",
  "evidence": ["short quote 1", "short quote 2"],
  "value_categories": ["intrinsic_scientific","societal_public"],
  "value_other": ""
}}

TEXT:
\"\"\"{text}\"
\"\"\"
"""

def init_vertex(project: str, location: str, model_name: str) -> GenerativeModel:
    vertexai.init(project=project, location=location)
    return GenerativeModel(model_name)

def build_prompt(text: str) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) > MAX_CHARS:
        cleaned = cleaned[:MAX_CHARS] + " …"
    return PROMPT_TEMPLATE.format(text=cleaned)

def call_gemini(model: GenerativeModel, prompt: str, retries: int = 5, base_delay: float = 1.0) -> Dict[str, Any]:
    last_err = None
    for attempt in range(retries):
        try:
            resp = model.generate_content(prompt, generation_config=GEN_CONFIG)
            raw = resp.text  # expecting JSON
            data = json.loads(raw)
            return data
        except Exception as e:
            last_err = e
            time.sleep(base_delay * (2 ** attempt) + 0.1 * attempt)
    raise RuntimeError(f"Gemini call failed after {retries} attempts: {last_err}")

ALLOWED_CATS = {
    "economic_financial_commercial",
    "societal_public",
    "strategic_organizational",
    "intrinsic_scientific",
    "utility_pragmatic",
}

def normalize_output(data: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Returns (label, explanation, value_class)
    - label in {"yes","no","unsure"}
    - explanation: compact text, optionally with minimal evidence inline
    - value_class: semicolon-delimited categories (allowed + optional 'other:<label>')
    """
    label = str(data.get("label", "")).strip().lower()
    if label not in {"yes", "no", "unsure"}:
        label = "unsure"

    explanation = (data.get("explanation") or "").strip()
    ev = data.get("evidence") or []
    if isinstance(ev, list) and ev:
        ev_join = " | ".join([str(x)[:200] for x in ev[:2]])
        if ev_join and ev_join not in explanation:
            explanation = (explanation + f" Evidence: {ev_join}").strip()
    explanation = " ".join(explanation.split())

    raw_cats = data.get("value_categories") or []
    cats: List[str] = []
    for c in raw_cats:
        c_norm = str(c).strip().lower()
        # normalize to our canonical keys
        if c_norm in ALLOWED_CATS:
            cats.append(c_norm)
    other = (data.get("value_other") or "").strip()
    if other:
        cats.append(f"other:{other}")

    # For NO/UNSURE, we typically expect no categories; but if the model returned any,
    # keep them for auditability.
    value_class = ";".join(cats)
    return label, explanation, value_class

def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="../../data/output/labelling_full_human.csv", help="Path to input CSV with a 'text' column.")
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--location", default=DEFAULT_LOCATION)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name.")
    parser.add_argument("--dry-run", type=int, default=0, help="If >0, only process the first N rows.")
    parser.add_argument("--output", default=None, help="Output CSV path; default appends _gemini before .csv")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    # Init Vertex + model
    model = init_vertex(args.project, args.location, args.model)

    df = pd.read_csv(args.csv)
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    # Prepare output columns
    if "gemini_class" not in df.columns:
        df["gemini_class"] = pd.NA
    if "gemini_explanation" not in df.columns:
        df["gemini_explanation"] = pd.NA
    if "value_class" not in df.columns:
        df["value_class"] = pd.NA

    n_rows = len(df) if args.dry_run <= 0 else min(args.dry_run, len(df))
    print(f"Processing {n_rows} row(s)...")

    processed = 0
    for idx in range(n_rows):
        row = df.iloc[idx]
        # Allow safe resume: skip labeled rows
        if pd.notna(row.get("gemini_class")) and str(row.get("gemini_class")).strip():
            continue
        text = row.get("text", "")

        try:
            payload = call_gemini(model, build_prompt(text))
            label, explanation, value_class = normalize_output(payload)
        except Exception as e:
            label, explanation, value_class = "unsure", f"Automatic classification failed: {e}", ""

        df.at[df.index[idx], "gemini_class"] = label
        df.at[df.index[idx], "gemini_explanation"] = explanation
        df.at[df.index[idx], "value_class"] = value_class
        processed += 1

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{n_rows}")

    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.csv)
        out_path = f"{base}_gemini{ext or '.csv'}"

    df.to_csv(out_path, index=False)
    print(f"Newly processed rows: {processed}")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
