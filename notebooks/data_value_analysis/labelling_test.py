#!/usr/bin/env python3
"""
Create a labelled dataset from structured chunks.

- Loads a chunked CSV (default: ../../data/output/documents_chunked_structured_human.csv)
- Processes all rows by default (optionally sample via --n)
- Adds two new columns:
  - answer: one of {yes, no, unsure}
  - explanation: short rationale for the label

Usage:
  python labelling_test.py \
    --input ../../data/output/documents_chunked_structured_human.csv \
    --output ../../data/output/labelling_full_human.csv

Note: This uses lightweight heuristics to approximate zero-shot classification
without downloading large models. Adjust patterns below as needed.
"""

import argparse
import csv
import os
import re
from typing import List, Tuple

QUESTION = "is there value in relation to data in this text?"


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def near_each_other(text: str, a: str, b: str, window: int = 5) -> bool:
    """Return True if tokens a and b appear within +/- window words.
    Order-agnostic.
    """
    toks = tokenize_words(text)
    apos = [i for i, t in enumerate(toks) if t == a]
    bpos = [i for i, t in enumerate(toks) if t == b]
    for i in apos:
        for j in bpos:
            if abs(i - j) <= window:
                return True
    return False


def contains_any(text: str, terms: List[str]) -> bool:
    low = text.lower()
    return any(t in low for t in terms)


def sentence_split(text: str) -> List[str]:
    # Simple sentence splitter to keep stdlib-only
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def label_value_of_data(text: str) -> Tuple[str, str]:
    """Heuristic zero-shot-like classifier.

    Returns (answer, explanation) where answer in {"yes", "no", "unsure"}.
    """
    if not text or not re.search(r"[A-Za-z]", text):
        return "unsure", "Unsure: text is empty or non-alphabetic."

    low = text.lower()

    # Synonyms/anchors
    data_terms = [
        "data", "dataset", "datasets", "research data", "metadata", "data asset", "data assets",
        "data products", "data outputs", "data set",
    ]
    value_terms = [
        "value", "valuable", "worth", "benefit", "benefits", "gain", "gains",
        "impact", "impacts", "utility", "usefulness", "advantage", "advantages",
        "return", "returns", "roi", "outcome", "outcomes", "added value", "value-added",
        "value creation", "value generation", "value proposition", "value extraction",
        "monetization", "monetisation", "economic value", "societal value", "public value",
        "business value", "strategic value", "maximize value", "maximise value",
        "increase value", "enhance value", "derive value", "unlock value",
    ]

    explicit_phrases = [
        # direct
        "value of data", "value from data", "value in data", "data value",
        "data is valuable", "data are valuable", "valuable data",
        # creation/extraction
        "value creation", "value generation", "value extraction", "unlock value",
        # maximization/enhancement
        "maximize the value", "maximise the value", "increase the value", "enhance the value",
        # domains
        "economic value", "societal value", "public value", "business value", "strategic value",
        # policy phrasing
        "realise the value of data", "realize the value of data",
        "deriving value from data", "derive value from data",
        "leveraging data for value", "data-driven value",
    ]

    # Any explicit phrase
    if contains_any(low, explicit_phrases):
        return "yes", "Explicit: direct phrase linking value and data."

    # Proximity: any value term near any data term
    for dt in data_terms:
        for vt in value_terms:
            if near_each_other(text, dt, vt, window=5):
                return "yes", f"Explicit: '{vt}' appears near '{dt}'."

    # Implicit cues: benefit/impact/utility/innovation from data sharing/FAIR/reuse/commercialization
    implicit_terms = [
        # impacts and benefits
        "benefit", "benefits", "impact", "impacts", "usefulness", "utility",
        "innovation", "innovations", "knowledge transfer", "translation",
        "societal", "economic", "commercial", "commercialisation", "commercialization",
        "valorisation", "valorization", "exploitation", "leverage", "leveraging",
        # FAIR + reuse
        "fair", "reusable", "reusability", "reuse", "re-use", "secondary use",
        # outcomes
        "uptake", "adoption", "added value", "value-added", "value chain",
        "maximize", "maximise", "enhance", "derive",
    ]

    # Check at sentence level with an anchor to data terms
    for sent in sentence_split(text):
        sent_low = sent.lower()
        if any(dt in sent_low for dt in data_terms) and any(t in sent_low for t in implicit_terms):
            return "yes", "Implicit: discusses benefits/impact/FAIR/commercial outcomes tied to data."

    # Short/uninformative texts → unsure
    if len(tokenize_words(text)) < 20:
        return "unsure", "Unsure: very short text; no clear value cues."

    # Default: no
    return "no", "No: no explicit or implicit value-of-data cues detected."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../../data/output/documents_chunked_structured_human.csv")
    ap.add_argument("--output", default="../../data/output/labelling_full_human.csv")
    ap.add_argument("--n", type=int, default=0, help="Sample size; 0 means all records")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    # Read input
    rows: List[dict] = []
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise RuntimeError("No rows read from input CSV.")

    # Select rows
    sample = rows if args.n in (0, None) else rows[: args.n]

    # Prepare output
    # Keep original columns and add answers (do not include the question column in full run)
    out_fieldnames = list(sample[0].keys()) + ["answer", "explanation"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for r in sample:
            text = r.get("text", "")
            answer, explanation = label_value_of_data(text)
            r_out = dict(r)
            r_out["answer"] = answer
            r_out["explanation"] = explanation
            writer.writerow(r_out)

    print(f"Wrote {len(sample)} rows to {args.output}")


if __name__ == "__main__":
    main()
