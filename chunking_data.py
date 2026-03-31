#!/usr/bin/env python3
"""
Human-oriented structured chunking for policy documents.

Goal: Produce chunks that make sense to a human reader by respecting
document structure first (headings/sections/lists), then paragraphs,
and only then sentences for overflow. Avoid token overlap to keep each
chunk as an atomic provision/article where possible.

Input:  data/input/documents_cleaned.csv  (name, text)
Output: data/output/documents_chunked_structured_human.csv (default)

Columns:
  - name:          source filename (e.g., aalto-university.md)
  - chunk_id:      sequential id per document (0-based)
  - heading_path:  breadcrumb of headings (e.g., "Policy > Scope")
  - section_label: detected section/article/list label (if any)
  - text:          chunk content
  - num_tokens:    token count (word-based by default)
  - num_sentences: sentence count

Configuration via CLI args. Defaults chosen for qualitative analysis.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------- Token utilities ----------------------------

def word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+", text)


def num_tokens(text: str) -> int:
    return len(word_tokens(text))


def split_sentences(text: str) -> List[str]:
    # Simple sentence split that also accounts for newlines
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


# -------------------------- Structure detection --------------------------

HEADING_RE = re.compile(r"^\s*(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")

SECTION_RE = re.compile(
    r"^\s*(?:(?P<label>(Article|Section|Clause|Policy|Principle|Guideline|Scope|Purpose|Definitions))\s+\d+[.:)]?\s*|(?P<num>(\d+(?:\.\d+){0,3}|[A-Za-z]\.|[ivxlcdm]+\.|\([A-Za-zivxlcdm]+\)))\s+)(?P<title>.+)$",
    re.IGNORECASE,
)

BULLET_RE = re.compile(r"^\s*[-*•]\s+.+$")


def classify_line(line: str) -> Tuple[str, Optional[dict]]:
    """Return (type, info) where type in {heading, section, bullet, numbered, para}.
    """
    m = HEADING_RE.match(line)
    if m:
        level = len(m.group("hashes"))
        return "heading", {"level": level, "title": m.group("title").strip()}

    m = SECTION_RE.match(line)
    if m:
        title = (m.group("title") or "").strip()
        label = m.group("label") or m.group("num")
        return "section", {"label": label.strip() if label else None, "title": title}

    if BULLET_RE.match(line):
        return "bullet", {}

    # Numbered lists not captured by SECTION_RE: e.g., "1.", "1)"
    if re.match(r"^\s*\d+[.)]\s+", line):
        return "numbered", {}

    return "para", {}


# ----------------------------- Chunk builder -----------------------------

@dataclass
class Context:
    headings: List[str] = field(default_factory=list)
    section_label: Optional[str] = None

    def path(self) -> str:
        return " > ".join(h for h in self.headings if h)


def pack_sentences(sentences: List[str], max_tokens: int) -> List[str]:
    out, cur, cur_tokens = [], [], 0
    for s in sentences:
        t = num_tokens(s)
        if cur and cur_tokens + t > max_tokens:
            out.append(" ".join(cur).strip())
            cur, cur_tokens = [s], t
        else:
            cur.append(s)
            cur_tokens += t
    if cur:
        out.append(" ".join(cur).strip())
    return out


def flush_chunk(chunks: List[dict], name: str, chunk_id: int, ctx: Context, buf: List[str]) -> int:
    if not buf:
        return chunk_id
    text = "\n".join(buf).strip()
    if not text:
        return chunk_id
    chunks.append({
        "name": name,
        "chunk_id": chunk_id,
        "heading_path": ctx.path(),
        "section_label": ctx.section_label or "",
        "text": text,
        "num_tokens": num_tokens(text),
        "num_sentences": len(split_sentences(text)),
    })
    return chunk_id + 1


def chunk_document_human(name: str, text: str, max_tokens: int, min_tokens: int) -> List[dict]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    chunks: List[dict] = []
    ctx = Context()
    buf: List[str] = []
    chunk_id = 0

    def maybe_flush(force: bool = False):
        nonlocal chunk_id, buf
        if force or (buf and num_tokens("\n".join(buf)) >= max_tokens):
            # If over max, split by paragraph, then by sentences
            material = "\n".join(buf)
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", material) if p.strip()]
            for para in paragraphs:
                if num_tokens(para) <= max_tokens:
                    chunk_id = flush_chunk(chunks, name, chunk_id, ctx, [para])
                else:
                    # fallback: sentence-based packing inside paragraph
                    parts = pack_sentences(split_sentences(para), max_tokens)
                    for part in parts:
                        chunk_id = flush_chunk(chunks, name, chunk_id, ctx, [part])
            buf = []

    for raw in lines:
        ltype, info = classify_line(raw)

        # On structure boundaries, flush current buffer first
        if ltype in ("heading", "section"):
            maybe_flush(force=True)

        if ltype == "heading":
            # Update heading path by level
            level = info.get("level", 1)
            title = info.get("title", "").strip()
            # Ensure list size
            while len(ctx.headings) < level:
                ctx.headings.append("")
            ctx.headings = ctx.headings[:level]
            ctx.headings[level - 1] = title
            ctx.section_label = None
            continue  # Do not include heading line in buffer; add as context only

        if ltype == "section":
            label = info.get("label")
            title = info.get("title", "").strip()
            ctx.section_label = label if label else title
            # Keep the section title as the first line of the next chunk for readability
            if title:
                buf.append(title)
            maybe_flush()
            continue

        # List grouping: accumulate consecutive bullets/numbered items into coherent blocks
        if ltype in ("bullet", "numbered"):
            buf.append(raw)
            if num_tokens("\n".join(buf)) >= max_tokens:
                maybe_flush(force=True)
            continue

        # Paragraph/text lines: accumulate. Blank-line boundaries improve paragraph detection
        buf.append(raw)
        if not raw.strip():
            maybe_flush()

    # Final flush
    maybe_flush(force=True)

    # Merge tiny trailing chunks into previous to satisfy min_tokens
    merged: List[dict] = []
    for ch in chunks:
        if merged and ch["num_tokens"] < min_tokens:
            prev = merged[-1]
            # Merge with a blank line separator
            combined_text = (prev["text"] + "\n\n" + ch["text"]).strip()
            prev["text"] = combined_text
            prev["num_tokens"] = num_tokens(combined_text)
            prev["num_sentences"] = len(split_sentences(combined_text))
        else:
            merged.append(ch)

    # Reindex chunk_id sequentially
    for i, ch in enumerate(merged):
        ch["chunk_id"] = i

    return merged


def run(input_csv: str, output_csv: str, max_tokens: int, min_tokens: int) -> None:
    # Read documents
    rows: List[dict] = []
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Expect columns: name, text
            name = row.get("name")
            text = row.get("text")
            if name and text:
                rows.append({"name": name, "text": text})

    if not rows:
        raise RuntimeError(f"No documents found in {input_csv}")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    fieldnames = [
        "name", "chunk_id", "heading_path", "section_label", "text", "num_tokens", "num_sentences",
    ]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for doc in rows:
            name, text = doc["name"], doc["text"]
            chunks = chunk_document_human(name, text, max_tokens=max_tokens, min_tokens=min_tokens)
            for ch in chunks:
                writer.writerow(ch)


def main():
    ap = argparse.ArgumentParser(description="Human-oriented structured chunking")
    ap.add_argument("--input", default="data/input/documents_cleaned.csv")
    ap.add_argument(
        "--output", default="data/output/documents_chunked_structured_human.csv",
        help="Output CSV path"
    )
    ap.add_argument("--max_tokens", type=int, default=512, help="Max tokens per chunk (word-based)")
    ap.add_argument("--min_tokens", type=int, default=40, help="Min tokens; merge small trailing chunks")
    args = ap.parse_args()

    run(args.input, args.output, args.max_tokens, args.min_tokens)
    print(f"Wrote chunks to {args.output}")


if __name__ == "__main__":
    main()
