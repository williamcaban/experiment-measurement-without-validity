#!/usr/bin/env python3
"""
extract.py - build the rater stimulus: the full text of all 55 papers.

WHAT THE STIMULUS IS, AND WHY IT IS THE WHOLE DOCUMENT
------------------------------------------------------
Each paper is extracted whole - body, bibliography and appendices - and handed
to the raters unmodified. No span is removed. Three reasons:

1. The author coded these papers from the PDFs. Krippendorff's alpha assumes
   interchangeable observers coding the SAME units, so removing a span from the
   raters' input and not from the author's would compare codes made on
   different material.

2. Removing the bibliography requires locating where it ends, and that is the
   step that goes wrong. The earlier approach - truncate at the "References"
   heading - discarded 69% of G6-length-bias including the Appendix D.2 where
   its Fleiss' kappa is reported, and more than 25% of the text in 30 of the 55
   papers, because most place appendices after the bibliography.

3. A bibliography is a hard case for the coding instrument, not a defect in the
   stimulus. Q1 asks which coefficient the authors COMPUTED; a reference to
   Cohen (1960) is a citation. The instrument says so explicitly and was
   validated on the four papers whose bibliography names a coefficient their
   body never computes. Deleting the hard case would be measuring the
   instrument on an easier task than the one it is claimed to perform.

WHAT IS COMMITTED, AND WHAT IS NOT
----------------------------------
The 55 papers are third-party copyrighted works, so the extracted text is
NEVER committed; .gitignore enforces it. What is committed is
stimulus_hashes.csv: one row per paper with the source PDF's checksum and the
extracted text's checksum. Replication is therefore: obtain the same PDFs
(manifest.csv records every identifier, version and checksum), install the
pinned engine, run `extract.py --verify`, and see the checksums match. That is
a stronger guarantee than shipping the text, because it also proves the
replicator's extraction toolchain agrees.

Extraction itself is pinned to an exact MuPDF build and is defined once, in
corpus.py, so no two scripts can build the text differently.

USAGE
  python3 extract.py --build     extract, write stimulus + hashes + roster
  python3 extract.py --verify    re-extract and compare against the hashes
  python3 extract.py --status    summarise what exists on disk
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import corpus

HERE = Path(__file__).parent
HOME = Path.home()

MANIFEST = HERE / "manifest.csv"
STIMULUS = HERE / "stimulus_fulltext.json"   # gitignored: copyrighted text
HASHES = HERE / "stimulus_hashes.csv"        # committed: the shareable artifact
ROSTER = HERE / "papers_55.csv"              # committed: paper_id list for runs

HASH_FIELDS = ["paper_id", "zotero_key", "category", "pages",
               "pdf_sha256", "text_sha256", "text_chars", "text_lines"]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def extract_all() -> list[dict]:
    """Extract every paper, asserting each PDF is the one the manifest names."""
    bind, mupdf = corpus.require_engine()
    print(f"engine: pymupdf {bind} / MuPDF {mupdf}\n")

    rows = list(csv.DictReader(open(MANIFEST, newline="")))
    if not rows:
        sys.exit(f"FATAL: {MANIFEST.name} is empty")

    out, problems = [], []
    for r in rows:
        pid = r["paper_id"]
        pdf = HOME / r["local_pdf"]
        if not pdf.exists():
            problems.append(f"{pid}: PDF not found at {r['local_pdf']}")
            continue

        raw = pdf.read_bytes()
        got = sha256_bytes(raw)
        want = (r.get("source_sha256") or "").strip()
        # A changed PDF silently changes the stimulus, so this is fatal rather
        # than a warning: every downstream number is conditioned on it.
        if want and got != want:
            problems.append(
                f"{pid}: PDF checksum mismatch\n"
                f"       manifest {want}\n"
                f"       on disk  {got}")
            continue

        lines = corpus.canonical_lines(pdf)
        text = corpus.canonical_text(lines)
        if not text.strip():
            problems.append(f"{pid}: extraction produced no text")
            continue

        out.append({
            "paper_id": pid,
            "zotero_key": r["zotero_key"],
            "category": r.get("category", ""),
            "title": r.get("title", ""),
            "pages": int(r["pdf_pages"]) if r.get("pdf_pages") else 0,
            "pdf_sha256": got,
            "text_sha256": sha256_text(text),
            "text_chars": len(text),
            "text_lines": len(lines),
            "text": text,
        })

    if problems:
        print("EXTRACTION PROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"  ! {p}", file=sys.stderr)
        sys.exit(f"\nFATAL: {len(problems)} paper(s) could not be extracted "
                 f"reproducibly; refusing to write a partial stimulus.")
    return out


def build() -> None:
    recs = extract_all()

    # ensure_ascii=True by design. The stimulus is a faithful extraction of real
    # PDFs, so it legitimately contains ligatures, Greek letters, math operators,
    # form feeds and a few font-specific Private Use Area glyphs from display
    # equations. Those are kept in the text, because normalising them would mean
    # editing the stimulus the raters actually saw. Escaping them on disk is a
    # different matter: text_sha256 is computed over the decoded string, not over
    # the file bytes, so escaping changes no hash and no rater input while making
    # the file safe to open in an editor that warns about ambiguous Unicode.
    STIMULUS.write_text(json.dumps(recs, ensure_ascii=True, indent=1) + "\n")

    with open(HASHES, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=HASH_FIELDS)
        w.writeheader()
        for r in recs:
            w.writerow({k: r[k] for k in HASH_FIELDS})

    # full_scan_55papers.csv keys papers by zotero_key and has no paper_id
    # column, so it cannot serve as a roster for the runner. Emit one.
    with open(ROSTER, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["paper_id", "zotero_key", "category"])
        w.writeheader()
        for r in recs:
            w.writerow({k: r[k] for k in ("paper_id", "zotero_key", "category")})

    chars = sum(r["text_chars"] for r in recs)
    big = max(recs, key=lambda r: r["text_chars"])
    print(f"papers extracted : {len(recs)}")
    print(f"total characters : {chars:,}")
    print(f"largest paper    : {big['paper_id']} "
          f"({big['text_chars']:,} chars, ~{big['text_chars'] // 4:,} tokens)")
    print(f"mean per paper   : {chars // len(recs):,} chars")
    print()
    print(f"wrote {STIMULUS.name:26} (gitignored - copyrighted text)")
    print(f"wrote {HASHES.name:26} (committed - the replication artifact)")
    print(f"wrote {ROSTER.name:26} (committed - roster for --papers runs)")
    print()
    print("next: python3 judges.py --build   # re-derive the context floor "
          "from this stimulus")


def verify() -> None:
    if not HASHES.exists():
        sys.exit(f"FATAL: {HASHES.name} not found; run --build first.")
    want = {r["paper_id"]: r for r in csv.DictReader(open(HASHES, newline=""))}
    recs = extract_all()
    got = {r["paper_id"]: r for r in recs}

    bad = []
    for pid, w in want.items():
        g = got.get(pid)
        if g is None:
            bad.append(f"{pid}: missing from this extraction")
            continue
        if g["pdf_sha256"] != w["pdf_sha256"]:
            bad.append(f"{pid}: different source PDF\n"
                       f"       expected {w['pdf_sha256']}\n"
                       f"       got      {g['pdf_sha256']}")
        elif g["text_sha256"] != w["text_sha256"]:
            bad.append(f"{pid}: same PDF, different extracted text - your "
                       f"extraction toolchain differs\n"
                       f"       expected {w['text_sha256']}\n"
                       f"       got      {g['text_sha256']}")
    extra = sorted(set(got) - set(want))

    print(f"papers checked   : {len(want)}")
    print(f"checksum matches : {len(want) - len(bad)}/{len(want)}")
    if extra:
        print(f"not in hashes    : {', '.join(extra)}")
    if bad:
        print("\nMISMATCHES:")
        for b in bad:
            print(f"  ! {b}")
        sys.exit(1)
    print("\nevery paper reproduces byte-identically from its source PDF")


def status() -> None:
    for f, note in ((MANIFEST, "source identifiers + PDF checksums"),
                    (STIMULUS, "extracted text (gitignored)"),
                    (HASHES, "checksums (committed)"),
                    (ROSTER, "paper roster (committed)")):
        if f.exists():
            n = sum(1 for _ in open(f)) - 1 if f.suffix == ".csv" else \
                len(json.loads(f.read_text()))
            print(f"  {f.name:26} {n:>4} records   {note}")
        else:
            print(f"  {f.name:26} {'--':>4}            {note} - ABSENT")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    elif a.verify:
        verify()
    else:
        status()
