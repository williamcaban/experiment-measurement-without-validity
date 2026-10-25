#!/usr/bin/env python3
"""
test_extract.py - regression tests for stimulus construction.

Two things are guarded here.

COPYRIGHT. The 55 papers are third-party works and their extracted text must
never be committed. That is a licensing constraint, not a preference, and it is
one `git add .` away from being violated, so it is asserted here rather than
left to a line in a README.

REPRODUCIBILITY. The stimulus is the independent variable of the whole
experiment: if it cannot be rebuilt byte-for-byte from the source PDFs, no
reported number means anything. These tests assert the chain holds end to end -
manifest checksum to PDF, pinned engine, extracted text checksum - and that it
FAILS when any link is broken.
"""

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import corpus  # noqa: E402

passed = failed = 0


def check(label: str, ok: bool, note: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {label}" + (f" - {note}" if note else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f" - {note}" if note else ""))


print("== 1. copyrighted text cannot be committed ==")
for name in ("stimulus_fulltext.json", "fulltext_20papers.json",
             "excerpts_verbatim.json"):
    r = subprocess.run(["git", "check-ignore", "-q", name], cwd=HERE)
    check(f"{name} is gitignored", r.returncode == 0,
          "extracted primary-source text must never be committed")
tracked = subprocess.run(["git", "ls-files"], cwd=HERE,
                         capture_output=True, text=True).stdout.split()
check("no full-text file is tracked",
      not any(t.startswith("stimulus_") and t.endswith(".json") for t in tracked)
      and "fulltext_20papers.json" not in tracked
      and "excerpts_verbatim.json" not in tracked,
      f"{len(tracked)} files tracked")
check("the checksum artifact IS committable",
      subprocess.run(["git", "check-ignore", "-q", "stimulus_hashes.csv"],
                     cwd=HERE).returncode != 0,
      "hashes are the shareable replication artifact")

print("\n== 2. the extraction engine is pinned, not merely requested ==")
bind, mupdf = corpus.require_engine()
check("pinned engine is present", (bind, mupdf) ==
      (corpus.PINNED_PYMUPDF, corpus.PINNED_MUPDF), f"pymupdf {bind}/MuPDF {mupdf}")
src = (HERE / "corpus.py").read_text()
check("require_engine aborts rather than warns", "raise SystemExit" in src)
check("the pin names an exact MuPDF build, not just a package version",
      corpus.PINNED_MUPDF in src and corpus.PINNED_PYMUPDF in src)

print("\n== 3. the stimulus reproduces from its sources ==")
HASHES = HERE / "stimulus_hashes.csv"
MANIFEST = HERE / "manifest.csv"
check("stimulus_hashes.csv exists", HASHES.exists(),
      "run: python3 extract.py --build")
if not HASHES.exists():
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1)

hashes = {r["paper_id"]: r for r in csv.DictReader(open(HASHES, newline=""))}
man = {r["paper_id"]: r for r in csv.DictReader(open(MANIFEST, newline=""))}
check("every manifest paper has a stimulus hash",
      set(man) == set(hashes), f"{len(man)} manifest, {len(hashes)} hashed")
check("all 55 papers present", len(hashes) == 55, str(len(hashes)))
check("no two papers share extracted text",
      len({h["text_sha256"] for h in hashes.values()}) == len(hashes),
      "identical text would mean a copy-paste error in extraction")
check("the PDF checksum in the hashes matches the manifest",
      all(hashes[p]["pdf_sha256"] == man[p]["source_sha256"]
          for p in hashes if man[p].get("source_sha256")),
      "the stimulus is bound to the exact source documents")

# Re-extract two papers and confirm the recorded text checksum reproduces.
# Two rather than all 55 because this runs on every change; --verify does all.
PROBE = ["G6-length-bias", "C7-neurips-data-curation"]
for pid in PROBE:
    pdf = Path.home() / man[pid]["local_pdf"]
    text = corpus.canonical_text(corpus.canonical_lines(pdf))
    got = hashlib.sha256(text.encode("utf-8")).hexdigest()
    check(f"{pid} text checksum reproduces", got == hashes[pid]["text_sha256"],
          f"{got[:16]}... vs {hashes[pid]['text_sha256'][:16]}...")
    check(f"{pid} recorded char count is right",
          len(text) == int(hashes[pid]["text_chars"]))

print("\n== 4. the whole document is kept ==")
g6_pdf = Path.home() / man["G6-length-bias"]["local_pdf"]
g6 = corpus.canonical_text(corpus.canonical_lines(g6_pdf))
check("G6 retains its bibliography",
      "References" in g6 or "REFERENCES" in g6)
check("G6 retains the appendix that reports its Fleiss' kappa",
      "Fleiss" in g6, "the span the old truncation discarded")
check("G6 retains Appendix D.2 specifically", "D.2" in g6)
idx_refs = g6.find("References")
check("the Fleiss report sits AFTER the References heading",
      idx_refs != -1 and g6.find("Fleiss") > idx_refs,
      "so any truncate-at-References rule loses it")

print("\n== 5. the roster is usable by the runner ==")
ROSTER = HERE / "papers_55.csv"
check("papers_55.csv exists", ROSTER.exists())
if ROSTER.exists():
    roster = list(csv.DictReader(open(ROSTER, newline="")))
    check("roster has a paper_id column",
          "paper_id" in (roster[0].keys() if roster else {}),
          "full_scan_55papers.csv does not, which is why this file exists")
    check("roster covers the whole corpus", len(roster) == 55, str(len(roster)))
    stim = HERE / "stimulus_fulltext.json"
    if stim.exists():
        ids = {r["paper_id"] for r in json.loads(stim.read_text())}
        check("every roster paper is in the stimulus",
              {r["paper_id"] for r in roster} <= ids)

print("\n" + "=" * 58)
print(f"{passed} passed, {failed} failed")
print("=" * 58)
sys.exit(1 if failed else 0)
