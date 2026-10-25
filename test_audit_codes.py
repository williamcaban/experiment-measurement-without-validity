#!/usr/bin/env python3
"""
test_audit_codes.py - regression tests for the lexical audit.

WHY THESE TESTS EXIST
---------------------
audit_codes.py is the error detector for the Q1 coding. It used to truncate
each paper at its LAST "References" heading and discard everything after it. On
G6-length-bias that discarded 69% of the document, including the Appendix D.2
where Fleiss' kappa is reported, so the audit emitted
`author_Q1=NIL, lexical_Q1=NIL, agrees=1` and CONFIRMED the very miscode it
exists to catch.

The central test below asserts that G6's appendix sentence is inside the text
the audit scans, and that it would NOT have been had the old rule survived.
Everything else here guards the distinction the fix depends on: a coefficient
next to a value is reported, a coefficient in a reference list is cited.

Offline apart from reading PDFs out of the local Zotero store.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import audit_codes as ac  # noqa: E402

passed = failed = 0


def check(label: str, ok: bool, note: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {label}" + (f" - {note}" if note else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f" - {note}" if note else ""))


print("== 1. the appendix is no longer discarded (the original bug) ==")
G6 = "IDEMAUR9"
text, refs_at = ac.document_text(G6)
head = text[:refs_at]
tail = text[refs_at:]
check("G6 text is scanned whole", len(text) > 100_000, f"{len(text):,} chars")
check("G6 names Fleiss somewhere in the document",
      len(re.findall(r"fleiss", text, re.I)) == 3,
      f"{len(re.findall(r'fleiss', text, re.I))} occurrences")
check("every Fleiss mention is AFTER the References heading",
      "fleiss" not in head.lower() and "fleiss" in tail.lower(),
      "which is why truncating there hid the finding")
check("the appendix sentence itself is present",
      "inter-rater reliability analysis using Fleiss" in text
      or "Fleiss’ Kappa" in text or "Fleiss' Kappa" in text)
check("Appendix D.2 survives into the scanned text", "D.2" in tail)
check("the discarded fraction would have been catastrophic",
      len(tail) / len(text) > 0.5,
      f"the old rule dropped {100 * len(tail) / len(text):.0f}% of this paper")

print("\n== 2. region labels claim only what is known ==")
check("G6's Fleiss mention is labelled POST_REFS",
      ac.first_region(dict(ac.DETECT)["KFL"], text, refs_at) == "POST_REFS")
# Behavioural, not a source grep: first_region's own comment contains the
# string "BIB" to explain why it is not used as a label.
import csv  # noqa: E402
_report = HERE / "audit_report.csv"
if _report.exists():
    seen = set()
    for _r in csv.DictReader(open(_report)):
        for _pair in (_r.get("regions") or "").split("|"):
            if ":" in _pair:
                seen.add(_pair.split(":", 1)[1])
    check("every region emitted is BODY or POST_REFS",
          seen <= {"BODY", "POST_REFS"}, f"observed {sorted(seen)}")
    check("both labels actually occur in the corpus",
          {"BODY", "POST_REFS"} <= seen,
          "a label that never fires is not evidence of anything")
else:
    check("audit_report.csv present for region check", False,
          "run: python3 audit_codes.py")
body_only = ac.first_region(re.compile("the", re.I), text, refs_at)
check("a match before the heading is labelled BODY", body_only == "BODY")

print("\n== 3. cited vs computed ==")
KAPPA = re.compile(r"cohen['’]?s?\s*kappa", re.I)
FLEISS = re.compile(r"fleiss", re.I)
ICC = re.compile(r"\bICC\b", re.I)

cases = [
    # (label, pattern, text, expected is_computed)
    ("a reference-list entry is not computed", KAPPA,
     "Cohen, J. 1960. A coefficient of agreement for nominal scales. "
     "Educational and Psychological Measurement 20(1):37-46.", False),
    ("a reference entry whose TITLE names kappa is not computed", KAPPA,
     "Cohen's kappa revisited: a critique. Psychological Bulletin "
     "76(5):378-382, 1971.", False),
    ("a DOI near the name is not a coefficient value", KAPPA,
     "Cohen's kappa and agreement. doi: 10.18653/v1/2021.naacl-main.80", False),
    ("a tutorial mention is not computed", KAPPA,
     "Cohen's kappa is one of the earliest and most commonly used "
     "inter-annotator agreement measures for categorical data.", False),
    ("an explicit value is computed", FLEISS,
     "We report Fleiss' kappa = 0.68 across the three annotators.", True),
    ("a value introduced by 'of' is computed", KAPPA,
     "aggregated human majority vote reached 91.67%, with a Cohen's "
     "Kappa of 0.734.", True),
    ("a range is computed", ICC,
     "inter-rater reliability was poor, with ICC ranging from 0.09 to 0.30",
     True),
    ("a value stated before the name is computed", FLEISS,
     "agreement was 0.81 by Fleiss' kappa", True),
]
for label, pat, txt, want in cases:
    check(label, ac.is_computed(pat, txt) is want,
          f"expected is_computed={want}")

print("\n== 4. the Greek fold still holds ==")
check("mathematical italic kappa folds to a word",
      "kappa" in ac.fold_greek("Cohen's \U0001d705 of 0.767"),
      "E1-agentprocessbench reported its kappa this way")
check("a folded kappa with a value reads as computed",
      ac.is_computed(KAPPA, ac.fold_greek("Cohen's \U0001d705 of 0.767")))

print("\n== 5. the audit still refuses to be a rater ==")
src = (HERE / "audit_codes.py").read_text()
check("no alpha or kappa is computed anywhere in this script",
      "krippendorff" not in src.lower().replace("krippendorff’s", "")
      or "import krippendorff" not in src,
      "it is an error detector and a criterion, never an observer")
check("the docstring says so explicitly",
      "must never enter the reliability matrix" in src)

print("\n" + "=" * 58)
print(f"{passed} passed, {failed} failed")
print("=" * 58)
sys.exit(1 if failed else 0)
