#!/usr/bin/env python3
"""
audit_codes.py — lexical audit of the 55-paper Q1 coding.

WHAT THIS IS NOT
----------------
This is NOT a rater and its output must never enter the reliability matrix or
produce a kappa or alpha. A deterministic rule agrees with itself on every
application, so it is trivially self-consistent regardless of whether it
operationalizes the construct correctly — the argument this paper makes in
Section 9 about deterministic single-rule graders applies to this script
verbatim. Krippendorff's alpha estimates agreement among interchangeable
observers; a fixed rule is a criterion, not an observer.

WHAT IT IS
----------
Two things:

1. An ERROR DETECTOR. Q1 ("which IRR metric does the paper name?") is close to a
   lexical question by construction. Any paper coded Q1=NIL whose full text
   nevertheless names a recognised coefficient is either a coding error or a case
   that needs an explicit note. Every flag is a prompt for human adjudication,
   never a verdict: a paper may name Cohen's kappa only in related work.

2. A LEXICAL-DETERMINACY MEASURE. Reporting how far surface lexical features
   alone reproduce each coded dimension quantifies how much of the coding
   actually required judgement — which is the argument for a human second coder,
   not a substitute for one.

Reads the author codes and resolves each paper's PDF through the local Zotero
store.

HOW THE BIBLIOGRAPHY IS HANDLED, AND WHY THIS CHANGED
-----------------------------------------------------
A bibliography entry for Cohen 1960 is not a reported coefficient, so the
reference list has to be discounted somehow. This script used to do that by
finding the LAST "References" heading and discarding everything from there to
the end of the file. That was wrong, and provably so: on G6-length-bias it
discarded 82,196 of 119,723 characters (69% of the paper), including the
Appendix D.2 where all three occurrences of "Fleiss" live. The audit therefore
reported `author_Q1=NIL, lexical_Q1=NIL, agrees=1` and CONFIRMED a miscode that
it exists to catch. Across the 55-paper corpus that truncation discarded more
than 25% of the text in 30 papers, because most of them place appendices after
the bibliography.

So nothing is discarded now. The whole document is scanned, and the separation
between "cited" and "computed" is made the way the coding instrument makes it:
a computed coefficient appears next to a VALUE. Each flag reports the region
the match fell in (before or after the first References heading) so that an
adjudicator can see at a glance whether a hit is bibliography-shaped, without
this script having to locate the end of the bibliography - the task that
produced the original bug.

Two lexical codes are reported per paper:
  lexical_Q1_named    - any recognised coefficient named anywhere
  lexical_Q1_computed - only coefficients that appear next to a value
The second is the construct the instrument asks for, so it is the one scored
for determinacy.

Output: audit_report.csv
"""

import csv
import json
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent

# zotero_key -> PDF path, taken from the committed manifest rather than from a
# live Zotero database, so the audit runs against the same corpus the manifest
# pins by checksum and does not depend on the author's local library.
HOME = Path.home()
PDFMAP = {r["zotero_key"]: str(HOME / r["local_pdf"])
          for r in csv.DictReader(open(HERE / "manifest.csv"))
          if r.get("local_pdf")}

# Ordered exactly as the coding instrument's Q1 precedence.
DETECT = [
    # Text is Greek-folded before matching, so "kappa"/"alpha" cover all variants.
    ("KCO", re.compile(r"cohen[’'’]?s?\s*kappa|kappa\s*\(?\s*cohen", re.I)),
    ("KFL", re.compile(r"fleiss", re.I)),
    ("KAL", re.compile(r"krippendorff", re.I)),
    ("ICC", re.compile(r"intraclass\s+correlation|\bICC\b", re.I)),
    ("PA",  re.compile(r"(?:percent|percentage|proportion|inter-?annotator|"
                       r"inter-?rater)\s+agreement|agreement\s+rate", re.I)),
]

REF_HEAD = re.compile(r"\n\s*(references|bibliography|works cited)\s*\n", re.I)

# Naming a coefficient is not reporting one: a paper may explain Cohen's kappa
# in a tutorial section, or critique it, without computing it. A REPORTED
# coefficient almost always sits next to a value. This second-order test is what
# separates the two, and it is what caught the two genuine miscodes in the scan
# (I2-piot-xrr, I3-kunilovskaya) that the name-only test buried among false
# positives.
# The cue list is deliberately explicit rather than "a number anywhere near the
# name", because bibliographies are full of numbers - volumes, pages, years,
# DOIs - and a bare proximity test reads "Psychological Bulletin 76(5):378-382"
# as a coefficient. Each cue is a way authors actually introduce a measured
# value. "ranging from" was added after I1-jafari's "ICC ranging from 0.09 to
# 0.30" - an unambiguously computed coefficient - read as merely discussed.
_CUE = (r"=|of|was|were|is|are|:|reaches|reaching|from|between|range[ds]?|"
        r"ranging|averaged?|yield(?:s|ed|ing)?|achiev(?:es|ed|ing)|obtained|"
        r"scored|measured at")
VALUE_NEAR = re.compile(
    r"(kappa|alpha|\bICC\b)[^.]{0,40}?(?:" + _CUE + r")\s*"
    r"[-\u2013]?\s*0?\.\d+"
    r"|0?\.\d+[^.]{0,25}?(kappa|alpha|\bICC\b)", re.I)

# Papers typeset the same letter with several different codepoints: U+03BA
# GREEK SMALL LETTER KAPPA, U+03F0 GREEK KAPPA SYMBOL, U+1D705 MATHEMATICAL
# ITALIC SMALL KAPPA, and the other math-alphanumeric blocks. Matching only
# U+03BA silently misses "Cohen's 𝜅 of 0.767" — which it did, on
# E1-agentprocessbench, until this fold was added. Fold every variant to a
# spelled-out name before matching.
_GREEK_FOLD = {}
for _cp in list(range(0x370, 0x400)) + list(range(0x1D6A8, 0x1D800)):
    try:
        _nm = unicodedata.name(chr(_cp))
    except ValueError:
        continue
    for _letter in ("KAPPA", "ALPHA"):
        if _nm.endswith(_letter):
            _GREEK_FOLD[_cp] = f" {_letter.lower()} "


def fold_greek(s: str) -> str:
    return s.translate(_GREEK_FOLD)


def document_text(key: str) -> tuple[str, int]:
    """The whole document, plus the offset where the bibliography starts.

    Nothing is truncated. The offset is the FIRST References heading, which is
    where the bibliography begins in both layouts (bibliography last, and
    bibliography followed by appendices). It is used only to label where a
    match fell, never to discard text.
    """
    raw = subprocess.run(["pdftotext", "-q", PDFMAP[key], "-"],
                         capture_output=True, text=True).stdout
    m = REF_HEAD.search(raw)
    # Measure the cut point on the same string that gets normalised, so the
    # offset still refers to the text that is actually searched.
    head = raw[:m.start()] if m else raw
    flat_head = re.sub(r"\s+", " ", fold_greek(head.replace("-\n", "")))
    flat = re.sub(r"\s+", " ", fold_greek(raw.replace("-\n", "")))
    return flat, (len(flat_head) if m else len(flat))


def context(pat: re.Pattern, text: str, width: int = 110) -> str:
    m = pat.search(text)
    if not m:
        return ""
    a = max(0, m.start() - width // 2)
    return text[a:m.start() + width].strip()


def first_region(pat: re.Pattern, text: str, refs_at: int) -> str:
    """Where this metric is first named: before or after the bibliography."""
    m = pat.search(text)
    if not m:
        return ""
    # Not "BIB": text after the References heading is a late bibliography
    # entry in some layouts and an appendix in others, and deciding which is
    # the boundary problem that produced the original bug. POST_REFS states
    # only what is known.
    return "BODY" if m.start() < refs_at else "POST_REFS"


def is_computed(pat: re.Pattern, text: str) -> bool:
    """True when some mention of this metric sits next to a value.

    This is the instrument's own test for "computed rather than cited", applied
    per metric instead of over the whole document, so a value belonging to one
    coefficient cannot vouch for another. VALUE_NEAR is reused unchanged
    because it demands a relational cue ("= of was is :") before the number,
    which is what keeps DOIs and page ranges in a bibliography from reading as
    coefficient values.
    Known false negatives: a coefficient whose value appears only in a TABLE
    carries no relational cue near its name, so it reads as merely discussed.
    G6-length-bias is the worked example - it reports Fleiss' kappa in
    Appendix D.2 with the values in Table 9, and this test returns False for
    it. So `computed` is a floor, never a verdict, and both lexical codes are
    reported rather than one being presented as the truth.
    """
    for m in pat.finditer(text):
        window = text[max(0, m.start() - 140): m.end() + 240]
        if VALUE_NEAR.search(window):
            return True
    return False


def main():
    rows = list(csv.DictReader(open(HERE / "full_scan_55papers.csv")))
    ids = {r["zotero_key"]: r["paper_id"]
           for r in csv.DictReader(open(HERE / "papers_55.csv"))}
    bypat = dict(DETECT)

    out, flags = [], []
    for r in rows:
        key = r["zotero_key"]
        pid = ids.get(key, key)
        aq1, aq4 = r["author_Q1"], r["author_Q4"]
        if key not in PDFMAP:
            out.append(dict(paper_id=pid, author_Q1=aq1, author_Q4=aq4,
                            lexical_Q1_named="NO_PDF", lexical_Q1_computed="NO_PDF",
                            agrees="", named="", computed="", regions="",
                            n_values="", evidence=""))
            continue

        t, refs_at = document_text(key)
        named = [code for code, pat in DETECT if pat.search(t)]
        comp = [code for code in named if is_computed(bypat[code], t)]
        regions = {code: first_region(bypat[code], t, refs_at) for code in named}

        # The instrument asks which metric the authors COMPUTED, so that is the
        # code scored for determinacy. Precedence follows DETECT's order.
        lex_named = named[0] if named else "NIL"
        lex_comp = comp[0] if comp else "NIL"
        agrees = (lex_comp == aq1)

        vals = [m.group(0).strip() for m in VALUE_NEAR.finditer(t)]

        # Flag A: coded as reporting nothing, yet a coefficient is present.
        # Severity now distinguishes three genuinely different situations,
        # where the old script collapsed them into one and hid the real cases
        # among bibliography noise.
        if aq1 == "NIL":
            real = [c for c in comp if c != "PA"]
            disc = [c for c in named if c != "PA" and c not in comp]
            if real:
                flags.append((pid, "NIL_BUT_COEFFICIENT_COMPUTED", aq1, lex_comp,
                              regions.get(real[0], ""),
                              context(bypat[real[0]], t)))
            elif any(regions.get(c) == "POST_REFS" for c in disc):
                # This is G6's shape: a coefficient named only after the
                # References heading, which may be a real finding reported in
                # an appendix. It ranks above a body mention precisely because
                # the old truncation could not see it at all.
                c = next(c for c in disc if regions.get(c) == "POST_REFS")
                flags.append((pid, "NIL_BUT_METRIC_NAMED_POST_REFS", aq1,
                              lex_named, "POST_REFS", context(bypat[c], t)))
            elif disc:
                flags.append((pid, "NIL_BUT_METRIC_DISCUSSED_IN_BODY", aq1,
                              lex_named, "BODY", context(bypat[disc[0]], t)))
        # Flag B: coded as naming a coefficient that never appears at all.
        if aq1 in ("KCO", "KFL", "KAL", "ICC") and aq1 not in named:
            flags.append((pid, "METRIC_CODED_BUT_ABSENT", aq1, lex_named, "", ""))
        # Flag B2: the metric is named but no value sits near it. Usually the
        # value is in a table (see is_computed), so this is a prompt to confirm
        # the paper really computed it, not a claim that it did not.
        if (aq1 in ("KCO", "KFL", "KAL", "ICC")
                and aq1 in named and aq1 not in comp):
            flags.append((pid, "METRIC_CODED_BUT_NOT_VALUE_ANCHORED", aq1,
                          lex_comp, regions.get(aq1, ""),
                          context(bypat[aq1], t)))
        # Flag C: Q4=ABS is only valid when Q1=NIL (strict rule).
        if aq4 == "ABS" and aq1 != "NIL":
            flags.append((pid, "ABS_WITH_NON_NIL_Q1", aq1, lex_named, "", ""))

        out.append(dict(paper_id=pid, author_Q1=aq1, author_Q4=aq4,
                        lexical_Q1_named=lex_named,
                        lexical_Q1_computed=lex_comp,
                        agrees=int(agrees),
                        named="|".join(named), computed="|".join(comp),
                        regions="|".join(f"{k}:{v}" for k, v in regions.items()),
                        n_values=len(vals),
                        evidence=context(bypat[comp[0]], t) if comp
                        else (context(bypat[named[0]], t) if named else "")))

    with open(HERE / "audit_report.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)

    scored = [o for o in out if o["lexical_Q1_named"] != "NO_PDF"]
    a_comp = sum(o["agrees"] for o in scored)
    a_name = sum(o["lexical_Q1_named"] == o["author_Q1"] for o in scored)
    print(f"papers audited                 : {len(scored)} of {len(out)}")
    print("whole document scanned, nothing truncated")
    # Both rates are reported because neither test is the construct on its
    # own: `named` over-fires on bibliographies and review sections, `computed`
    # under-fires when the value is in a table. The spread between them is the
    # honest measure of how much of Q1 is NOT lexically determined.
    print(f"lexical(named)    == author Q1 : {a_name}/{len(scored)} "
          f"({a_name/len(scored):.0%})   over-fires on citations")
    print(f"lexical(computed) == author Q1 : {a_comp}/{len(scored)} "
          f"({a_comp/len(scored):.0%})   under-fires on table values")
    print(f"author Q1 distribution         : "
          f"{dict(Counter(o['author_Q1'] for o in scored))}")
    print(f"lexical named distribution     : "
          f"{dict(Counter(o['lexical_Q1_named'] for o in scored))}")
    print(f"lexical computed distribution  : "
          f"{dict(Counter(o['lexical_Q1_computed'] for o in scored))}")

    order = ["NIL_BUT_COEFFICIENT_COMPUTED", "NIL_BUT_METRIC_NAMED_POST_REFS",
             "METRIC_CODED_BUT_ABSENT", "ABS_WITH_NON_NIL_Q1",
             "METRIC_CODED_BUT_NOT_VALUE_ANCHORED",
             "NIL_BUT_METRIC_DISCUSSED_IN_BODY"]
    flags.sort(key=lambda f: order.index(f[1]) if f[1] in order else 99)
    print(f"\nFLAGS FOR HUMAN ADJUDICATION: {len(flags)}")
    print("(ordered by severity. NIL_BUT_METRIC_NAMED_POST_REFS ranks high "
          "because it is\n G6's shape: a finding reported in an appendix, "
          "which the old truncation could not see.)")
    for pid, kind, aq1, lex, region, ev in flags:
        print(f"  [{kind}] {pid}: author_Q1={aq1} lexical={lex}"
              + (f" region={region}" if region else ""))
        if ev:
            print(f"      ...{ev}...")


if __name__ == "__main__":
    main()
