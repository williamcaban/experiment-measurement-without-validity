#!/usr/bin/env python3
"""
validate.py - the probe gate. Run this on a results file before trusting it.

WHAT IT CHECKS, AND WHY EACH ONE EARNS ITS PLACE
------------------------------------------------
BINDING. Every row must carry the same instrument_sha, and each row's
stimulus_sha must match the paper it claims to code. Without this a results
file can silently mix codings made on two different stimuli or two different
instrument wordings, which is how a stale 60-row file once passed as complete.

VOCABULARY. Codes outside a question's vocabulary are listed, not coerced. A
pre-hardening results file contains cells whose "code" is the instrument's own
definition line echoed back; prefix matching accepted those as real codes.

INSTRUMENT CONFORMANCE. The instrument states "If Q1=NIL then Q4=ABS (no
exceptions)". A rater that breaks the rule it was given is evidence about
instrument-following, and that is a finding this paper is about, so the rate is
measured per rater rather than quietly repaired.

DEGENERACY. A rater that returns the same four codes for every paper has not
coded anything. Agreement statistics cannot distinguish that from competence,
so it is checked directly.

GROUNDING (the probe). For Q1-Q3 the instrument demands "<one sentence from the
full text>", so the quoted sentence should be findable in the paper. This
catches a rater inventing its evidence.

WHAT THE PROBE DELIBERATELY DOES NOT DO
---------------------------------------
It does not check numbers. Digits are stripped from both sides before matching
and no numeric comparison is ever reported. The reason is I3-kunilovskaya: its
list of coefficients is its own annotation CODEBOOK, not a set of results, and
the alphas it quotes appear in several roles across the paper. A probe that
verified "does the number in the rationale appear as a reported value" would
flag correct codings there and, worse, would invite resolving such flags by
re-reading the number rather than the construct. Numbers in a rationale are not
the evidence the instrument asked for; the sentence is.

Q4 is also not probed. The instrument asks for "<one sentence of
justification>" there, which is the rater's own reasoning and is not supposed
to appear in the paper.

EXIT STATUS
  0  no hard failures (binding, vocabulary, degeneracy all clean)
  1  at least one hard failure; grounding rates are reported, never fatal,
     because extraction normalisation makes exact quotation imperfect

USAGE
  python3 validate.py [--results results.csv] [--stimulus stimulus_fulltext.json]
                      [--out validation_report.txt] [--sample N]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent

VOCAB = {
    "Q1": {"KCO", "KFL", "KAL", "ICC", "PA", "NIL"},
    "Q2": {"F2", "FN", "RV", "CS", "UNK"},
    "Q3": {"BIN", "NOM", "ORD", "CON", "UNK"},
    "Q4": {"OK", "MM", "INC", "ABS"},
}
PROBED = ("Q1", "Q2", "Q3")        # Q4 is justification, not quotation
LONG_SHINGLE, SHORT_SHINGLE = 8, 5


def norm(s: str) -> str:
    """Lowercase, de-punctuate, DROP ALL DIGITS, collapse whitespace.

    Digits go before any comparison is possible, so no code path downstream can
    compare a number even by accident. See the module docstring.
    """
    s = s.lower().replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"[^a-z' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def shingles(words: list[str], n: int) -> set[str]:
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def rationales(raw: str) -> dict[str, str]:
    """Pull the text after '|' on each Q line of a stored reply."""
    out: dict[str, str] = {}
    for line in raw.replace(" // ", "\n").splitlines():
        flat = re.sub(r"[*`#_]+", " ", line)
        m = re.match(r"^\s*(Q[1-4])\s*[:\-–]\s*(.+)$", flat)
        if not m:
            continue
        rest = m.group(2)
        if "|" in rest:
            out.setdefault(m.group(1), rest.split("|", 1)[1].strip())
    return out


def ground(quote: str, paper_words: list[str], paper_long: set[str],
           paper_short: set[str]) -> str:
    """GROUNDED / PARTIAL / NOT_FOUND / NO_QUOTE for one quoted sentence."""
    q = norm(quote)
    qw = q.split()
    if len(qw) < 4:
        return "NO_QUOTE"
    if len(qw) >= LONG_SHINGLE and shingles(qw, LONG_SHINGLE) & paper_long:
        return "GROUNDED"
    if len(qw) >= SHORT_SHINGLE and shingles(qw, SHORT_SHINGLE) & paper_short:
        return "PARTIAL"
    return "NOT_FOUND"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_fulltext.csv")
    ap.add_argument("--stimulus", default="stimulus_fulltext.json")
    ap.add_argument("--out", default="validation_report_fulltext.txt")
    ap.add_argument("--sample", type=int, default=0,
                    help="probe only the first N rows (grounding is the slow part)")
    a = ap.parse_args()

    res = HERE / a.results
    if not res.exists():
        sys.exit(f"FATAL: {res} not found")
    rows = list(csv.DictReader(open(res, newline="")))
    if not rows:
        sys.exit(f"FATAL: {res.name} has no rows")

    stim_path = HERE / a.stimulus
    stim = {}
    if stim_path.exists():
        stim = {r["paper_id"]: r for r in json.loads(stim_path.read_text())}

    L: list[str] = []
    def out(s: str = "") -> None:
        L.append(s)

    hard = 0
    out("=" * 68)
    out("PROBE GATE - results validation")
    out("=" * 68)
    out(f"results   : {a.results}  ({len(rows)} rows)")
    out(f"stimulus  : {a.stimulus}"
        + ("" if stim else "  (ABSENT - binding and grounding skipped)"))
    out("")

    # ---- binding -------------------------------------------------------- #
    out("-- binding ----------------------------------------------------------")
    shas = Counter(r.get("instrument_sha", "") for r in rows)
    if "" in shas and len(shas) == 1:
        out("  instrument_sha : absent from this file (predates binding)")
    elif len(shas) == 1:
        out(f"  instrument_sha : {next(iter(shas))} (uniform)")
    else:
        hard += 1
        out(f"  ! instrument_sha : {len(shas)} DIFFERENT values in one file")
        for s, n in shas.most_common():
            out(f"      {s or '(empty)'} x{n}")
        out("    rows coded under different instrument wordings are not")
        out("    comparable and must not be pooled")

    if stim and rows[0].get("stimulus_sha"):
        want = {p: hashlib.sha256(r["text"].encode()).hexdigest()[:16]
                for p, r in stim.items()}
        bad = [r for r in rows
               if r["paper_id"] in want
               and r["stimulus_sha"] != want[r["paper_id"]]]
        absent = sorted({r["paper_id"] for r in rows if r["paper_id"] not in want})
        if bad:
            hard += 1
            out(f"  ! stimulus_sha mismatch on {len(bad)} row(s): coded against "
                f"a different text than {a.stimulus}")
            for r in bad[:5]:
                out(f"      {r['paper_id']} {r['model']} run {r.get('run','?')}")
        else:
            out(f"  stimulus_sha   : all {len(rows) - len(bad)} rows match "
                f"{a.stimulus}")
        if absent:
            out(f"  papers not in the stimulus file: {', '.join(absent[:8])}"
                + (" ..." if len(absent) > 8 else ""))
    out("")

    # ---- vocabulary ----------------------------------------------------- #
    out("-- code vocabulary --------------------------------------------------")
    offenders = defaultdict(Counter)
    for r in rows:
        for q in VOCAB:
            v = (r.get(q) or "").strip().upper()
            if v not in VOCAB[q]:
                offenders[q][v] += 1
    if offenders:
        hard += 1
        for q, c in sorted(offenders.items()):
            out(f"  ! {q}: {sum(c.values())} value(s) outside the vocabulary")
            for v, n in c.most_common(4):
                out(f"      {v[:72]!r} x{n}")
        out("    these are NOT coerced to the nearest code; prefix matching is")
        out("    what let an echoed instrument line count as a real code")
    else:
        out(f"  all {len(rows) * 4} codes are in vocabulary")
    out("")

    # ---- instrument conformance ----------------------------------------- #
    out("-- instrument conformance: 'if Q1=NIL then Q4=ABS (no exceptions)' --")
    for mdl in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == mdl]
        nil = [r for r in sub if (r.get("Q1") or "").upper() == "NIL"]
        viol = [r for r in nil if (r.get("Q4") or "").upper() != "ABS"]
        abs_not_nil = [r for r in sub
                       if (r.get("Q4") or "").upper() == "ABS"
                       and (r.get("Q1") or "").upper() != "NIL"]
        rate = f"{100 * len(viol) / len(nil):.0f}%" if nil else "n/a"
        out(f"  {mdl:10} Q1=NIL on {len(nil):3}  rule broken {len(viol):3} "
            f"({rate:>4})   Q4=ABS without Q1=NIL {len(abs_not_nil):3}")
    out("  reported, not repaired: breaking a stated rule is a measurement of")
    out("  instrument-following, which is part of what this paper argues")
    out("")

    # ---- degeneracy ----------------------------------------------------- #
    out("-- degeneracy -------------------------------------------------------")
    for mdl in sorted({r["model"] for r in rows}):
        for run in sorted({r.get("run", "1") for r in rows if r["model"] == mdl}):
            sub = [r for r in rows
                   if r["model"] == mdl and r.get("run", "1") == run]
            if len(sub) < 3:
                continue
            pats = Counter(tuple((r.get(q) or "") for q in ("Q1", "Q2", "Q3", "Q4"))
                           for r in sub)
            top, n = pats.most_common(1)[0]
            if n == len(sub):
                hard += 1
                out(f"  ! {mdl} run {run}: ALL {n} papers got the identical "
                    f"code vector {top} - this rater coded nothing")
            elif n / len(sub) > 0.8:
                out(f"  {mdl} run {run}: {n}/{len(sub)} papers share one code "
                    f"vector {top} - near-degenerate, interpret with care")
    if not any("!" in x for x in L[-6:]):
        out("  no rater returned a constant code vector")
    out("")

    # ---- grounding probe ------------------------------------------------- #
    out("-- grounding probe (Q1-Q3 quoted sentences) -------------------------")
    out("   digits are stripped from both sides before matching and no numeric")
    out("   comparison is made anywhere: see the module docstring on I3.")
    out("")
    if not stim:
        out("   skipped: no stimulus file to probe against")
    elif not rows[0].get("raw"):
        out("   skipped: this results file has no 'raw' column")
    else:
        cache: dict[str, tuple] = {}
        tally = defaultdict(Counter)
        examples: list[str] = []
        probe_rows = rows[:a.sample] if a.sample else rows
        for r in probe_rows:
            pid = r["paper_id"]
            if pid not in stim:
                continue
            if pid not in cache:
                w = norm(stim[pid]["text"]).split()
                cache[pid] = (w, shingles(w, LONG_SHINGLE),
                              shingles(w, SHORT_SHINGLE))
            pw, pl, ps = cache[pid]
            rats = rationales(r.get("raw", ""))
            for q in PROBED:
                verdict = ground(rats.get(q, ""), pw, pl, ps) \
                    if q in rats else "NO_QUOTE"
                tally[q][verdict] += 1
                if verdict == "NOT_FOUND" and len(examples) < 5:
                    examples.append(
                        f"      {pid} {r['model']} {q}: "
                        f"{rats.get(q, '')[:90]!r}")
        hdr = (f"   {'Q':>3}  {'GROUNDED':>9} {'PARTIAL':>8} {'NOT_FOUND':>10} "
               f"{'NO_QUOTE':>9}")
        out(hdr)
        out("   " + "-" * (len(hdr) - 3))
        for q in PROBED:
            t = tally[q]
            n = sum(t.values()) or 1
            out(f"   {q:>3}  {t['GROUNDED']:>6} {100*t['GROUNDED']/n:3.0f}% "
                f"{t['PARTIAL']:>7} {t['NOT_FOUND']:>10} {t['NO_QUOTE']:>9}")
        out("")
        out("   GROUNDED = an 8-word run of the quote appears in the paper")
        out("   PARTIAL  = only a 5-word run matches (paraphrase or re-wrap)")
        out("   NOT_FOUND= no 5-word run matches; candidate invented evidence")
        if examples:
            out("")
            out("   NOT_FOUND examples:")
            for e in examples:
                out(e)
        out("")
        out("   Q4 is not probed: the instrument asks there for the rater's own")
        out("   justification, which is not supposed to appear in the paper.")
    out("")

    out("=" * 68)
    out(f"HARD FAILURES: {hard}")
    if hard:
        out("binding, vocabulary and degeneracy failures block use of this file.")
    else:
        out("no hard failures. Grounding rates above are diagnostic, not a gate.")
    out("=" * 68)

    report = "\n".join(L)
    (HERE / a.out).write_text(report + "\n")
    print(report)
    print(f"\nwritten to {a.out}")
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
