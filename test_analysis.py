#!/usr/bin/env python3
"""
test_analysis.py - regression tests for compute_alpha.py and validate.py.

WHY THESE TESTS EXIST
---------------------
compute_alpha.py used to crash on any results file with more than one run
(`ValueError: Index contains duplicate entries`) and, worse, used prefix
matching to encode codes - so a cell whose "code" was the instrument's own
definition line echoed back ("INC = Q1=PA; OR correct metric family...") was
silently counted as a valid INC. Both defects are asserted against here.

validate.py is the gate that decides whether a results file may be used at
all, so the tests that matter are the ones proving it BLOCKS: drifted
instrument hashes, out-of-vocabulary codes, and a rater that returned one
constant code vector. A gate that only ever passes is not a gate.

One rule is asserted explicitly because it is a methodological commitment, not
an implementation detail: the grounding probe must never compare numbers.

Offline. The only results fixture is the committed results_fulltext.csv; the
small-sample and gate fixtures are derived from it at run time so the
repository does not have to carry extra results files whose only purpose is to
be test input.
"""

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable
FULL = "results_fulltext.csv"


def subset(n_papers: int, name: str) -> str:
    """Write the first n_papers of run 1 to `name` and return that name.

    compute_alpha.py warns about sampling noise below MIN_ITEMS, and
    validate.py has to be shown a file it accepts as well as files it
    rejects, so both need a small well-formed results file. Deriving it from
    the real one keeps the fixture honest and the repository clean.
    """
    rows = list(csv.DictReader(open(HERE / FULL, newline="")))
    keep, seen = [], []
    for r in rows:
        if r["run"] != "1":
            continue
        if r["paper_id"] not in seen:
            if len(seen) == n_papers:
                break
            seen.append(r["paper_id"])
        keep.append(r)
    with open(HERE / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(keep)
    return name

passed = failed = 0


def check(label: str, ok: bool, note: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {label}" + (f" - {note}" if note else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f" - {note}" if note else ""))


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as fh:
        tmp = fh.name
    try:
        return subprocess.run([PY, str(HERE / script), "--out", tmp, *args],
                              cwd=HERE, capture_output=True, text=True)
    finally:
        Path(tmp).unlink(missing_ok=True)


sys.path.insert(0, str(HERE))
import compute_alpha as ca  # noqa: E402
import validate as va       # noqa: E402

print("== 1. multi-run data no longer crashes ==")
MULTI = FULL
r = run("compute_alpha.py", "--results", MULTI)
check("a 5-run file is analysed without error", r.returncode == 0,
      (r.stderr.strip().splitlines() or [""])[-1][:90])
check("no pandas reshape error", "cannot reshape" not in r.stderr
      and "duplicate entries" not in r.stderr)
check("alpha is reported per run, not pooled",
      "computed separately for each run" in r.stdout)
check("all 5 runs appear in the table",
      all(f"\n      {i}  " in r.stdout for i in range(1, 6)))
check("runs are explicitly NOT treated as extra raters",
      "NOT treated as extra raters" in r.stdout)
check("within-rater instability is reported separately",
      "within-rater stability" in r.stdout,
      "it is not inter-rater disagreement")

print("\n== 2. codes are never coerced ==")
check("an exact code encodes", ca.encode("INC", "Q4") == 2.0)
# The defect: startswith("INC") accepted the instrument's own definition line.
echoed = "INC = Q1=PA; OR correct metric family but no threshold"
check("an echoed instrument line does NOT encode as a code",
      ca.encode(echoed, "Q4") != ca.encode(echoed, "Q4"),
      "must be NaN, not INC")
check("a lookalike code does not encode", ca.encode("OKAY", "Q4") !=
      ca.encode("OKAY", "Q4"))
check("case and padding are tolerated", ca.encode("  inc ", "Q4") == 2.0)
# The committed results file is clean, so an unrecognised code has to be
# injected to prove the report surfaces it rather than silently dropping it.
_oov = subset(3, "_oov_fixture.csv")
_rows = list(csv.DictReader(open(HERE / _oov, newline="")))
_rows[0]["Q4"] = "INC = Q1=PA; OR correct metric family without threshold"
with open(HERE / _oov, "w", newline="") as _fh:
    _w = csv.DictWriter(_fh, fieldnames=list(_rows[0].keys()))
    _w.writeheader(); _w.writerows(_rows)
_oov_out = run("compute_alpha.py", "--results", _oov).stdout
check("out-of-vocabulary values are counted in the report",
      "outside the vocabulary" in _oov_out and "never coerced" in _oov_out,
      "an echoed instrument line must be reported, not coerced to INC")
(HERE / "_oov_fixture.csv").unlink(missing_ok=True)

print("\n== 3. missing data is reported, never imputed ==")
import numpy as np  # noqa: E402
m = np.array([[0.0, 1.0, np.nan], [0.0, 1.0, 2.0]])
check("alpha tolerates a missing cell", ca.alpha_of(m, "Q4") is not None)
fk, blocked = ca.fleiss(m, "Q4")
check("Fleiss reports how many items it could not use", blocked == 1,
      f"blocked={blocked}")
check("Fleiss returns None rather than a silent nan with <3 raters",
      fk is None)
check("the census is printed", "cell census" in r.stdout)
check("a small sample is called out rather than reported",
      "sampling noise" in run("compute_alpha.py", "--results",
                              subset(3, "_small_fixture.csv")).stdout,
      "alpha on 3 items must not be presented as a result")
(HERE / "_small_fixture.csv").unlink(missing_ok=True)

print("\n== 4. the probe never compares numbers (the I3 commitment) ==")
check("digits are stripped by norm()", not any(c.isdigit() for c in
      va.norm("Fleiss kappa = 0.68 across 5 raters")),
      va.norm("Fleiss kappa = 0.68 across 5 raters"))
check("two sentences differing only in their numbers normalise identically",
      va.norm("alpha of 0.585 over 41 papers")
      == va.norm("alpha of 0.902 over 72 papers"),
      "so no numeric comparison is even expressible")
src = (HERE / "validate.py").read_text()
check("Q4 is excluded from the probe", va.PROBED == ("Q1", "Q2", "Q3"),
      "Q4 is the rater's own justification, not a quotation")
check("the I3 rationale is recorded in the source", "I3-kunilovskaya" in src)

print("\n== 5. grounding behaves ==")
paper = va.norm("we conducted an inter-rater reliability analysis using "
                "Fleiss Kappa a standard metric for evaluating multi rater "
                "categorical agreement across eight segments").split()
pl, ps = va.shingles(paper, va.LONG_SHINGLE), va.shingles(paper, va.SHORT_SHINGLE)
cases = [
    ("a verbatim quote is GROUNDED",
     "we conducted an inter-rater reliability analysis using Fleiss Kappa a "
     "standard metric", "GROUNDED"),
    ("the same quote with different numbers is still GROUNDED",
     "we conducted an inter-rater reliability analysis using Fleiss Kappa a "
     "standard metric for 999 raters", "GROUNDED"),
    ("invented evidence is NOT_FOUND",
     "the authors computed Cohen kappa on a fixed pair of expert annotators "
     "throughout the study", "NOT_FOUND"),
    ("a too-short quote is NO_QUOTE", "see above", "NO_QUOTE"),
]
for label, quote, want in cases:
    got = va.ground(quote, paper, pl, ps)
    check(label, got == want, f"got {got}, wanted {want}")

print("\n== 6. the gate blocks ==")
def gate_on(rows: list[dict], fields: list[str]) -> int:
    d = Path(tempfile.mkdtemp())
    f = HERE / "_gate_fixture.csv"
    try:
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
        return run("validate.py", "--results", f.name).returncode
    finally:
        f.unlink(missing_ok=True)

SMALL = subset(3, "_gate_base.csv")
base = list(csv.DictReader(open(HERE / SMALL, newline="")))
fields = list(base[0].keys())
check("a clean hardened file passes",
      run("validate.py", "--results", SMALL).returncode == 0)

drift = [dict(r) for r in base]
drift[0]["instrument_sha"] = "0000000000000000"
check("mixed instrument hashes block the file", gate_on(drift, fields) == 1)

badcode = [dict(r) for r in base]
badcode[0]["Q4"] = "INC = Q1=PA; OR correct metric family"
check("an out-of-vocabulary code blocks the file",
      gate_on(badcode, fields) == 1)

degen = [dict(r) for r in base if r["model"] == "gemma"]
for r in degen:
    r["Q1"], r["Q2"], r["Q3"], r["Q4"] = "NIL", "UNK", "UNK", "ABS"
check("a constant code vector blocks the file", gate_on(degen, fields) == 1)

stale = [dict(r) for r in base]
stale[0]["stimulus_sha"] = "ffffffffffffffff"
check("a stimulus_sha mismatch blocks the file", gate_on(stale, fields) == 1)

(HERE / "_gate_base.csv").unlink(missing_ok=True)

print("\n" + "=" * 58)
print(f"{passed} passed, {failed} failed")
print("=" * 58)
sys.exit(1 if failed else 0)
