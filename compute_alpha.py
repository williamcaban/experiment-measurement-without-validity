#!/usr/bin/env python3
"""
compute_alpha.py - inter-rater reliability for the second-rater experiment.

WHAT CHANGED AND WHY
--------------------
The previous version pivoted results on (paper_id x model) and so raised
`ValueError: Index contains duplicate entries, cannot reshape` the moment a
results file contained more than one run. It also hardcoded results.csv,
treated a missing cell and an unparseable one identically, and emitted a
hardcoded prose paragraph describing a 20-paper excerpt stimulus.

Three principles now govern this script.

1. RUNS ARE NOT EXTRA RATERS. Pooling five runs of three models into fifteen
   columns would mix two different quantities: disagreement BETWEEN raters and
   instability WITHIN one rater. They answer different questions and the paper
   needs them apart. So alpha is computed once per run, across raters, and
   reported as a distribution over runs; within-rater instability is measured
   separately.

2. MISSING DATA IS REPORTED, NEVER IMPUTED. A cell the runner deferred is
   missing, not a code. Krippendorff's alpha tolerates missing cells by design;
   Fleiss' kappa does not, so it is reported only on a complete matrix and
   otherwise says how many items were incomplete. Every count is printed, so a
   reader can see what the estimate rests on.

3. AN UNRECOGNISED CODE IS A FINDING, NOT A ROUNDING ERROR. Values outside a
   question's vocabulary are counted and listed rather than coerced.

The headline estimate is the per-run alpha, median with range, because that is
the reliability a practitioner coding once would actually experience. A
modal-consensus alpha over runs is reported alongside, clearly labelled, to
show what repeated sampling buys.

USAGE
  python3 compute_alpha.py [--results results.csv] [--author author_codes.csv]
                           [--out alpha_report.txt] [--question Q4]
                           [--collapse]
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import krippendorff
import numpy as np

HERE = Path(__file__).parent

VOCAB = {
    "Q1": ["KCO", "KFL", "KAL", "ICC", "PA", "NIL"],
    "Q2": ["F2", "FN", "RV", "CS", "UNK"],
    "Q3": ["BIN", "NOM", "ORD", "CON", "UNK"],
    "Q4": ["OK", "MM", "INC", "ABS"],
}
MODELS = ["nemotron", "gemma", "qwen"]
THRESHOLD = 0.667   # tentative floor; 0.800 is Krippendorff's "reliable"
MIN_ITEMS = 20      # below this an alpha is noise; the report says so

# The paper also quotes a binary form of Q4, correct use versus misuse, because
# that is the distinction its headline misuse rate rests on. Alpha on four codes
# and alpha on two are different quantities, so the collapsed form is a separate
# question with its own vocabulary rather than an overwrite of Q4.
COLLAPSE = {
    "Q4": {"OK": "OK", "MM": "MIS", "INC": "MIS", "ABS": "MIS"},
}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load_results(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"FATAL: {path} not found")
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise SystemExit(f"FATAL: {path.name} has no rows")
    for need in ("paper_id", "model", "Q1", "Q2", "Q3", "Q4"):
        if need not in rows[0]:
            raise SystemExit(f"FATAL: {path.name} lacks a {need!r} column")
    # A results file written before runs were tracked has no `run` column.
    # Defaulting it to 1 is what previously let a stale single-run file pass as
    # a complete multi-run one, so say so loudly instead.
    if "run" not in rows[0]:
        print(f"NOTE: {path.name} has no 'run' column; treating it as a single "
              f"run. This file predates run tracking.")
        for r in rows:
            r["run"] = "1"
    return rows


def load_author(path: Path) -> dict[str, dict[str, str]]:
    """paper_id -> {Q1..Q4} from the author's codes, whatever the column names."""
    if not path.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(path, newline="")):
        pid = r.get("paper_id")
        if not pid:
            continue
        out[pid] = {q: (r.get(f"author_{q}") or r.get(q) or "").strip()
                    for q in VOCAB}
    return out


def load_failures(path: Path) -> Counter:
    if not path.exists():
        return Counter()
    return Counter(r.get("reason", "?")
                   for r in csv.DictReader(open(path, newline="")))


# --------------------------------------------------------------------------- #
# reliability
# --------------------------------------------------------------------------- #

def encode(value: str, question: str) -> float:
    """Code -> integer, or NaN. Exact match only; no prefix coercion."""
    v = (value or "").strip().upper()
    vocab = VOCAB[question]
    return float(vocab.index(v)) if v in vocab else float("nan")


def alpha_of(matrix: np.ndarray, question: str) -> float | None:
    """Krippendorff's alpha over a raters x items matrix, NaN = missing."""
    if matrix.size == 0 or np.isnan(matrix).all():
        return None
    # Alpha is undefined when every rater gave every item the same code: there
    # is no observed variance to normalise against. That is perfect agreement,
    # not an error, so it is reported as such rather than as a number.
    finite = matrix[~np.isnan(matrix)]
    if finite.size and len(set(finite.tolist())) == 1:
        return float("inf")
    try:
        return float(krippendorff.alpha(
            reliability_data=matrix,
            level_of_measurement="nominal",
            value_domain=list(range(len(VOCAB[question])))))
    except Exception as exc:                      # noqa: BLE001
        print(f"  (alpha not computable: {type(exc).__name__}: {exc})")
        return None


def fleiss(matrix: np.ndarray, question: str) -> tuple[float | None, int]:
    """Fleiss' kappa, plus the number of items that blocked it.

    Rule R3 of the paper prescribes Fleiss' kappa for >=3 raters on a complete
    nominal matrix. Incompleteness is the common case here, so the count of
    incomplete items is returned with the result instead of a bare nan.
    """
    items = matrix.T                                  # items x raters
    complete = items[~np.isnan(items).any(axis=1)]
    blocked = len(items) - len(complete)
    if len(complete) == 0:
        return None, blocked
    n_raters = complete.shape[1]
    if n_raters < 3:
        return None, blocked
    cats = range(len(VOCAB[question]))
    counts = np.array([[np.sum(row == c) for c in cats] for row in complete])
    p_i = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_j = counts.sum(axis=0) / counts.sum()
    p_bar, p_e = p_i.mean(), (p_j ** 2).sum()
    if p_e == 1:
        return float("inf"), blocked
    return float((p_bar - p_e) / (1 - p_e)), blocked


def install_collapse(question: str, rows: list[dict],
                     author: dict[str, dict[str, str]]) -> str:
    """Register a derived two-value question and rewrite the codes into it.

    Done at the source so nothing downstream needs a special case: encode(),
    alpha_of() and fleiss() read VOCAB[question] exactly as they always do, and
    the report header names the derived question, so a reader can tell from the
    report alone which vocabulary the number was computed on.

    A code outside the source vocabulary passes through unmapped rather than
    becoming blank, so it is still counted and listed as out-of-vocabulary
    instead of being quietly reclassified as missing.
    """
    if question not in COLLAPSE:
        raise SystemExit(f"FATAL: --collapse is defined for "
                         f"{', '.join(sorted(COLLAPSE))} only, not {question}")
    mapping = COLLAPSE[question]
    uncovered = [c for c in VOCAB[question] if c not in mapping]
    if uncovered:
        raise SystemExit(f"FATAL: the {question} collapse map does not cover "
                         f"{uncovered}; every code must map somewhere or the "
                         f"alpha would silently rest on fewer cells")
    derived = f"{question}-binary"
    VOCAB[derived] = list(dict.fromkeys(mapping.values()))
    for r in rows:
        src = r.get(question, "")
        r[derived] = mapping.get(src, src)
    for codes in author.values():
        src = codes.get(question, "")
        codes[derived] = mapping.get(src, src)
    return derived


def build_matrix(codes: dict, papers: list[str], raters: list[str],
                 question: str) -> np.ndarray:
    return np.array([[encode(codes.get((p, r), ""), question) for p in papers]
                     for r in raters], dtype=float)


def fmt(a: float | None) -> str:
    if a is None:
        return "  n/a"
    if a == float("inf"):
        return "1.000*"          # perfect agreement, no variance to normalise
    return f"{a:6.3f}"


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_fulltext.csv")
    ap.add_argument("--author", default="author_codes_55.csv")
    ap.add_argument("--out", default="alpha_report.txt")
    ap.add_argument("--question", default="Q4", choices=sorted(VOCAB))
    ap.add_argument("--collapse", action="store_true",
                    help="merge the question's codes into the two-value form "
                         "the paper quotes as its headline figure (Q4: OK vs "
                         "MIS); reported as a separate question, not as Q4")
    a = ap.parse_args()

    res_path = HERE / a.results
    rows = load_results(res_path)
    author = load_author(HERE / a.author)
    failures = load_failures(Path(str(res_path) + ".failures.csv"))

    # Must follow load_author, which populates one key per question in VOCAB.
    if a.collapse:
        a.question = install_collapse(a.question, rows, author)

    runs = sorted({r["run"] for r in rows}, key=lambda s: (len(s), s))
    papers = sorted({r["paper_id"] for r in rows})
    models = [m for m in MODELS if m in {r["model"] for r in rows}]
    raters = (["author"] if author else []) + models

    # codes[(run, paper, rater)] = code
    codes: dict[tuple[str, str, str], str] = {}
    dupes, unknown = [], Counter()
    for r in rows:
        key = (r["run"], r["paper_id"], r["model"])
        if key in codes:
            dupes.append(key)
        codes[key] = r[a.question]
        if encode(r[a.question], a.question) != encode(r[a.question], a.question):
            unknown[r[a.question]] += 1

    L = []
    def out(s: str = "") -> None:
        L.append(s)

    out("=" * 68)
    out("SECOND-RATER IRR EXPERIMENT - RELIABILITY REPORT")
    out("=" * 68)
    out(f"results file   : {a.results}")
    out(f"question       : {a.question}  (vocabulary: "
        f"{', '.join(VOCAB[a.question])})")
    out(f"papers         : {len(papers)}")
    out(f"model raters   : {', '.join(models)}")
    out(f"author rater   : {'yes' if author else 'NO - author_codes.csv absent'}")
    out(f"runs           : {len(runs)}  ({', '.join(runs)})")
    out("")

    # ---- cell census ----------------------------------------------------- #
    out("-- cell census ------------------------------------------------------")
    expected = len(papers) * len(models) * len(runs)
    present = sum(1 for k in codes if k[2] in models)
    out(f"  expected cells (papers x models x runs) : {expected}")
    out(f"  present in results                      : {present}")
    out(f"  missing                                 : {expected - present}")
    if dupes:
        out(f"  ! DUPLICATE cells                       : {len(dupes)}")
        for d in dupes[:5]:
            out(f"      run {d[0]} {d[1]} {d[2]}")
        out("    a duplicate means one cell was coded twice in one run; the "
            "last row won")
    if unknown:
        out(f"  ! codes outside the vocabulary          : {sum(unknown.values())}")
        for v, n in unknown.most_common():
            out(f"      {v!r} x{n}  -> treated as MISSING, never coerced")
    if failures:
        out(f"  deferred cells in the failure ledger    : {sum(failures.values())}")
        for reason, n in failures.most_common():
            out(f"      {reason:12} {n}")
    if expected != present:
        out("  NOTE: missing cells are left missing. Krippendorff's alpha")
        out("        tolerates them; Fleiss' kappa does not and will say so.")
    out("")

    # ---- between-rater agreement, per run -------------------------------- #
    out("-- between-rater agreement, computed separately for each run --------")
    out("   (runs are NOT treated as extra raters: that would mix disagreement")
    out("    between raters with instability within one rater)")
    out("")
    hdr = f"   {'run':>4}  {'alpha':>7}  {'Fleiss k':>9}  {'blocked':>7}  {'n items':>7}"
    out(hdr)
    out("   " + "-" * (len(hdr) - 3))
    per_run = {}
    for run in runs:
        c = {(p, r): codes.get((run, p, r), author.get(p, {}).get(a.question, ""))
             if r == "author" else codes.get((run, p, r), "")
             for p in papers for r in raters}
        m = build_matrix(c, papers, raters, a.question)
        al = alpha_of(m, a.question)
        fk, blocked = fleiss(m, a.question)
        per_run[run] = al
        out(f"   {run:>4}  {fmt(al)}  {fmt(fk):>9}  {blocked:>7}  {len(papers):>7}")
    out("")

    numeric = [v for v in per_run.values() if v is not None and v != float("inf")]
    if numeric:
        out(f"   HEADLINE  median alpha over {len(numeric)} run(s) = "
            f"{median(numeric):.3f}   [min {min(numeric):.3f}, "
            f"max {max(numeric):.3f}]")
        out(f"   tentative floor {THRESHOLD:.3f} / reliable 0.800 -> "
            f"{'MEETS floor' if median(numeric) >= THRESHOLD else 'BELOW floor'}")
        out("   This is the reliability a practitioner coding once would get.")
        # An alpha over a handful of items is dominated by sampling noise, and
        # printing "MEETS floor" next to it invites exactly the over-reading
        # this paper argues against. Say so in the report, not in a footnote.
        if len(papers) < MIN_ITEMS:
            out("")
            out(f"   !! only {len(papers)} items: an alpha this small is "
                f"sampling noise.")
            out(f"      Do not report it. {MIN_ITEMS}+ items are needed for the "
                f"estimate to mean anything,")
            out("      and the verdict line above should be ignored.")
    out("")

    # ---- pairwise, per run ----------------------------------------------- #
    out("-- pairwise alpha ---------------------------------------------------")
    pair_vals = defaultdict(list)
    for run in runs:
        c = {(p, r): (author.get(p, {}).get(a.question, "") if r == "author"
                      else codes.get((run, p, r), ""))
             for p in papers for r in raters}
        for i, r1 in enumerate(raters):
            for r2 in raters[i + 1:]:
                m = build_matrix(c, papers, [r1, r2], a.question)
                v = alpha_of(m, a.question)
                if v is not None and v != float("inf"):
                    pair_vals[f"{r1} / {r2}"].append(v)
    for pair, vals in sorted(pair_vals.items()):
        out(f"   {pair:24} median {median(vals):6.3f}  "
            f"[{min(vals):6.3f}, {max(vals):6.3f}]  n_runs={len(vals)}")
    out("")

    # ---- within-rater stability ------------------------------------------ #
    if len(runs) > 1:
        out("-- within-rater stability across runs -------------------------------")
        out("   the share of papers where a model gave the SAME code every run.")
        out("   This is rater non-determinism, and it is not inter-rater")
        out("   disagreement; the two are reported apart on purpose.")
        out("")
        for mdl in models:
            stable = unstable = incomplete = 0
            for p in papers:
                vals = [codes.get((run, p, mdl), "") for run in runs]
                got = [v for v in vals if encode(v, a.question) == encode(v, a.question)]
                if len(got) < len(runs):
                    incomplete += 1
                elif len(set(got)) == 1:
                    stable += 1
                else:
                    unstable += 1
            scored = stable + unstable
            pct = f"{100 * stable / scored:.0f}%" if scored else "n/a"
            out(f"   {mdl:10} identical on {stable:3}/{scored:<3} papers ({pct:>4})"
                f"   varied {unstable:3}   incomplete {incomplete:3}")
        out("")

        # ---- modal-consensus secondary ----------------------------------- #
        cons = {}
        for p in papers:
            for mdl in models:
                vals = [codes.get((run, p, mdl), "") for run in runs]
                got = [v for v in vals
                       if encode(v, a.question) == encode(v, a.question)]
                if got:
                    cons[(p, mdl)] = Counter(got).most_common(1)[0][0]
        for p in papers:
            cons[(p, "author")] = author.get(p, {}).get(a.question, "")
        m = build_matrix(cons, papers, raters, a.question)
        al = alpha_of(m, a.question)
        fk, blocked = fleiss(m, a.question)
        out("-- SECONDARY: alpha on each rater's modal code across runs ----------")
        out(f"   alpha {fmt(al)}   Fleiss k {fmt(fk)} (blocked {blocked})")
        out("   Shows what repeated sampling buys. It is NOT the headline: no")
        out("   practitioner codes a paper five times and takes the mode.")
        out("")

    # ---- prevalence ------------------------------------------------------ #
    out("-- prevalence -------------------------------------------------------")
    if author:
        dist = Counter(author[p].get(a.question, "?") for p in papers if p in author)
        n = sum(dist.values())
        out(f"   author ({n} papers):")
        for code in VOCAB[a.question]:
            k = dist.get(code, 0)
            out(f"     {code:4} {k:3}  {100 * k / n:5.1f}%" if n else f"     {code:4}   0")
    for mdl in models:
        dist = Counter(codes[k] for k in codes if k[2] == mdl)
        n = sum(dist.values())
        if not n:
            continue
        out(f"   {mdl} ({n} codings over {len(runs)} run(s)):")
        for code in VOCAB[a.question]:
            k = dist.get(code, 0)
            out(f"     {code:4} {k:3}  {100 * k / n:5.1f}%")
    out("")

    # ---- serving provenance --------------------------------------------- #
    provs = Counter((r["model"], r.get("provider", "")) for r in rows
                    if r.get("provider"))
    if provs:
        out("-- serving provenance ----------------------------------------------")
        out("   which provider answered. A model served at two quantizations is")
        out("   two functions, so routing spread belongs in the record.")
        quants = Counter((r["model"], r.get("pinned_quantizations", ""))
                         for r in rows if r.get("pinned_quantizations"))
        for mdl in models:
            ps = {p: n for (m, p), n in provs.items() if m == mdl}
            qs = {q for (m, q) in quants if m == mdl}
            out(f"   {mdl:10} providers: "
                + ", ".join(f"{p}={n}" for p, n in sorted(ps.items())))
            if qs:
                out(f"   {'':10} pinned quantization: {', '.join(sorted(qs))}")
        out("")
    else:
        out("-- serving provenance ----------------------------------------------")
        out("   no provider column: this results file predates provenance")
        out("   recording, so routing variance cannot be separated from model")
        out("   non-determinism in it.")
        out("")

    out("=" * 68)
    report = "\n".join(L)
    (HERE / a.out).write_text(report + "\n")
    print(report)
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
