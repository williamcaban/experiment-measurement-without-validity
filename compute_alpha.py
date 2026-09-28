#!/usr/bin/env python3
"""
compute_alpha.py — Krippendorff's alpha for the second-rater experiment.

Usage:
  pip install krippendorff pandas
  python compute_alpha.py

Input:  results.csv   (from second_rater.py)
        author_codes.csv  (hand-coded author Q4 values)
Output: alpha_report.txt  (4-way α + all pairwise αs + prevalence table)
"""

from pathlib import Path
from collections import Counter

import numpy as np
import krippendorff
import pandas as pd

RESULTS_FILE     = Path(__file__).parent / "results.csv"
AUTHOR_FILE      = Path(__file__).parent / "author_codes.csv"
REPORT_FILE      = Path(__file__).parent / "alpha_report.txt"

# Encode Q4 nominal codes as integers
Q4_ENC = {"OK": 0, "MM": 1, "INC": 2, "ABS": 3}
Q4_DEC = {v: k for k, v in Q4_ENC.items()}

def load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS_FILE)

def load_author() -> pd.Series:
    df = pd.read_csv(AUTHOR_FILE)
    return df.set_index("paper_id")["author_Q4"]

def encode(val: str) -> float:
    val = str(val).strip().upper()
    for k in Q4_ENC:
        if val.startswith(k):
            return float(Q4_ENC[k])
    return np.nan

def compute_alpha_4way(pivot: pd.DataFrame, raters: list[str]) -> float:
    """Compute Krippendorff alpha across all raters."""
    data = pivot[raters].T.values.astype(float)
    return krippendorff.alpha(
        reliability_data=data,
        level_of_measurement="nominal",
        value_domain=list(Q4_ENC.values())
    )

def compute_fleiss_kappa(pivot: pd.DataFrame, raters: list[str]) -> float:
    """Fleiss' kappa for a complete matrix (rows=items, cols=raters), nominal.
    Rule R3 of the paper prescribes Fleiss' kappa for >=3 raters, complete
    matrix, nominal scale; it is reported alongside Krippendorff's alpha.
    Returns nan if any rating is missing."""
    data = pivot[raters].values.astype(float)
    if np.isnan(data).any():
        return float("nan")
    cats = sorted(Q4_ENC.values())
    counts = np.array([[np.sum(row == c) for c in cats] for row in data])
    n_raters = data.shape[1]
    p_i = ((counts ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / counts.sum()
    p_e = (p_j ** 2).sum()
    return float((p_bar - p_e) / (1 - p_e))

def compute_alpha_pair(a: np.ndarray, b: np.ndarray) -> float:
    data = np.array([a, b], dtype=float)
    return krippendorff.alpha(
        reliability_data=data,
        level_of_measurement="nominal",
        value_domain=list(Q4_ENC.values())
    )

def main():
    df      = load_results()
    author  = load_author()

    # Pivot: rows=paper_id, cols=model, values=Q4 encoded
    pivot = df.pivot(index="paper_id", columns="model", values="Q4")
    for col in pivot.columns:
        pivot[col] = pivot[col].apply(encode)

    # Add author codes
    pivot["author"] = author.apply(encode)

    raters = ["author", "nemotron", "gemma", "qwen"]
    missing = [r for r in raters if r not in pivot.columns]
    if missing:
        print(f"WARNING: missing raters in results: {missing}")
        raters = [r for r in raters if r in pivot.columns]

    # ── 4-way alpha ──────────────────────────────────────────────────────────
    alpha_4way = compute_alpha_4way(pivot, raters)

    # ── Pairwise alphas ──────────────────────────────────────────────────────
    pairwise = {}
    for i, r1 in enumerate(raters):
        for r2 in raters[i+1:]:
            a1 = pivot[r1].values
            a2 = pivot[r2].values
            pairwise[f"{r1}↔{r2}"] = compute_alpha_pair(a1, a2)

    # ── Prevalence table ─────────────────────────────────────────────────────
    prev_rows = []
    for paper_id in pivot.index:
        row = {"paper_id": paper_id}
        for r in raters:
            val = pivot.at[paper_id, r]
            row[r] = Q4_DEC.get(int(val), "?") if not np.isnan(val) else "?"
        # Majority vote
        votes = [row[r] for r in raters if row[r] != "?"]
        cnt   = Counter(votes)
        row["majority"] = cnt.most_common(1)[0][0] if cnt else "?"
        row["consensus"] = "✓" if cnt.most_common(1)[0][1] >= 3 else "△"
        prev_rows.append(row)

    prev_df = pd.DataFrame(prev_rows).set_index("paper_id")

    # ── Prevalence summary ───────────────────────────────────────────────────
    author_dist  = Counter(pivot["author"].apply(
        lambda x: Q4_DEC.get(int(x), "?") if not np.isnan(x) else "?"
    ).tolist())
    majority_dist = Counter(prev_df["majority"].tolist())

    # ── Report ───────────────────────────────────────────────────────────────
    lines = []
    lines.append("="*60)
    lines.append("SECOND-RATER IRR EXPERIMENT — RESULTS")
    lines.append("="*60)
    lines.append(f"Papers coded:  {len(pivot)}")
    lines.append(f"Raters:        {', '.join(raters)}")
    lines.append("")

    lines.append("── Krippendorff's α (Q4: Structural Validity) ──")
    lines.append(f"  4-way α (all raters):  {alpha_4way:.3f}")
    fk = compute_fleiss_kappa(pivot, raters)
    lines.append(f"  4-way Fleiss' κ (Rule R3, complete nominal matrix): {fk:.3f}")
    lines.append(f"  Threshold (exploratory): ≥ 0.67")
    if alpha_4way >= 0.67:
        lines.append(f"  ✓ MEETS exploratory threshold")
    elif alpha_4way >= 0.50:
        lines.append(f"  △ Below threshold — report with caution")
    else:
        lines.append(f"  ✗ Below acceptable range — discuss in limitations")
    lines.append("")

    lines.append("── Pairwise α ──")
    for pair, a in sorted(pairwise.items()):
        lines.append(f"  {pair:30s} α = {a:.3f}")
    lines.append("")

    lines.append("── Per-Paper Q4 Codes ──")
    header = f"{'Paper':<30} {'Auth':>5} {'Nemo':>5} {'Gemm':>5} {'Qwen':>5} {'Maj':>5} {'Con':>4}"
    lines.append(header)
    lines.append("-" * len(header))
    for paper_id, row in prev_df.iterrows():
        auth = row.get("author", "?")
        nemo = row.get("nemotron", "?")
        gemm = row.get("gemma", "?")
        qwen = row.get("qwen", "?")
        maj  = row.get("majority", "?")
        con  = row.get("consensus", "?")
        lines.append(
            f"{paper_id:<30} {auth:>5} {nemo:>5} {gemm:>5} {qwen:>5} {maj:>5} {con:>4}"
        )
    lines.append("")

    lines.append("── Prevalence (Q4 distribution) ──")
    lines.append("  Author codes:")
    for code in ("OK", "MM", "INC", "ABS"):
        n = author_dist.get(code, 0)
        lines.append(f"    {code}: {n} ({n/len(pivot)*100:.0f}%)")
    lines.append(f"  → Non-OK (majority vote): "
                 f"{sum(majority_dist.get(c,0) for c in ('MM','INC','ABS'))} / {len(pivot)} "
                 f"= {sum(majority_dist.get(c,0) for c in ('MM','INC','ABS'))/len(pivot)*100:.0f}%")
    lines.append("")

    lines.append("── §5.2 Update Paragraph (draft) ──")
    lines.append(
        "To assess coding reliability, a stratified subsample of N=20 papers (36% of "
        "the corpus; 2, 1, 3, 2, 2, 2, 2, 1, and 5 papers from categories A-I) was coded "
        "independently by three LLM raters from distinct model families (NVIDIA "
        "Nemotron-Ultra-550B; Google Gemma-4-31B; Alibaba Qwen3-80B-A3B). Each LLM rater "
        "received the coding instrument together with the paper's abstract and a verbatim "
        "excerpt from its evaluation-methodology text, with no access to the primary-rater "
        "codes. Four-way agreement across the author and the three LLM raters on the "
        f"structural validity dimension (Q4) was Fleiss' kappa = {fk:.2f} and "
        f"Krippendorff's alpha = {alpha_4way:.2f}. This falls below the exploratory "
        "threshold of alpha >= 0.67; the 55-paper prevalence estimate is therefore treated "
        "as a structured expert estimate rather than an IRR-validated finding, and this is "
        "noted as a limitation."
    )
    lines.append("="*60)

    report = "\n".join(lines)
    REPORT_FILE.write_text(report)
    print(report)
    print(f"\nReport saved to {REPORT_FILE}")

if __name__ == "__main__":
    main()
