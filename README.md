# Second-Rater IRR Coding Experiment

Supporting code and data for the reliability check in "Measurement Without
Validity: The Compounding Reliability Problem in Agentic AI Evaluation"
(KNOSYS-D-26-17924, Knowledge-Based Systems).

## What this measures

The paper's literature scan codes 55 agentic-evaluation papers on four
questions and finds that 80% of them misuse or omit inter-rater reliability.
That scan was coded by the author, so this experiment asks whether the coding
is reproducible: three LLMs from different model families independently apply
the same coding instrument to the same 55 papers, and agreement is computed
against the author's codes.

Reliability depends on what is being coded:

| Coding dimension | Krippendorff's alpha | Fleiss' kappa |
|---|---|---|
| Q1 which coefficient the authors computed | **0.834** | 0.818-0.846 |
| Q2 rater design | 0.371 | 0.348-0.387 |
| Q3 measurement scale | 0.525 | 0.514-0.578 |
| Q4 failure-mode judgment | 0.536 | 0.515-0.568 |
| Q4 collapsed to correct vs. misuse | 0.549 | |

Alpha is the median over five independent runs; the matrix is complete (four
raters x 55 papers, no missing cells). The determinate question of *which*
coefficient a paper computed is coded reliably. The interpretive judgment of
*whether* that use is correct is not, so the paper reports the 80% figure as a
structured expert estimate carrying a reproducibility range of 60.7% to 84.4%
(the spread across the four raters) rather than as a validated point estimate.

## Design choices that matter

**The stimulus is each paper's complete text, not an excerpt.** An excerpt
encodes the researcher's judgment about where the relevant evidence lies, and a
rater handed a pre-selected passage cannot disagree with that choice. One paper
in this corpus reports its Fleiss' kappa only in an appendix, so any excerpt
drawn from the methods section would have made the correct code unreachable.
Bibliographies are left in place deliberately: a reference-list entry naming a
coefficient is exactly the distractor the instrument must resist.

**Runs are not extra raters.** Each rater codes every paper five times.
Alpha is computed per run and reported as a median with its range, and
within-rater instability is reported separately from between-rater
disagreement, because pooling the two would conflate them.

**Nothing is imputed.** Missing cells are reported, unrecognised codes are
reported as findings rather than coerced into the nearest valid code, and the
cell census is printed on every run.

## Environment

Dependencies are pinned because they are part of the method: `pymupdf` fixes
the text extraction that defines the stimulus (`corpus.py` refuses to run
against a different version, since different extraction means a different
stimulus checksum), and the `openai`/`httpx` pins fix retry and error
semantics during the coding runs.

```bash
uv sync --frozen        # installs exactly what uv.lock records
export OPENROUTER_API_KEY=sk-or-...
```

`requirements.txt` is generated from `uv.lock` for anyone not using uv; the
authoritative definition is `pyproject.toml` plus `uv.lock`.

The coding runs were executed against a **paid OpenRouter subscription**, not
the free tier, so rate limits and routing behaviour correspond to paid access.

## The 55 PDFs are not included

The coded papers are third-party copyrighted works and cannot be
redistributed, so neither the PDFs nor the extracted text is in this
repository. Replication means obtaining the same papers yourself.
`manifest.csv` identifies every paper with its arXiv or DOI identifier, version,
fetch URL, and the SHA-256 of the exact PDF used; `stimulus_hashes.csv` holds
the SHA-256 of each extracted text. Once you have the PDFs, `extract.py`
verifies every checksum and will refuse to proceed on a mismatch, so you can
confirm byte-identical stimulus reconstruction before running anything.

## Pipeline

```bash
uv run python extract.py                 # PDFs  -> stimulus_fulltext.json + stimulus_hashes.csv
uv run python extract.py --verify        # re-extract and confirm hashes still match

uv run python judges.py --build          # record each judge: provider, context window, quantization
uv run python judges.py --verify         # fail if a judge's served configuration has drifted

uv run python second_rater.py            # the coding runs; resumable, skips completed cells
uv run python second_rater.py --status    # progress without making API calls

uv run python validate.py --results results_fulltext.csv   # gate: binding, vocabulary, conformance, grounding
uv run python compute_alpha.py --question Q1               # any of Q1..Q4 (default Q4)
```

`RUN_ID=n` selects the run; `STIMULUS_FILE`, `SUBSAMPLE_FILE`, `RESULTS_FILE`
and `PIN_QUANTIZATION` override the defaults. `second_rater.py` runs a preflight
gate that fails closed if the stimulus, instrument, judge set or served
quantization does not match what the study froze.

`audit_codes.py` is a separate check on the *author's* codes rather than a
rater: it searches each full document for every coefficient name and flags any
paper where the author's code and the text disagree. It is what found three
miscoded papers, each of which is corrected in `full_scan_55papers.csv`.

Tests are offline and make no API calls:

```bash
for t in test_*.py; do uv run python "$t"; done    # 165 checks
```

## Files

| File | Purpose |
|---|---|
| `manifest.csv` | The corpus definition: identifier, version, fetch URL and PDF SHA-256 for all 55 papers |
| `papers_55.csv` | Roster mapping `paper_id` to `zotero_key` and category |
| `full_scan_55papers.csv` | The author's Q1-Q4 codes for all 55 papers, with notes (Table A.2 in the paper) |
| `author_codes_55.csv` | The same codes in the shape the analysis consumes |
| `stimulus_hashes.csv` | SHA-256 of each extracted document: the shareable replication artifact |
| `judge_manifest.json` | The three judges as recorded entities: model id, providers, context windows, quantizations |
| `corpus.py` | Shared extraction library with the pinned-engine check |
| `extract.py` | Builds and verifies the stimulus from the PDFs |
| `judges.py` | Builds and verifies the judge manifest |
| `second_rater.py` | Runs the coding; preflight gate, resume, failure ledger |
| `validate.py` | The gate: binding, vocabulary, instrument conformance, evidence grounding |
| `compute_alpha.py` | Per-run alpha and Fleiss' kappa, pairwise breakdown, stability, prevalence |
| `audit_codes.py` | Lexical audit of the author's own codes against the full documents |
| `results_fulltext.csv` | All 825 ratings (55 papers x 3 raters x 5 runs) |
| `results_fulltext.failures.csv` | Failure ledger for the runs above (empty: no failures occurred) |
| `alpha_report_Q1.txt` .. `_Q4.txt` | Output of `compute_alpha.py`, one report per coding dimension |
| `validation_report_fulltext.txt` | Output of `validate.py` |
| `audit_report.csv` | Output of `audit_codes.py`: per paper, which coefficient names appear in the full document, where, whether they are value-anchored, and whether that agrees with the author's code. This is the per-code evidence trail, re-derived mechanically rather than hand-assembled |
| `test_*.py` | Offline regression tests (165 checks) |

## The coding instrument

| Q | Question | Codes |
|---|---|---|
| Q1 | Which IRR metric did the authors compute on their own annotations? | KCO, KFL, KAL, ICC, PA, NIL |
| Q2 | What was the rater design? | F2, FN, RV, CS, UNK |
| Q3 | What was the measurement scale? | BIN, NOM, ORD, CON, UNK |
| Q4 | Did the metric fit the design and scale? | OK, MM, INC, ABS |

Q1 asks what was *computed*, not what was mentioned: a reference-list entry is
a citation, naming a coefficient is not reporting one, and a computed
coefficient appears with a value. Appendices count.

Q4: `OK` metric fits design and scale with a stated threshold and rationale;
`MM` structural mismatch; `INC` percent agreement only, or the correct metric
family with no threshold or rationale; `ABS` no IRR metric reported.
Rule: if Q1 is `NIL` then Q4 is `ABS`, with no exceptions. Rule: apply `MM`
before `INC`, because a structural mismatch takes priority.

## Citation

```
Caban, W. (2026). Measurement Without Validity: The Compounding Reliability
Problem in Agentic AI Evaluation. Knowledge-Based Systems (under review).
https://github.com/williamcaban/experiment-measurement-without-validity
```

Released under the Apache 2.0 licence.
