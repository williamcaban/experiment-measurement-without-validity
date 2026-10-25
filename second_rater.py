#!/usr/bin/env python3
"""
second_rater.py — LLM second-rater coding runs over the IRR corpus.

Three LLM raters from distinct model families apply the 4-dimension coding
instrument to each paper in the reliability subsample. These models are not
deterministic at temperature 0, so the design calls for several independent
passes over the same stimulus; each pass is one RUN_ID inside one results file.

WHAT THIS SCRIPT GUARANTEES

1. NO CONTAMINATION. A results file is bound to exactly one (stimulus,
   instrument) pair. Every row records the sha256 of the exact text that was
   sent and of the exact instrument wording used. Startup refuses to append to
   a file whose schema or hashes differ from the current ones, so a run over
   new stimulus cannot silently merge with rows produced from the old one.
   Nothing is inferred from file names.

2. RESUMABILITY. Work is addressed as (run, paper, model) cells. A cell counts
   as done only when a row exists whose four codes are all in the instrument's
   vocabulary. Rows are flushed and fsynced one at a time, so a kill at any
   point leaves a consistent file and re-invoking the same command picks up
   exactly the cells that are missing.

3. TRANSPORT FAULT TOLERANCE. Timeouts, dropped connections, 429s and 5xx are
   retried with exponential backoff and jitter, honouring Retry-After when the
   server sends it. Errors retrying cannot fix (auth, bad model slug, malformed
   request) abort immediately instead of burning the retry budget.

4. FAILURES NEVER ENTER THE MATRIX AND ARE NEVER LOST. An empty, truncated or
   unparseable completion is retried; if it still fails, the cell is written to
   an append-only failure ledger rather than to results, the run moves on to the
   next paper, and the cell is retried in a sweep at the end of the run and
   again on any later invocation. A permanently missing cell stays visible and
   countable instead of being silently coded.

USAGE
  export OPENROUTER_API_KEY=sk-or-...        # or put it in experiment/.env
  python second_rater.py --status            # offline: what is done, what is not
  python second_rater.py --preflight --ping  # checks, including live models
  python second_rater.py                     # run (resumes automatically)
  python second_rater.py --sweep-only        # retry only the deferred cells
  python second_rater.py --fresh             # archive existing results, start over
  python second_rater.py --limit 2           # smoke test on the first 2 papers

ENVIRONMENT
  STIMULUS_FILE   stimulus json            (default: stimulus_fulltext.json)
  STIMULUS_NOUN   instrument wording       (default: from the stimulus field)
  RESULTS_FILE    results csv              (default: results_fulltext.csv)
  RUN_ID          which pass this is       (default: 1)
  SUBSAMPLE_FILE  which papers to code     (default: papers_55.csv)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import random
import re
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import openai
from openai import OpenAI

HERE = Path(__file__).parent

# Secrets
_env_file = HERE / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"

MODELS = {
    "nemotron": "nvidia/nemotron-3-ultra-550b-a55b",   # NVIDIA - 550B MoE
    "gemma":    "google/gemma-4-31b-it",               # Google - 31B
    "qwen":     "qwen/qwen3-next-80b-a3b-instruct",    # Alibaba - 80B MoE
}

# Pacing and fault tolerance
INTER_PAPER_WAIT = float(os.environ.get("INTER_PAPER_WAIT", 12))
MAX_ATTEMPTS     = int(os.environ.get("MAX_ATTEMPTS", 4))
BACKOFF_BASE     = float(os.environ.get("BACKOFF_BASE", 4))
BACKOFF_CAP      = float(os.environ.get("BACKOFF_CAP", 90))
RATE_LIMIT_WAIT  = float(os.environ.get("RATE_LIMIT_WAIT", 65))
READ_TIMEOUT     = float(os.environ.get("READ_TIMEOUT", 300))

# Quantization pin, per judge. The same slug served at fp4 and at bf16 is two
# different functions, and OpenRouter's routing is not deterministic, so an
# unpinned run samples several numerically distinct raters under one name -
# which would appear in the results as model non-determinism and be
# indistinguishable from it. Each pin is the one with the widest fallback pool
# for that model; the authority for these values is judge_manifest.json, which
# preflight checks this dict against. Override with
# PIN_QUANTIZATION="nemotron=fp4,gemma=fp8"; PIN_QUANTIZATION=none disables
# pinning entirely (recorded per row, so an unpinned run is never mistaken for
# a pinned one).
PIN_QUANT = {"nemotron": "fp4", "gemma": "fp8", "qwen": "fp8"}
_pin_env = os.environ.get("PIN_QUANTIZATION", "").strip()
if _pin_env.lower() in ("none", "off"):
    PIN_QUANT = {}
elif _pin_env:
    PIN_QUANT = dict(kv.split("=", 1) for kv in _pin_env.split(",") if "=" in kv)
SWEEP_PASSES     = int(os.environ.get("SWEEP_PASSES", 2))
SWEEP_PAUSE      = float(os.environ.get("SWEEP_PAUSE", 30))

# Output budget. Nemotron emits reasoning tokens before its answer and the
# provider does not honour enable_thinking=False, so a short budget returns
# finish_reason=length with empty content. Truncation is detected and the
# budget escalated on the next attempt rather than guessed once up front.
MAX_TOKENS     = int(os.environ.get("MAX_TOKENS", 1500))
MAX_TOKENS_COT = int(os.environ.get("MAX_TOKENS_COT", 8000))
MAX_TOKENS_CAP = int(os.environ.get("MAX_TOKENS_CAP", 16000))
COT_MODELS     = {"nemotron"}

STIMULUS_FILE  = HERE / os.environ.get("STIMULUS_FILE", "stimulus_fulltext.json")
RESULTS_FILE   = HERE / os.environ.get("RESULTS_FILE", "results_fulltext.csv")
SUBSAMPLE_FILE = HERE / os.environ.get("SUBSAMPLE_FILE", "papers_55.csv")
RUN_ID         = os.environ.get("RUN_ID", "1")

FAILURES_FILE = RESULTS_FILE.with_suffix(".failures.csv")
ARCHIVE_DIR   = HERE / "archive"

PRINT_LOCK = threading.Lock()


def say(msg: str) -> None:
    with PRINT_LOCK:
        print(msg, flush=True)


# Coding instrument.
# NOUN is substituted wherever the instrument refers to the material being
# coded. Changing it changes the instrument, which changes INSTRUMENT_SHA,
# which makes it impossible to pool excerpt-stimulus and full-text-stimulus
# rows in one results file. That is intended, not an inconvenience.
SYSTEM_PROMPT_T = """You are a research methodology analyst coding papers for a systematic
review of inter-rater reliability (IRR) practices in AI evaluation research.

Your task is to read the paper {NOUN} and answer 4 structured coding questions.
Base your codes ONLY on what is explicitly stated in the text provided.
Do not infer or assume practices not mentioned. If uncertain, choose the most
conservative code and note it in your justification."""

CODING_INSTRUMENT_T = """
CODING INSTRUMENT

For each question, choose EXACTLY ONE code from the list. Output only the 3-letter code,
not the question text.

Q1 CODES - which IRR metric the authors COMPUTED on their own annotations:
  KCO = Cohen's kappa
  KFL = Fleiss' kappa
  KAL = Krippendorff's alpha
  ICC = Intraclass correlation coefficient
  PA  = Percentage/proportion agreement only (no kappa or alpha)
  NIL = No recognized IRR metric (automated scoring, ELO, Pearson, or nothing)

WHAT COUNTS AS COMPUTED, FOR Q1:
The {NOUN} you are given is the complete paper. It includes the reference list
and any appendices.
- A work in the reference list is a CITATION, not a reported coefficient.
  "Cohen, J. (1960). A coefficient of agreement for nominal scales" is a
  citation and is NOT grounds for KCO.
- Naming a metric is not reporting one. A paper that defines, reviews,
  recommends or criticises a coefficient without computing it on its own data
  is NIL for that metric.
- A computed coefficient is one the authors obtained from their own
  annotations. It almost always appears with a value, as in
  "Fleiss' kappa = 0.68" or "we measured an alpha of .81".
- Appendices count as part of the paper. A coefficient reported only in an
  appendix is still computed and reported.

Q2 CODES - Rater design:
  F2  = Exactly 2 raters, same fixed pair on every item
  FN  = Fixed panel of 3+ raters scoring every item
  RV  = Rotating/varying rater identity across items
  CS  = Crowdsourcing platform or open-ended user voting pool
  UNK = Not described, or paper has no rater design

Q3 CODES - Measurement scale:
  BIN = Binary (pass/fail, yes/no, 0/1, A vs B)
  NOM = Nominal unordered categories
  ORD = Ordered/ordinal (1-5 Likert, ternary, rubric)
  CON = Continuous or interval (numeric averages, probabilities)
  UNK = Not described, or paper has no rating scale

Q4 CODES - Structural validity (apply rules in this order, stop at first match):
  ABS = Q1 is NIL (no recognized IRR metric reported)
  MM  = Q1=KCO but Q2 is not F2; OR Q1=KCO/KFL but Q3 is ORD or CON
  INC = Q1=PA; OR correct metric family but no threshold and no rationale stated
  OK  = Metric fits Q2 and Q3, threshold stated or implied, rationale provided

STRICT RULES:
- If Q1=NIL then Q4=ABS (no exceptions)
- Check MM before INC
- Q1 records the metric the authors computed. Whether that choice was
  appropriate is Q4's job, not Q1's: code the metric they used even when it is
  the wrong one for their design.
- Never cite a reference-list entry as the evidence sentence for Q1.

REQUIRED OUTPUT FORMAT - output ONLY these 4 lines, nothing before them:
Q1: KCO | <one sentence from the {NOUN}>
Q2: F2 | <one sentence from the {NOUN}>
Q3: BIN | <one sentence from the {NOUN}>
Q4: OK | <one sentence of justification>

Replace KCO/F2/BIN/OK with the actual codes for THIS paper.
"""

USER_TAIL = "Apply the coding instrument to this paper. Output exactly 4 lines."

VALID_CODES = {
    "Q1": {"KCO", "KFL", "KAL", "ICC", "PA", "NIL"},
    "Q2": {"F2", "FN", "RV", "CS", "UNK"},
    "Q3": {"BIN", "NOM", "ORD", "CON", "UNK"},
    "Q4": {"ABS", "MM", "INC", "OK"},
}

RESULT_FIELDS = [
    "paper_id", "category", "model", "model_id", "run",
    "Q1", "Q2", "Q3", "Q4", "raw",
    "instrument_sha", "stimulus_sha", "stimulus_chars",
    "finish_reason", "prompt_tokens", "completion_tokens",
    # Serving provenance. The same slug is served by many providers at
    # different quantizations, and OpenRouter's routing is not deterministic
    # (two identical calls to nemotron were answered by Venice and then
    # BaseTen). Which provider answered is therefore part of the observation,
    # not an implementation detail: without it, provider-routing variance is
    # indistinguishable from model non-determinism. generation_id allows any
    # single cell to be retrieved later from OpenRouter's own records.
    "provider", "generation_id", "pinned_quantizations",
    "attempts", "latency_s", "timestamp",
]

FAILURE_FIELDS = [
    "paper_id", "model", "model_id", "run", "reason", "detail",
    "attempts", "instrument_sha", "stimulus_sha", "timestamp",
]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Instrument:
    """The exact wording sent to the raters, plus its fingerprint."""

    def __init__(self, noun: str):
        self.noun = noun
        self.system = SYSTEM_PROMPT_T.format(NOUN=noun)
        self.body = CODING_INSTRUMENT_T.format(NOUN=noun)
        self.tail = USER_TAIL
        self.sha = sha(self.system + "\x00" + self.body + "\x00" + self.tail)[:16]

    def user_prompt(self, paper_id: str, title: str, text: str) -> str:
        return (
            f"{self.body}\n\n"
            f"PAPER {self.noun.upper()} [{paper_id}] - {title}\n"
            f"{'=' * 60}\n"
            f"{text}\n"
            f"{'=' * 60}\n\n"
            f"{self.tail}"
        )


def load_stimulus(path: Path, subsample: Path) -> tuple[list[dict], str]:
    """Return the ordered work list and the stimulus noun.

    Accepts records whose text lives in `text`, `full_text` or `excerpt`. The
    field name also sets the default instrument noun, so the wording of the
    instrument and the nature of the stimulus cannot drift apart.
    """
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise SystemExit(f"FATAL: {path.name} is not a non-empty JSON list")

    field = next((f for f in ("text", "full_text", "excerpt") if f in records[0]), None)
    if field is None:
        raise SystemExit(
            f"FATAL: records in {path.name} have no text field "
            f"(looked for 'text', 'full_text', 'excerpt'); keys: {list(records[0])}"
        )
    noun = os.environ.get("STIMULUS_NOUN") or ("excerpt" if field == "excerpt" else "full text")

    with open(subsample, newline="") as fh:
        wanted = [row["paper_id"] for row in csv.DictReader(fh)]
    if not wanted:
        raise SystemExit(f"FATAL: {subsample.name} lists no papers")

    by_id = {r["paper_id"]: r for r in records}
    missing = [p for p in wanted if p not in by_id]
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)} subsample papers absent from {path.name}: "
            f"{', '.join(missing)}"
        )

    work = []
    for pid in wanted:
        rec = by_id[pid]
        text = (rec.get(field) or "").strip()
        if not text:
            raise SystemExit(f"FATAL: {pid} has empty stimulus text in {path.name}")
        work.append({
            "paper_id": pid,
            "title": rec.get("title", ""),
            "category": rec.get("category", ""),
            "text": text,
            "sha": sha(text)[:16],
            "chars": len(text),
        })
    return work, noun


def read_rows(path: Path, fields: list[str]) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        if header != fields:
            raise SystemExit(
                f"FATAL: {path.name} has an incompatible schema.\n"
                f"  found:    {header}\n"
                f"  expected: {fields}\n"
                "This file was written by a different version of the instrument or\n"
                "pipeline, so appending to it would mix incomparable rows. Either\n"
                "archive it with --fresh, or point RESULTS_FILE at a new path."
            )
        return list(reader)


def assert_uncontaminated(rows: list[dict], instrument: Instrument,
                          work: list[dict]) -> None:
    """A results file holds exactly one (instrument, stimulus) pair."""
    if not rows:
        return
    want = {w["paper_id"]: w["sha"] for w in work}
    bad_instr = {r["instrument_sha"] for r in rows} - {instrument.sha}
    if bad_instr:
        raise SystemExit(
            f"FATAL: {RESULTS_FILE.name} contains rows coded under a different\n"
            f"instrument wording ({', '.join(sorted(bad_instr))}); this run uses\n"
            f"{instrument.sha}. Pooling them would compare raters who were asked\n"
            "different questions. Use --fresh, or a new RESULTS_FILE."
        )
    drift = sorted({
        f"{r['paper_id']} (file {r['stimulus_sha']} vs stimulus "
        f"{want.get(r['paper_id'], 'ABSENT')})"
        for r in rows if want.get(r["paper_id"]) != r["stimulus_sha"]
    })
    if drift:
        raise SystemExit(
            f"FATAL: {RESULTS_FILE.name} contains rows coded from different "
            "stimulus text:\n  " + "\n  ".join(drift[:12]) +
            (f"\n  ... and {len(drift) - 12} more" if len(drift) > 12 else "") +
            "\nUse --fresh, or a new RESULTS_FILE."
        )
    blank = sum(1 for r in rows if not (r.get("run") or "").strip())
    if blank:
        raise SystemExit(f"FATAL: {RESULTS_FILE.name} has {blank} rows with no run id.")


def valid_result(row: dict) -> bool:
    return all(row.get(q) in VALID_CODES[q] for q in VALID_CODES)


class Ledger:
    """Append-only results and failure logs, durable one row at a time."""

    def __init__(self, results: Path, failures: Path):
        self.results_path, self.failures_path = results, failures
        self._rf = self._open(results, RESULT_FIELDS)
        self._ff = self._open(failures, FAILURE_FIELDS)
        self.results_writer = csv.DictWriter(self._rf, fieldnames=RESULT_FIELDS)
        self.failures_writer = csv.DictWriter(self._ff, fieldnames=FAILURE_FIELDS)
        self._lock = threading.Lock()

    @staticmethod
    def _open(path: Path, fields: list[str]):
        new = not path.exists() or path.stat().st_size == 0
        fh = open(path, "a", newline="", encoding="utf-8")
        if new:
            csv.DictWriter(fh, fieldnames=fields).writeheader()
            fh.flush()
            os.fsync(fh.fileno())
        return fh

    def result(self, row: dict) -> None:
        with self._lock:
            self.results_writer.writerow(row)
            self._rf.flush()
            os.fsync(self._rf.fileno())

    def failure(self, row: dict) -> None:
        with self._lock:
            self.failures_writer.writerow(row)
            self._ff.flush()
            os.fsync(self._ff.fileno())

    def close(self) -> None:
        self._rf.close()
        self._ff.close()


FATAL_EXC = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.BadRequestError,
    openai.UnprocessableEntityError,
)
RETRY_EXC = (
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
    openai.ConflictError,
    httpx.TimeoutException,
    httpx.TransportError,
)
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}


def backoff(attempt: int) -> float:
    return min(BACKOFF_BASE * (2 ** (attempt - 1)), BACKOFF_CAP) * random.uniform(0.75, 1.25)


def retry_after(exc: Exception) -> float | None:
    hdrs = getattr(getattr(exc, "response", None), "headers", None) or {}
    raw = hdrs.get("retry-after") or hdrs.get("Retry-After")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


class Outcome:
    def __init__(self, ok: bool, reason: str = "", detail: str = "", **kw):
        self.ok, self.reason, self.detail = ok, reason, detail
        self.attempts = kw.pop("attempts", 0)
        self.__dict__.update(kw)


def parse_codes(raw: str) -> tuple[dict | None, str]:
    """Strict parse: every question must carry one code from its vocabulary."""
    codes: dict[str, str] = {}
    for line in raw.splitlines():
        # Models emphasise these lines in several ways: "**Q1:** KCO", "`Q1`: KCO",
        # "## Q1 - KCO". Emphasis is not meaning, so strip it before matching.
        flat = re.sub(r"[*`#_]+", " ", line)
        m = re.match(r"^\s*(Q[1-4])\s*[:\-\u2013]\s*(.+)$", flat)
        if not m:
            continue
        q, rest = m.group(1), m.group(2)
        token = re.split(r"[|\s,;]", rest.strip(), maxsplit=1)[0]
        codes.setdefault(q, token.strip().upper().strip(".:-"))
    absent = [q for q in ("Q1", "Q2", "Q3", "Q4") if q not in codes]
    if absent:
        return None, f"no code line for {','.join(absent)}"
    bad = [f"{q}={codes[q]}" for q in sorted(VALID_CODES) if codes[q] not in VALID_CODES[q]]
    if bad:
        return None, f"code outside vocabulary: {', '.join(bad)}"
    return codes, ""


def code_one_cell(client: OpenAI, instrument: Instrument, model_name: str,
                  model_id: str, paper: dict) -> Outcome:
    """Call one model on one paper, retrying what retrying can fix.

    A failed Outcome carries the reason the caller records in the failure
    ledger, and the cell stays retriable. Fatal configuration errors raise,
    because no amount of retrying or skipping fixes a bad key or model slug.
    """
    budget = MAX_TOKENS_COT if model_name in COT_MODELS else MAX_TOKENS
    extra: dict = {}
    if model_name in COT_MODELS:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    if PIN_QUANT.get(model_name):
        extra["provider"] = {"quantizations": [PIN_QUANT[model_name]]}
    extra = extra or None
    last = ("UNKNOWN", "no attempt completed")
    tag = f"    [{model_name:9}]"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.monotonic()
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": instrument.system},
                    {"role": "user", "content": instrument.user_prompt(
                        paper["paper_id"], paper["title"], paper["text"])},
                ],
                temperature=0,
                max_tokens=budget,
                extra_body=extra,
            )
        except FATAL_EXC as exc:
            raise SystemExit(
                f"FATAL: [{model_name}] {type(exc).__name__}: {exc}\n"
                "Retrying cannot fix this (credentials, model slug or request shape).\n"
                "Completed cells are saved; fix the cause and re-run to resume."
            )
        except RETRY_EXC as exc:
            wait = retry_after(exc)
            if wait is None:
                wait = (RATE_LIMIT_WAIT if isinstance(exc, openai.RateLimitError)
                        else backoff(attempt))
            last = (type(exc).__name__.upper(), f"{type(exc).__name__}: {exc}"[:400])
            say(f"{tag} {type(exc).__name__} on attempt {attempt}/{MAX_ATTEMPTS} "
                f"- waiting {wait:.0f}s")
            if attempt < MAX_ATTEMPTS:
                time.sleep(wait)
            continue
        except openai.APIStatusError as exc:
            last = (f"HTTP_{exc.status_code}", f"{exc}"[:400])
            if exc.status_code not in RETRY_STATUS:
                say(f"{tag} HTTP {exc.status_code} is not retryable - cell deferred")
                break
            wait = retry_after(exc) or backoff(attempt)
            say(f"{tag} HTTP {exc.status_code} on attempt {attempt}/{MAX_ATTEMPTS} "
                f"- waiting {wait:.0f}s")
            if attempt < MAX_ATTEMPTS:
                time.sleep(wait)
            continue
        except Exception as exc:
            last = ("UNEXPECTED", f"{type(exc).__name__}: {exc}"[:400])
            say(f"{tag} unexpected {type(exc).__name__} on attempt "
                f"{attempt}/{MAX_ATTEMPTS}: {exc}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(backoff(attempt))
            continue

        latency = time.monotonic() - started

        # OpenRouter can answer 200 with an error payload, or with no choices.
        err = getattr(resp, "error", None)
        if err is None and hasattr(resp, "model_dump"):
            err = resp.model_dump().get("error")
        if err or not getattr(resp, "choices", None):
            last = ("NO_CHOICES", f"error={err} choices={getattr(resp, 'choices', None)}"[:400])
            say(f"{tag} HTTP 200 with no usable choice on attempt "
                f"{attempt}/{MAX_ATTEMPTS}: {err}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(backoff(attempt))
            continue

        choice = resp.choices[0]
        finish = getattr(choice, "finish_reason", "") or ""
        content = (getattr(choice.message, "content", None) or "").strip()
        usage = getattr(resp, "usage", None)
        pt = getattr(usage, "prompt_tokens", "") if usage else ""
        ct = getattr(usage, "completion_tokens", "") if usage else ""
        served_by = getattr(resp, "provider", "") or ""
        gen_id = getattr(resp, "id", "") or ""

        if not content:
            if finish == "length":
                # Deterministic cause: reasoning consumed the whole budget.
                # Escalate instead of re-rolling the same losing call.
                old, budget = budget, min(int(budget * 2), MAX_TOKENS_CAP)
                last = ("TRUNCATED", f"empty content, finish_reason=length, budget={old}")
                say(f"{tag} truncated at {old} output tokens "
                    f"(attempt {attempt}/{MAX_ATTEMPTS})"
                    + (f" - retrying with {budget}" if budget > old
                       else f" - budget capped at {MAX_TOKENS_CAP}, re-rolling"))
            else:
                last = ("EMPTY", f"empty content, finish_reason={finish!r}")
                say(f"{tag} empty completion, finish_reason={finish!r} "
                    f"(attempt {attempt}/{MAX_ATTEMPTS})")
            if attempt < MAX_ATTEMPTS:
                time.sleep(backoff(attempt))
            continue

        codes, why = parse_codes(content)
        if codes is None:
            last = ("PARSE", f"{why} || {content[:300]}")
            say(f"{tag} unparseable ({why}) on attempt {attempt}/{MAX_ATTEMPTS}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(backoff(attempt))
            continue

        return Outcome(True, codes=codes, raw=content, finish_reason=finish,
                       prompt_tokens=pt, completion_tokens=ct,
                       provider=served_by, generation_id=gen_id,
                       attempts=attempt, latency=latency)

    return Outcome(False, reason=last[0], detail=last[1], attempts=MAX_ATTEMPTS)


class Stop:
    """Cooperative shutdown: finish the paper in flight, then exit resumably."""

    def __init__(self):
        self.asked = False
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._handle)

    def _handle(self, signum, frame):
        if self.asked:
            print("\n[ABORT] second signal - exiting now; completed cells are saved.")
            sys.exit(130)
        self.asked = True
        print(f"\n[STOP] signal {signum} received - finishing the current paper, "
              "then exiting. Re-run the same command to resume.")


def cells_done(rows: list[dict], run_id: str) -> set[tuple[str, str]]:
    return {(r["paper_id"], r["model"]) for r in rows
            if r.get("run") == run_id and valid_result(r)}


def cells_deferred(fail_rows: list[dict], run_id: str,
                   done: set[tuple[str, str]]) -> dict[tuple[str, str], list[dict]]:
    """Cells with a recorded failure and still no valid result: retriable."""
    out: dict[tuple[str, str], list[dict]] = {}
    for r in fail_rows:
        if r.get("run") != run_id:
            continue
        key = (r["paper_id"], r["model"])
        if key not in done:
            out.setdefault(key, []).append(r)
    return out


def print_status(work: list[dict], instrument: Instrument,
                 rows: list[dict], fail_rows: list[dict]) -> None:
    runs = sorted({r["run"] for r in rows} | {r["run"] for r in fail_rows} | {RUN_ID},
                  key=lambda s: (len(s), s))
    total = len(work) * len(MODELS)
    print(f"stimulus      : {STIMULUS_FILE.name}  ({len(work)} papers, "
          f"noun={instrument.noun!r}, {sum(w['chars'] for w in work):,} chars)")
    print(f"instrument    : sha {instrument.sha}")
    print(f"results       : {RESULTS_FILE.name}  ({len(rows)} rows)")
    print(f"failures      : {FAILURES_FILE.name}  ({len(fail_rows)} events)")
    print(f"cells per run : {total}  ({len(work)} papers x {len(MODELS)} models)\n")
    print(f"{'run':>5}  {'valid':>6}  {'deferred':>9}  {'untried':>8}  outstanding reasons")
    for run in runs:
        done = cells_done(rows, run)
        pend = cells_deferred(fail_rows, run, done)
        untried = total - len(done) - len(pend)
        reasons: dict[str, int] = {}
        for events in pend.values():
            reasons[events[-1]["reason"]] = reasons.get(events[-1]["reason"], 0) + 1
        detail = ", ".join(f"{k} x{v}" for k, v in sorted(reasons.items())) or "-"
        print(f"{run:>5}  {len(done):>6}  {len(pend):>9}  {untried:>8}  {detail}"
              + ("   <- RUN_ID" if run == RUN_ID else ""))
    invalid = [r for r in rows if not valid_result(r)]
    if invalid:
        print(f"\nWARNING: {len(invalid)} results row(s) carry codes outside the "
              "vocabulary; they are ignored and their cells count as not done.")


def make_client() -> OpenAI:
    # max_retries=0: this script owns the retry policy, so every attempt is
    # counted and logged instead of happening invisibly inside the SDK.
    return OpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=0,
                  timeout=httpx.Timeout(READ_TIMEOUT, connect=15.0))


def preflight(work: list[dict], instrument: Instrument, rows: list[dict],
              fail_rows: list[dict], ping: bool) -> None:
    print("-- preflight ---------------------------------------------")
    ok = True

    def check(label: str, passed: bool, note: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f" - {note}" if note else ""))

    check("API key present", bool(API_KEY) and API_KEY.startswith("sk-"),
          f"{API_KEY[:8]}...{API_KEY[-4:]}" if API_KEY
          else "set OPENROUTER_API_KEY or experiment/.env")
    check(f"stimulus readable: {STIMULUS_FILE.name}", True,
          f"{len(work)} papers, noun={instrument.noun!r}")
    check("every paper has text", all(w["chars"] > 0 for w in work))
    longest = max(work, key=lambda w: w["chars"])
    check("stimulus size", True,
          f"max {longest['chars']:,} chars ({longest['paper_id']}), "
          f"~{longest['chars'] // 4:,} tokens est.")
    check("stimulus text differs per paper", len({w["sha"] for w in work}) == len(work))
    check("results schema compatible", True, f"{len(rows)} existing rows")
    check("no instrument or stimulus drift in results file", True, "verified at load")
    done = cells_done(rows, RUN_ID)
    pend = cells_deferred(fail_rows, RUN_ID, done)
    check(f"run {RUN_ID} work remaining", True,
          f"{len(work) * len(MODELS) - len(done)} cells "
          f"({len(pend)} of them previously deferred)")
    check("output budgets", True, ", ".join(
        f"{m}={MAX_TOKENS_COT if m in COT_MODELS else MAX_TOKENS}" for m in MODELS))
    check("client timeout set", READ_TIMEOUT > 0, f"{READ_TIMEOUT:.0f}s read, 15s connect")
    check("retry policy", MAX_ATTEMPTS >= 2,
          f"{MAX_ATTEMPTS} attempts/cell, backoff {BACKOFF_BASE:.0f}s x2 "
          f"cap {BACKOFF_CAP:.0f}s, {SWEEP_PASSES} end-of-run sweeps")

    # Judge attributes. With full documents as the stimulus the context window
    # is a property of the rater, not of the plumbing: the largest paper is
    # ~60k tokens, and a judge served behind a smaller window silently
    # truncates it. Silent truncation would not raise, would not fail to parse,
    # and would produce a confident code on a partial paper - the single worst
    # failure available here, because it is invisible in the output. So the
    # frozen manifest is a hard gate, not a note.
    man_path = HERE / "judge_manifest.json"
    if not man_path.exists():
        check("judge manifest present", False,
              "judge_manifest.json missing - run: python3 judges.py --build")
    else:
        man = json.loads(man_path.read_text())
        req = man["replication"]["min_context_tokens_required"]
        check("judge manifest present", True,
              f"built {man['built_utc']}, requires >={req:,} tokens of context")
        check("judge set matches manifest",
              set(man["judges"]) == set(MODELS),
              f"manifest={sorted(man['judges'])} runner={sorted(MODELS)}")
        for name, j in man["judges"].items():
            if name not in MODELS:
                continue
            check(f"  {name} slug unchanged", j["slug"] == MODELS[name],
                  f"manifest={j['slug']} runner={MODELS.get(name)}")
            lo = j.get("provider_context_min")
            check(f"  {name} every provider >= requirement",
                  bool(lo) and lo >= req,
                  f"smallest provider context {lo:,} vs {req:,} required")
        # The requirement was derived from the largest stimulus; if this run
        # uses a different stimulus file, re-derive rather than assume.
        hdr = max(len(instrument.user_prompt(w["paper_id"], "", "")) for w in work)
        worst_chars = longest["chars"] + hdr
        # 3.9 chars/token is the densest of the three measured tokenizers.
        worst_tokens = int(worst_chars / 3.9)
        check("largest prompt fits the requirement", worst_tokens <= req,
              f"~{worst_tokens:,} tokens worst case vs {req:,} required"
              + ("" if worst_tokens <= req
                 else " - re-run judges.py --build for this stimulus"))
        for name, j in man["judges"].items():
            if name not in MODELS:
                continue
            want = j.get("pinned_quantization")
            check(f"  {name} quantization pin matches manifest",
                  PIN_QUANT.get(name) == want,
                  f"runner={PIN_QUANT.get(name)!r} manifest={want!r}"
                  + ("" if PIN_QUANT.get(name) == want
                     else " - an unpinned or differently pinned judge is not "
                          "the rater the manifest describes"))
            n = j.get("n_providers_at_pin", 0)
            check(f"  {name} pin has a fallback pool", n >= 2,
                  f"{n} distinct provider(s) serve {want}"
                  + ("" if n >= 2 else " - an outage will stall this judge"))
    check("quantization pin", True,
          ", ".join(f"{k}={v}" for k, v in sorted(PIN_QUANT.items()))
          if PIN_QUANT
          else "NONE - routing may vary quantization between cells")

    if ping:
        client = make_client()
        for name, mid in MODELS.items():
            try:
                r = client.chat.completions.create(
                    model=mid, max_tokens=4, temperature=0,
                    messages=[{"role": "user", "content": "Reply with the word OK."}])
                txt = (r.choices[0].message.content or "").strip() if r.choices else ""
                check(f"live model {name}", True, f"{mid} -> {txt[:20]!r}")
            except Exception as exc:
                check(f"live model {name}", False, f"{mid}: {type(exc).__name__}: {exc}")

    print("-" * 58)
    print("preflight: " + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE - not ready"))
    if not ok:
        sys.exit(1)


def archive_existing() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ARCHIVE_DIR.mkdir(exist_ok=True)
    moved = []
    for path in (RESULTS_FILE, FAILURES_FILE):
        if path.exists() and path.stat().st_size > 0:
            dest = ARCHIVE_DIR / f"{path.stem}__{stamp}{path.suffix}"
            path.rename(dest)
            moved.append(dest.name)
    print(f"[FRESH] archived: {', '.join(moved) if moved else '(nothing to archive)'}")


def code_cells(client: OpenAI, instrument: Instrument, ledger: Ledger,
               targets: list[tuple[dict, list[str]]], stop: Stop,
               label: str) -> tuple[int, int]:
    n_ok = n_fail = 0
    for idx, (paper, model_names) in enumerate(targets, 1):
        if stop.asked:
            print("[STOP] halting before the next paper; re-run to resume.")
            break
        say(f"\n[{label} {idx}/{len(targets)}] {paper['paper_id']} "
            f"({paper['category']}, {paper['chars']:,} chars) -> {', '.join(model_names)}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
            futures = {
                pool.submit(code_one_cell, client, instrument, name, MODELS[name], paper): name
                for name in model_names
            }
            for fut in concurrent.futures.as_completed(futures):
                name = futures[fut]
                out = fut.result()
                ts = datetime.now(timezone.utc).isoformat()
                if out.ok:
                    ledger.result({
                        "paper_id": paper["paper_id"], "category": paper["category"],
                        "model": name, "model_id": MODELS[name], "run": RUN_ID,
                        **out.codes,
                        "raw": out.raw.replace("\n", " // "),
                        "instrument_sha": instrument.sha,
                        "stimulus_sha": paper["sha"], "stimulus_chars": paper["chars"],
                        "finish_reason": out.finish_reason,
                        "prompt_tokens": out.prompt_tokens,
                        "completion_tokens": out.completion_tokens,
                        "provider": out.provider,
                        "generation_id": out.generation_id,
                        "pinned_quantizations": PIN_QUANT.get(name, ""),
                        "attempts": out.attempts, "latency_s": f"{out.latency:.1f}",
                        "timestamp": ts,
                    })
                    n_ok += 1
                    c = out.codes
                    say(f"  ok    [{name:9}] Q1={c['Q1']:3} Q2={c['Q2']:3} "
                        f"Q3={c['Q3']:3} Q4={c['Q4']:3}  "
                        f"({out.attempts} att, {out.latency:.0f}s, {out.completion_tokens} out)")
                else:
                    ledger.failure({
                        "paper_id": paper["paper_id"], "model": name,
                        "model_id": MODELS[name], "run": RUN_ID,
                        "reason": out.reason, "detail": out.detail,
                        "attempts": out.attempts,
                        "instrument_sha": instrument.sha,
                        "stimulus_sha": paper["sha"], "timestamp": ts,
                    })
                    n_fail += 1
                    say(f"  DEFER [{name:9}] {out.reason} after {out.attempts} attempts "
                        "- retried in the sweep and on later runs")
        if idx < len(targets) and not stop.asked:
            time.sleep(INTER_PAPER_WAIT)
    return n_ok, n_fail


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LLM second-rater coding runs (resumable, fault tolerant).")
    ap.add_argument("--status", action="store_true", help="offline progress report, no calls")
    ap.add_argument("--preflight", action="store_true", help="run checks and exit")
    ap.add_argument("--ping", action="store_true",
                    help="with --preflight: one tiny live call per model")
    ap.add_argument("--fresh", action="store_true",
                    help="archive existing results/failures before starting")
    ap.add_argument("--sweep-only", action="store_true", help="retry deferred cells only")
    ap.add_argument("--limit", type=int, default=0, help="code only the first N papers")
    ap.add_argument("--papers", default="", help="comma-separated paper_ids to code")
    args = ap.parse_args()

    work, noun = load_stimulus(STIMULUS_FILE, SUBSAMPLE_FILE)
    instrument = Instrument(noun)

    if args.fresh:
        archive_existing()

    rows = read_rows(RESULTS_FILE, RESULT_FIELDS)
    fail_rows = read_rows(FAILURES_FILE, FAILURE_FIELDS)
    assert_uncontaminated(rows, instrument, work)

    if args.status:
        print_status(work, instrument, rows, fail_rows)
        return
    if args.preflight:
        preflight(work, instrument, rows, fail_rows, args.ping)
        return
    if not API_KEY:
        raise SystemExit("FATAL: OPENROUTER_API_KEY is not set (env or experiment/.env).")

    if args.papers:
        keep = {p.strip() for p in args.papers.split(",") if p.strip()}
        unknown = keep - {w["paper_id"] for w in work}
        if unknown:
            raise SystemExit(f"FATAL: unknown paper_id(s): {', '.join(sorted(unknown))}")
        work = [w for w in work if w["paper_id"] in keep]
    if args.limit:
        work = work[:args.limit]
    scope = {w["paper_id"] for w in work}

    done = cells_done(rows, RUN_ID)
    deferred = cells_deferred(fail_rows, RUN_ID, done)
    print(f"[RUN {RUN_ID}] stimulus={STIMULUS_FILE.name} noun={noun!r} "
          f"instrument={instrument.sha}")
    print(f"[RUN {RUN_ID}] {len(done)} cells already valid, {len(deferred)} deferred, "
          f"{len(work) * len(MODELS) - len(done) - len(deferred)} never attempted")

    # Cells that have had no turn go first; cells deferred by an earlier
    # invocation follow in the same pass. A cell that keeps failing must not
    # delay one that has never been tried, but it must not need a sweep to be
    # picked up either -- that is what "retriable on a later run" means.
    fresh_targets, carryover_targets = [], []
    for paper in work:
        fresh = [m for m in MODELS
                 if (paper["paper_id"], m) not in done
                 and (paper["paper_id"], m) not in deferred]
        again = [m for m in MODELS if (paper["paper_id"], m) in deferred]
        if fresh and not args.sweep_only:
            fresh_targets.append((paper, fresh))
        if again:
            carryover_targets.append((paper, again))

    stop = Stop()
    client = make_client()
    ledger = Ledger(RESULTS_FILE, FAILURES_FILE)
    total_ok = total_fail = 0
    try:
        if fresh_targets:
            ok, bad = code_cells(client, instrument, ledger, fresh_targets, stop, "PAPER")
            total_ok, total_fail = total_ok + ok, total_fail + bad
        else:
            print("[RUN] no un-attempted cells"
                  + (" (--sweep-only)" if args.sweep_only else ""))
        if carryover_targets and not stop.asked:
            n = sum(len(ms) for _, ms in carryover_targets)
            print(f"\n[CARRYOVER] {n} cell(s) deferred by an earlier invocation")
            ok, bad = code_cells(client, instrument, ledger, carryover_targets,
                                 stop, "CARRYOVER")
            total_ok, total_fail = total_ok + ok, total_fail + bad

        # Sweep: every deferred cell gets another turn before the run ends.
        for sweep in range(1, SWEEP_PASSES + 1):
            if stop.asked:
                break
            done_now = cells_done(read_rows(RESULTS_FILE, RESULT_FIELDS), RUN_ID)
            pend_now = cells_deferred(read_rows(FAILURES_FILE, FAILURE_FIELDS),
                                      RUN_ID, done_now)
            pend_now = {k: v for k, v in pend_now.items() if k[0] in scope}
            if not pend_now:
                print("\n[SWEEP] nothing deferred - matrix complete for this run")
                break
            groups = [(p, [m for m in MODELS if (p["paper_id"], m) in pend_now])
                      for p in work]
            groups = [g for g in groups if g[1]]
            print(f"\n[SWEEP {sweep}/{SWEEP_PASSES}] retrying {len(pend_now)} "
                  f"deferred cell(s) after a {SWEEP_PAUSE:.0f}s pause")
            time.sleep(SWEEP_PAUSE)
            ok, bad = code_cells(client, instrument, ledger, groups, stop, f"SWEEP{sweep}")
            total_ok, total_fail = total_ok + ok, total_fail + bad
    finally:
        ledger.close()

    rows = read_rows(RESULTS_FILE, RESULT_FIELDS)
    fail_rows = read_rows(FAILURES_FILE, FAILURE_FIELDS)
    done = cells_done(rows, RUN_ID)
    deferred = {k: v for k, v in cells_deferred(fail_rows, RUN_ID, done).items()
                if k[0] in scope}
    print(f"\n[END] run {RUN_ID}: +{total_ok} cells coded, {total_fail} deferral events")
    print(f"[END] run {RUN_ID} now {len(done)}/{len(work) * len(MODELS)} valid cells "
          f"over the selected papers")
    if deferred:
        print(f"[END] {len(deferred)} cell(s) still deferred - re-run the same command "
              "to retry them:")
        for (pid, model), events in sorted(deferred.items()):
            print(f"       {pid:28} {model:9} {events[-1]['reason']} "
                  f"({len(events)} failed attempt-set(s))")
        print("[END] These cells are MISSING, not coded. Report them as missing; "
              "they must not enter the matrix as a category.")
    else:
        print("[END] no deferred cells")
    print(f"[END] results: {RESULTS_FILE.name}   failures: {FAILURES_FILE.name}")
    if stop.asked:
        sys.exit(130)


if __name__ == "__main__":
    main()
