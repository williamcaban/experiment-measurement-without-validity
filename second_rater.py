#!/usr/bin/env python3
"""
second_rater.py — OpenRouter multi-model IRR coding experiment
Runs 3 LLMs (Nemotron, Gemma, Qwen) over 20 paper excerpts using
the 4-dimension IRR coding instrument.

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  pip install openai
  python second_rater.py

Rate limit design:
  - Free tier: 20 RPM global, 1000 RPD (with $10 credit purchase)
  - 3 concurrent calls per paper (ThreadPoolExecutor)
  - 10-second mandatory sleep between papers → effective ~18 RPM
  - Total runtime: ~15-25 minutes for 20 papers
"""

import json
import csv
import time
import os
import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────
# Load .env if present (key stored in experiment/.env, gitignored)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

API_KEY  = os.environ.get("OPENROUTER_API_KEY", "sk-or-YOUR_KEY_HERE")
BASE_URL = "https://openrouter.ai/api/v1"

MODELS = {
    "nemotron": "nvidia/nemotron-3-ultra-550b-a55b",        # NVIDIA — 550B MoE
    "gemma":    "google/gemma-4-31b-it",                    # Google — 31B
    "qwen":     "qwen/qwen3-next-80b-a3b-instruct",         # Alibaba — 80B MoE
}

INTER_PAPER_WAIT = 12   # seconds between papers (keeps RPM ≤ 15)
MAX_RETRIES      = 3    # retries per call on error
RETRY_WAIT       = 65   # seconds to wait after a 429 (rate limit hit)
MAX_TOKENS       = 1500 # Nemotron needs room for CoT before final output

EXCERPTS_FILE    = Path(__file__).parent / "excerpts.json"
RESULTS_FILE     = Path(__file__).parent / "results.csv"

# ── Coding instrument ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a research methodology analyst coding papers for a systematic
review of inter-rater reliability (IRR) practices in AI evaluation research.

Your task is to read the paper excerpt and answer 4 structured coding questions.
Base your codes ONLY on what is explicitly stated in the text provided.
Do not infer or assume practices not mentioned. If uncertain, choose the most
conservative code and note it in your justification."""

CODING_INSTRUMENT = """
CODING INSTRUMENT

For each question, choose EXACTLY ONE code from the list. Output only the 3-letter code,
not the question text.

Q1 CODES — IRR metric reported:
  KCO = Cohen's kappa explicitly named
  KFL = Fleiss' kappa explicitly named
  KAL = Krippendorff's alpha explicitly named
  ICC = Intraclass correlation coefficient
  PA  = Percentage/proportion agreement only (no kappa or alpha)
  NIL = No recognized IRR metric (automated scoring, ELO, Pearson, or nothing)

Q2 CODES — Rater design:
  F2  = Exactly 2 raters, same fixed pair on every item
  FN  = Fixed panel of 3+ raters scoring every item
  RV  = Rotating/varying rater identity across items
  CS  = Crowdsourcing platform or open-ended user voting pool
  UNK = Not described, or paper has no rater design

Q3 CODES — Measurement scale:
  BIN = Binary (pass/fail, yes/no, 0/1, A vs B)
  NOM = Nominal unordered categories
  ORD = Ordered/ordinal (1-5 Likert, ternary, rubric)
  CON = Continuous or interval (numeric averages, probabilities)
  UNK = Not described, or paper has no rating scale

Q4 CODES — Structural validity (apply rules in this order, stop at first match):
  ABS = Q1 is NIL (no recognized IRR metric reported)
  MM  = Q1=KCO but Q2 is not F2; OR Q1=KCO/KFL but Q3 is ORD or CON
  INC = Q1=PA; OR correct metric family but no threshold and no rationale stated
  OK  = Metric fits Q2 and Q3, threshold stated or implied, rationale provided

STRICT RULES:
- If Q1=NIL → Q4=ABS (no exceptions)
- Check MM before INC

REQUIRED OUTPUT FORMAT — output ONLY these 4 lines, nothing before them:
Q1: KCO | <one sentence from excerpt>
Q2: F2 | <one sentence from excerpt>
Q3: BIN | <one sentence from excerpt>
Q4: OK | <one sentence of justification>

Replace KCO/F2/BIN/OK with the actual codes for THIS paper.
"""

def build_user_prompt(paper_id: str, title: str, excerpt: str) -> str:
    return (
        f"{CODING_INSTRUMENT}\n\n"
        f"PAPER EXCERPT [{paper_id}] — {title}\n"
        f"{'='*60}\n"
        f"{excerpt}\n"
        f"{'='*60}\n\n"
        "Apply the coding instrument to this excerpt. Output exactly 4 lines."
    )

# ── API call with retry ───────────────────────────────────────────────────────
def call_model(client: OpenAI, model_id: str, model_name: str,
               paper_id: str, title: str, excerpt: str) -> str:
    extra_body = {}
    max_tokens = MAX_TOKENS
    if model_name == "nemotron":
        # Nemotron reasons before answering. The enable_thinking=False flag is
        # passed but is NOT honored by the OpenRouter provider for this model
        # (verified 2026-09-28: a 3000-token budget returned reasoning_tokens=3000,
        # finish_reason=length and empty content on one excerpt). The budget is
        # therefore set high enough for reasoning plus the 4-line answer; the
        # flag is retained so the request is identical to earlier runs.
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        max_tokens = 8000
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(paper_id, title, excerpt)},
                ],
                temperature=0,    # deterministic — required for reproducibility
                max_tokens=max_tokens,
                extra_body=extra_body or None,
            )
            content = resp.choices[0].message.content
            if content is None or not content.strip():
                # Empty completion (e.g. provider returned only a refusal/finish_reason
                # with no text). Treat as retryable rather than crashing the parser.
                print(f"    [{model_name}] Empty completion (attempt {attempt+1}) — "
                      f"finish_reason={getattr(resp.choices[0], 'finish_reason', '?')}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(5)
                    continue
                return "ERROR: empty completion after max retries"
            return content.strip()
        except Exception as exc:
            err = str(exc)
            if "429" in err or "rate limit" in err.lower():
                print(f"    [{model_name}] 429 — sleeping {RETRY_WAIT}s (attempt {attempt+1})")
                time.sleep(RETRY_WAIT)
            elif attempt < MAX_RETRIES - 1:
                print(f"    [{model_name}] Error (attempt {attempt+1}): {exc}")
                time.sleep(5)
            else:
                print(f"    [{model_name}] Failed after {MAX_RETRIES} attempts: {exc}")
                return f"ERROR: {exc}"
    return "ERROR: max retries exceeded"

# ── Parse codes from model output ─────────────────────────────────────────────
def parse_codes(raw: str) -> dict:
    result = {"Q1": "PARSE_ERR", "Q2": "PARSE_ERR",
              "Q3": "PARSE_ERR", "Q4": "PARSE_ERR", "raw": raw}
    for line in raw.splitlines():
        line = line.strip()
        for q in ("Q1", "Q2", "Q3", "Q4"):
            if line.startswith(f"{q}:"):
                parts = line.split("|", 1)
                code = parts[0].replace(f"{q}:", "").strip()
                result[q] = code
                break
    return result

# ── Main experiment ───────────────────────────────────────────────────────────
def main():
    if not RESULTS_FILE.parent.exists():
        RESULTS_FILE.parent.mkdir(parents=True)

    excerpts = json.loads(EXCERPTS_FILE.read_text())
    # excerpts.json carries verbatim excerpts for all 55 scanned papers; the
    # reliability study codes only the 20-paper stratified subsample listed in
    # author_codes.csv.
    with open(Path(__file__).parent / "author_codes.csv", newline="") as fh:
        subsample = {row["paper_id"] for row in csv.DictReader(fh)}
    excerpts = [e for e in excerpts if e["paper_id"] in subsample]
    assert len(excerpts) == 20, f"expected 20 subsample excerpts, found {len(excerpts)}"

    # Resume: skip already-coded papers
    done: set[str] = set()
    if RESULTS_FILE.exists() and RESULTS_FILE.stat().st_size > 0:
        with open(RESULTS_FILE) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(f"{row['paper_id']}|{row['model']}")

    fieldnames = ["paper_id", "category", "model",
                  "Q1", "Q2", "Q3", "Q4", "raw", "timestamp"]

    write_header = not RESULTS_FILE.exists() or RESULTS_FILE.stat().st_size == 0
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for paper in excerpts:
            pid     = paper["paper_id"]
            title   = paper["title"]
            excerpt = paper["excerpt"]
            cat     = paper["category"]

            # Check if all models already coded this paper
            already_done = all(f"{pid}|{m}" in done for m in MODELS)
            if already_done:
                print(f"[SKIP] {pid} — all 3 models already coded")
                continue

            print(f"\n[PAPER {cat}] {pid}")
            print(f"  Title: {title[:70]}...")

            # Fire all 3 models concurrently for this paper
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                future_to_model = {
                    pool.submit(
                        call_model, client, model_id, model_name,
                        pid, title, excerpt
                    ): model_name
                    for model_name, model_id in MODELS.items()
                    if f"{pid}|{model_name}" not in done
                }

                for future in concurrent.futures.as_completed(future_to_model):
                    model_name = future_to_model[future]
                    raw        = future.result()
                    codes      = parse_codes(raw)
                    ts         = datetime.now(timezone.utc).isoformat()

                    row = {
                        "paper_id":  pid,
                        "category":  cat,
                        "model":     model_name,
                        "Q1":        codes["Q1"],
                        "Q2":        codes["Q2"],
                        "Q3":        codes["Q3"],
                        "Q4":        codes["Q4"],
                        "raw":       codes["raw"].replace("\n", " // "),
                        "timestamp": ts,
                    }
                    writer.writerow(row)
                    f.flush()

                    q4_ok = codes["Q4"] != "PARSE_ERR"
                    icon  = "✓" if q4_ok else "✗"
                    print(f"  {icon} [{model_name:10}] Q1={codes['Q1']} "
                          f"Q2={codes['Q2']} Q3={codes['Q3']} Q4={codes['Q4']}")

            print(f"  [WAIT] {INTER_PAPER_WAIT}s before next paper...")
            time.sleep(INTER_PAPER_WAIT)

    print(f"\n[DONE] Results saved to {RESULTS_FILE}")
    print("Next step: run compute_alpha.py to calculate Krippendorff's α")

if __name__ == "__main__":
    main()
