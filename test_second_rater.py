"""test_second_rater.py - the coding runner's guarantees, exercised offline.

Run:  python test_second_rater.py       (no API key and no network needed)

Every claim the runner makes about contamination, resumability, fault tolerance
and quarantined failures is checked here against a stub client that plays back
scripted API behaviour. If a result in the paper depends on the runner having
behaved, this is the file that says it did.

No network. A stub client plays back a scripted sequence of API behaviours per
(model, call index): timeouts, 429s with Retry-After, empty completions,
truncation, garbage text, out-of-vocabulary codes, and fatal auth errors.
"""
import csv, importlib.util, json, os, sys, tempfile, types
from pathlib import Path

import httpx, openai

TMP = Path(tempfile.mkdtemp(prefix="sr_test_"))
SRC = Path(__file__).parent / "second_rater.py"

# ---- fixtures ---------------------------------------------------------------
PAPERS = [
    {"paper_id": "P1", "title": "First", "category": "A", "text": "alpha text one"},
    {"paper_id": "P2", "title": "Second", "category": "B", "text": "beta text two"},
]
(TMP / "stim.json").write_text(json.dumps(PAPERS))
with open(TMP / "sub.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["paper_id"]); w.writeheader()
    for p in PAPERS: w.writerow({"paper_id": p["paper_id"]})

def load(**env):
    for k, v in env.items(): os.environ[k] = str(v)
    os.environ["OPENROUTER_API_KEY"] = "sk-test"
    os.environ.setdefault("INTER_PAPER_WAIT", "0")
    os.environ.setdefault("BACKOFF_BASE", "0.01")
    os.environ.setdefault("BACKOFF_CAP", "0.01")
    os.environ.setdefault("RATE_LIMIT_WAIT", "0.01")
    os.environ.setdefault("SWEEP_PAUSE", "0")
    spec = importlib.util.spec_from_file_location(f"sr_{len(sys.modules)}", SRC)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

REQ = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")

def resp(status, headers=None):
    return httpx.Response(status, headers=headers or {}, request=REQ)

def ok_body(q1="KCO", q2="F2", q3="BIN", q4="OK"):
    return f"Q1: {q1} | s\nQ2: {q2} | s\nQ3: {q3} | s\nQ4: {q4} | s"

def completion(content, finish="stop", choices=True):
    msg = types.SimpleNamespace(content=content)
    ch = [types.SimpleNamespace(message=msg, finish_reason=finish)] if choices else []
    return types.SimpleNamespace(choices=ch,
                                 usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=20))

class Stub:
    """script: dict model_name -> list of behaviours, consumed in order."""
    def __init__(self, script, record=None):
        self.script, self.calls, self.record = script, {}, record if record is not None else []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))
    def _create(self, **kw):
        name = next(n for n, mid in MODELS_REF.items() if mid == kw["model"])
        i = self.calls.get(name, 0); self.calls[name] = i + 1
        self.record.append((name, kw["max_tokens"]))
        seq = self.script.get(name, ["ok"])
        beh = seq[i] if i < len(seq) else seq[-1]
        if beh == "ok":            return completion(ok_body())
        if beh == "timeout":       raise openai.APITimeoutError(request=REQ)
        if beh == "conn":          raise openai.APIConnectionError(request=REQ)
        if beh == "429":           raise openai.RateLimitError("rate", response=resp(429, {"retry-after": "0.01"}), body=None)
        if beh == "500":           raise openai.InternalServerError("boom", response=resp(500), body=None)
        if beh == "503":           raise openai.APIStatusError("unavail", response=resp(503), body=None)
        if beh == "402":           raise openai.APIStatusError("payment", response=resp(402), body=None)
        if beh == "401":           raise openai.AuthenticationError("bad key", response=resp(401), body=None)
        if beh == "empty":         return completion("")
        if beh == "trunc":         return completion("", finish="length")
        if beh == "nochoice":      return completion(None, choices=False)
        if beh == "garbage":       return completion("I cannot comply with this request.")
        if beh == "badcode":       return completion(ok_body(q1="KAPPA"))
        if beh == "partial":       return completion("Q1: KCO | s\nQ2: F2 | s")
        if beh == "messy":         return completion("**Q1:** KCO | s\nQ2 - F2 | s\nQ3: BIN | s\nQ4: OK | s")
        raise AssertionError(beh)

PASS = FAIL = 0
def check(label, cond, note=""):
    global PASS, FAIL
    if cond: PASS += 1; print(f"  PASS  {label}" + (f" - {note}" if note else ""))
    else:    FAIL += 1; print(f"  FAIL  {label}" + (f" - {note}" if note else ""))

def rows(p):
    return list(csv.DictReader(open(p))) if Path(p).exists() else []

# ---- 1. per-cell retry behaviour -------------------------------------------
print("\n== 1. retry / defer behaviour per cell ==")
m = load(STIMULUS_FILE=TMP/"stim.json", SUBSAMPLE_FILE=TMP/"sub.csv",
         RESULTS_FILE=TMP/"r1.csv", MAX_ATTEMPTS=4)
MODELS_REF = m.MODELS
work, noun = m.load_stimulus(m.STIMULUS_FILE, m.SUBSAMPLE_FILE)
inst = m.Instrument(noun)
paper = work[0]

cases = [
    ("transient timeout then success", ["timeout", "ok"],          True,  None, 2),
    ("connection drop then success",   ["conn", "ok"],             True,  None, 2),
    ("429 with Retry-After then ok",   ["429", "ok"],              True,  None, 2),
    ("500 then success",               ["500", "ok"],              True,  None, 2),
    ("503 then success",               ["503", "ok"],              True,  None, 2),
    ("persistent timeout -> defer",    ["timeout"]*4,              False, "APITIMEOUTERROR", 4),
    ("empty then success",             ["empty", "ok"],            True,  None, 2),
    ("always empty -> defer",          ["empty"]*4,                False, "EMPTY", 4),
    ("200 no choices then ok",         ["nochoice", "ok"],         True,  None, 2),
    ("garbage then success",           ["garbage", "ok"],          True,  None, 2),
    ("always garbage -> defer",        ["garbage"]*4,              False, "PARSE", 4),
    ("out-of-vocab code -> defer",     ["badcode"]*4,              False, "PARSE", 4),
    ("partial code lines -> defer",    ["partial"]*4,              False, "PARSE", 4),
    ("messy formatting parses",        ["messy"],                  True,  None, 1),
    ("402 not retried, deferred",      ["402"]*4,                  False, "HTTP_402", 4),
]
for label, seq, want_ok, want_reason, want_att in cases:
    rec = []
    out = m.code_one_cell(Stub({"gemma": seq}, rec), inst, "gemma", m.MODELS["gemma"], paper)
    detail = f"ok={out.ok} reason={out.reason or '-'} attempts={out.attempts} calls={len(rec)}"
    good = out.ok == want_ok and (want_reason is None or out.reason == want_reason)
    if want_ok: good = good and out.attempts == want_att
    if label.startswith("402"): good = good and len(rec) == 1   # not retried
    check(label, good, detail)

print("\n== 1b. truncation escalates the output budget ==")
rec = []
out = m.code_one_cell(Stub({"nemotron": ["trunc", "trunc", "ok"]}, rec), inst,
                      "nemotron", m.MODELS["nemotron"], paper)
budgets = [b for _, b in rec]
check("escalates 8000 -> 16000 -> cap, then succeeds",
      out.ok and budgets == [8000, 16000, 16000], f"budgets={budgets}")
rec = []
out = m.code_one_cell(Stub({"nemotron": ["trunc"]*4}, rec), inst,
                      "nemotron", m.MODELS["nemotron"], paper)
budgets2 = [b for _, b in rec]
check("re-rolls at the cap, bounded by MAX_ATTEMPTS",
      not out.ok and out.reason == "TRUNCATED" and budgets2 == [8000, 16000, 16000, 16000],
      f"reason={out.reason} budgets={budgets2}")

print("\n== 1c. fatal config error aborts, does not spin ==")
rec = []
try:
    m.code_one_cell(Stub({"gemma": ["401"]*4}, rec), inst, "gemma", m.MODELS["gemma"], paper)
    check("401 raises SystemExit", False, "no exception")
except SystemExit as e:
    check("401 raises SystemExit after 1 call", len(rec) == 1 and "Authentication" in str(e),
          f"calls={len(rec)}")

# ---- 2. end-to-end: failures quarantined, results clean, resume works ------
print("\n== 2. end-to-end run with one permanently empty model ==")
R2 = TMP / "r2.csv"
m2 = load(STIMULUS_FILE=TMP/"stim.json", SUBSAMPLE_FILE=TMP/"sub.csv",
          RESULTS_FILE=R2, RUN_ID="1", MAX_ATTEMPTS=2, SWEEP_PASSES=2)
MODELS_REF = m2.MODELS
m2.make_client = lambda: Stub({"qwen": ["empty"]*50})
sys.argv = ["second_rater.py"]
m2.main()
res, fails = rows(R2), rows(R2.with_suffix(".failures.csv"))
check("only valid cells in results", len(res) == 4 and all(m2.valid_result(r) for r in res),
      f"{len(res)} rows: {[(r['paper_id'], r['model']) for r in res]}")
check("no PARSE_ERR / ERROR text in results",
      not any("ERR" in r["Q4"] or r["raw"].startswith("ERROR") for r in res))
check("failing model never appears in results", all(r["model"] != "qwen" for r in res))
check("deferred cells recorded in the failure ledger",
      len(fails) == 6 and {f["model"] for f in fails} == {"qwen"},
      f"{len(fails)} events (1 main + 2 sweeps) x 2 papers")
check("every result row carries stimulus + instrument sha",
      all(r["instrument_sha"] == m2.Instrument("full text").sha and len(r["stimulus_sha"]) == 16
          for r in res))
check("attempts and latency recorded", all(r["attempts"] and r["latency_s"] for r in res))

print("\n== 2b. re-run resumes: only the deferred cells are retried ==")
m3 = load(STIMULUS_FILE=TMP/"stim.json", SUBSAMPLE_FILE=TMP/"sub.csv",
          RESULTS_FILE=R2, RUN_ID="1", MAX_ATTEMPTS=2, SWEEP_PASSES=0)
MODELS_REF = m3.MODELS
rec = []
m3.make_client = lambda: Stub({"qwen": ["ok"]*50}, rec)
sys.argv = ["second_rater.py"]
m3.main()
res2 = rows(R2)
check("deferred cells retried on a new invocation with SWEEP_PASSES=0", len(rec) == 2 and all(n == "qwen" for n, _ in rec),
      f"calls={rec}")
check("matrix now complete, no duplicate cells",
      len(res2) == 6 and len({(r['paper_id'], r['model']) for r in res2}) == 6)
check("previously coded cells were not re-called",
      sum(1 for r in res2 if r["model"] == "gemma") == 2)

print("\n== 2c. a later run with a new RUN_ID starts empty but keeps run 1 ==")
m4 = load(STIMULUS_FILE=TMP/"stim.json", SUBSAMPLE_FILE=TMP/"sub.csv",
          RESULTS_FILE=R2, RUN_ID="2", MAX_ATTEMPTS=2, SWEEP_PASSES=0)
MODELS_REF = m4.MODELS
rec = []
m4.make_client = lambda: Stub({}, rec)
sys.argv = ["second_rater.py"]
m4.main()
res3 = rows(R2)
check("run 2 coded all 6 cells without touching run 1",
      len(rec) == 6 and sum(1 for r in res3 if r["run"] == "1") == 6
      and sum(1 for r in res3 if r["run"] == "2") == 6, f"{len(res3)} rows total")

# ---- 3. contamination guards -----------------------------------------------
print("\n== 3. contamination guards ==")
PAPERS2 = [dict(p) for p in PAPERS]
PAPERS2[0]["text"] = "alpha text one MODIFIED"
(TMP / "stim2.json").write_text(json.dumps(PAPERS2))
m5 = load(STIMULUS_FILE=TMP/"stim2.json", SUBSAMPLE_FILE=TMP/"sub.csv", RESULTS_FILE=R2)
try:
    w5, n5 = m5.load_stimulus(m5.STIMULUS_FILE, m5.SUBSAMPLE_FILE)
    m5.assert_uncontaminated(rows(R2), m5.Instrument(n5), w5)
    check("edited stimulus text is refused", False, "no abort")
except SystemExit as e:
    check("edited stimulus text is refused", "different stimulus text" in str(e), "aborted")

m6 = load(STIMULUS_FILE=TMP/"stim.json", SUBSAMPLE_FILE=TMP/"sub.csv",
          RESULTS_FILE=R2, STIMULUS_NOUN="excerpt")
try:
    w6, n6 = m6.load_stimulus(m6.STIMULUS_FILE, m6.SUBSAMPLE_FILE)
    m6.assert_uncontaminated(rows(R2), m6.Instrument(n6), w6)
    check("changed instrument wording is refused", False, "no abort")
except SystemExit as e:
    check("changed instrument wording is refused", "different" in str(e) and "instrument" in str(e))

PAPERS3 = PAPERS + [{"paper_id": "P3", "title": "T", "category": "C", "text": ""}]
(TMP / "stim3.json").write_text(json.dumps(PAPERS3))
with open(TMP / "sub3.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["paper_id"]); w.writeheader()
    for p in PAPERS3: w.writerow({"paper_id": p["paper_id"]})
m7 = load(STIMULUS_FILE=TMP/"stim3.json", SUBSAMPLE_FILE=TMP/"sub3.csv", RESULTS_FILE=TMP/"r7.csv")
try:
    m7.load_stimulus(m7.STIMULUS_FILE, m7.SUBSAMPLE_FILE)
    check("empty stimulus text is refused", False, "no abort")
except SystemExit as e:
    check("empty stimulus text is refused", "empty stimulus" in str(e))

with open(TMP / "sub4.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["paper_id"]); w.writeheader()
    w.writerow({"paper_id": "P1"}); w.writerow({"paper_id": "P_MISSING"})
m8 = load(STIMULUS_FILE=TMP/"stim.json", SUBSAMPLE_FILE=TMP/"sub4.csv", RESULTS_FILE=TMP/"r8.csv")
try:
    m8.load_stimulus(m8.STIMULUS_FILE, m8.SUBSAMPLE_FILE)
    check("subsample paper absent from stimulus is refused", False, "no abort")
except SystemExit as e:
    check("subsample paper absent from stimulus is refused", "absent from" in str(e))

# an out-of-vocabulary code already in a results file must not count as done
bad = rows(R2)[:1][0].copy(); bad["Q4"] = "PARSE_ERR"; bad["paper_id"] = "P1"; bad["model"] = "zzz"
with open(R2, "a", newline="") as fh:
    csv.DictWriter(fh, fieldnames=m4.RESULT_FIELDS).writerow(bad)
check("invalid row does not count as a completed cell",
      ("P1", "zzz") not in m4.cells_done(rows(R2), "1"))

print(f"\n{'='*58}\n{PASS} passed, {FAIL} failed\n{'='*58}")
sys.exit(1 if FAIL else 0)
