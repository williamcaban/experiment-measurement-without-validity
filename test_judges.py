#!/usr/bin/env python3
"""
test_judges.py - regression tests for the judge manifest and its preflight gate.

WHY THESE TESTS EXIST
---------------------
With full documents as the stimulus, a judge served behind too small a context
window truncates the largest papers SILENTLY. Nothing raises, nothing fails to
parse, and the model returns a confident code on a partial paper. That is the
worst failure mode available in this experiment because it is invisible in the
output - exactly the shape of the bug that let a truncated extraction hide
G6's Fleiss' kappa in an appendix.

So the gate must fail closed. These tests assert that it does, by corrupting a
copy of the manifest three ways and requiring a non-zero exit each time. They
exist so that a later change cannot relax the gate and leave the suite green.

Offline except for test group 4, which is skipped without a network.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST = HERE / "judge_manifest.json"
STIM = "stimulus_fulltext.json"

passed = failed = 0


def check(label: str, ok: bool, note: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"  PASS  {label}" + (f" - {note}" if note else ""))
    else:
        failed += 1
        print(f"  FAIL  {label}" + (f" - {note}" if note else ""))


def preflight_exit(tmp_results: str) -> int:
    """Run --preflight against the current manifest; return its exit code."""
    env = dict(os.environ, STIMULUS_FILE=STIM, RESULTS_FILE=tmp_results)
    r = subprocess.run([sys.executable, str(HERE / "second_rater.py"), "--preflight"],
                       cwd=HERE, env=env, capture_output=True, text=True)
    return r.returncode


def with_corrupted(mutate) -> int:
    """Apply `mutate` to the manifest, run preflight, always restore."""
    backup = MANIFEST.read_bytes()
    try:
        man = json.loads(backup)
        mutate(man)
        MANIFEST.write_text(json.dumps(man, indent=2))
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
            tmp = fh.name
        try:
            return preflight_exit(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)
    finally:
        MANIFEST.write_bytes(backup)


print("== 1. manifest shape ==")
check("judge_manifest.json exists", MANIFEST.exists(),
      "run: python3 judges.py --build")
if not MANIFEST.exists():
    sys.exit("cannot continue without the manifest")

man = json.loads(MANIFEST.read_text())
check("schema tag present", man.get("schema") == "knosys-judge-manifest/1",
      man.get("schema"))
for field in ("built_utc", "instrument_sha", "stimulus_generation",
              "judges", "replication"):
    check(f"top-level field {field!r}", field in man)
rep = man.get("replication", {})
for field in ("min_context_tokens_required", "derivation", "requirement",
              "providers_below_requirement", "pinned_quantizations",
              "quantization_requirement"):
    check(f"replication field {field!r}", field in rep)

sys.path.insert(0, str(HERE))
import second_rater as sr  # noqa: E402

check("manifest judge set == runner MODELS",
      set(man["judges"]) == set(sr.MODELS),
      f"manifest={sorted(man['judges'])} runner={sorted(sr.MODELS)}")
for name, slug in sr.MODELS.items():
    check(f"{name} slug matches manifest",
          man["judges"].get(name, {}).get("slug") == slug, slug)

print("\n== 2. the quantization pin is real and survivable ==")
check("runner pins match the manifest",
      {k: v.get("pinned_quantization") for k, v in man["judges"].items()}
      == dict(sr.PIN_QUANT),
      f"manifest={ {k: v.get('pinned_quantization') for k, v in man['judges'].items()} } "
      f"runner={dict(sr.PIN_QUANT)}")
check("replication block records the pins",
      rep["pinned_quantizations"] == dict(sr.PIN_QUANT), rep["pinned_quantizations"])
for name, j in man["judges"].items():
    pin = j.get("pinned_quantization")
    check(f"{name} is pinned", bool(pin), str(pin))
    check(f"{name} pin is offered by the model",
          pin in j["quantizations_offered"],
          f"{pin} in {j['quantizations_offered']}")
    # Endpoint count overstates resilience; one provider's endpoints fail
    # together, so the pool that matters is distinct provider names.
    check(f"{name} pin has >=2 distinct providers",
          j.get("n_providers_at_pin", 0) >= 2,
          f"{j.get('n_providers_at_pin')} ({', '.join(j.get('providers_at_pin', []))})")
    check(f"{name} every endpoint at the pin clears the requirement",
          all(e["context_length"] >= rep["min_context_tokens_required"]
              for e in j["endpoints_at_pin"] if e["context_length"]),
          f"min {min((e['context_length'] for e in j['endpoints_at_pin'] if e['context_length']), default=0):,}")
    check(f"{name} measured through its pin",
          j["measured"]["served_by"] in j.get("providers_at_pin", []),
          f"measured via {j['measured']['served_by']}, pool "
          f"{j.get('providers_at_pin')}")

print("\n== 3. the requirement is actually derived, not asserted ==")
req = rep["min_context_tokens_required"]
worst = max(j["measured"]["prompt_tokens"] for j in man["judges"].values())
check("requirement exceeds the largest measured prompt", req > worst,
      f"{req:,} > {worst:,}")
check("requirement leaves room for the output budget",
      req - worst >= sr.MAX_TOKENS_CAP,
      f"headroom {req - worst:,} vs output cap {sr.MAX_TOKENS_CAP:,}")
check("no provider is below the requirement",
      rep["providers_below_requirement"] == [],
      f"{len(rep['providers_below_requirement'])} listed")
for name, j in man["judges"].items():
    check(f"{name} every provider clears the requirement",
          j["provider_context_min"] >= req,
          f"min {j['provider_context_min']:,} vs {req:,}")
    check(f"{name} measured under its own tokenizer",
          3.0 < j["measured"]["chars_per_token"] < 6.0,
          f"{j['measured']['chars_per_token']} chars/token")

print("\n== 4. the gate fails closed ==")
check("a provider below the requirement blocks the run",
      with_corrupted(lambda m: m["judges"]["gemma"].update(
          provider_context_min=32768)) == 1)
check("a stimulus larger than the requirement blocks the run",
      with_corrupted(lambda m: m["replication"].update(
          min_context_tokens_required=5000)) == 1)
check("a changed model slug blocks the run",
      with_corrupted(lambda m: m["judges"]["qwen"].update(
          slug="qwen/not-the-same-model")) == 1)
check("a judge missing from the manifest blocks the run",
      with_corrupted(lambda m: m["judges"].pop("gemma")) == 1)
check("a quantization pin that disagrees with the runner blocks the run",
      with_corrupted(lambda m: m["judges"]["gemma"].update(
          pinned_quantization="bf16")) == 1)
check("a pin with no fallback pool blocks the run",
      with_corrupted(lambda m: m["judges"]["qwen"].update(
          n_providers_at_pin=1)) == 1)
with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
    _t = fh.name
try:
    check("an absent manifest blocks the run", (lambda: (
        shutil.move(MANIFEST, MANIFEST.with_suffix(".json.hidden")),
        preflight_exit(_t),
        shutil.move(MANIFEST.with_suffix(".json.hidden"), MANIFEST),
    )[1])() == 1)
finally:
    Path(_t).unlink(missing_ok=True)
    if MANIFEST.with_suffix(".json.hidden").exists():
        shutil.move(MANIFEST.with_suffix(".json.hidden"), MANIFEST)
check("the intact manifest passes", with_corrupted(lambda m: None) == 0)

print("\n== 5. serving provenance is recorded per cell ==")
for field in ("provider", "generation_id", "pinned_quantizations"):
    check(f"RESULT_FIELDS carries {field!r}", field in sr.RESULT_FIELDS)
check("provider is not in FAILURE_FIELDS by accident",
      "provider" not in sr.FAILURE_FIELDS,
      "a failed cell has no serving provider")
check("quantization pin is a per-judge mapping",
      isinstance(sr.PIN_QUANT, dict) and set(sr.PIN_QUANT) <= set(sr.MODELS),
      str(dict(sr.PIN_QUANT)))

print("\n" + "=" * 58)
print(f"{passed} passed, {failed} failed")
print("=" * 58)
sys.exit(1 if failed else 0)
