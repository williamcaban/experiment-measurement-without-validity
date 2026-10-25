#!/usr/bin/env python3
"""
judges.py - record the identity of the three LLM raters as verifiable attributes.

WHY THIS FILE EXISTS
--------------------
A model slug is not a rater. The same slug on OpenRouter is served by many
providers that differ in two ways that change what the rater can see and how it
computes:

  * CONTEXT WINDOW. The slug advertises a model-level context length, but each
    provider serves its own. For google/gemma-4-31b-it the providers range from
    131,072 to 262,144 tokens. A replicator who runs the same slug behind a
    smaller window gets a silently truncated stimulus, which is a different
    experiment with the same name. Because the stimulus here is the full
    document (the largest paper is ~59.5k tokens), the context window is a
    material property of the rater and is recorded as such.

  * QUANTIZATION. The same slug is served at fp4, fp8, bf16 and "unknown"
    depending on provider. Numerically these are different functions.

This script freezes both into judge_manifest.json so that (a) every run asserts
its serving configuration against the frozen record, and (b) a replicator can
check their own configuration before spending anything.

USAGE
  python3 judges.py --build     query the live API, measure, write the manifest
  python3 judges.py --verify    re-query and diff against the frozen manifest
  python3 judges.py --show      print the frozen manifest as a table

--build makes three live calls (one per judge) to measure the true prompt-token
count of the largest stimulus under each model's own tokenizer. Character-count
estimates are not used for the requirement: tokenizers differ (measured here at
3.93-4.01 chars/token across the three) and the floor must be exact.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import corpus

HERE = Path(__file__).parent
MANIFEST = HERE / "judge_manifest.json"
API = "https://openrouter.ai/api/v1"

# Must match second_rater.py MODELS exactly; asserted by --verify.
JUDGES = {
    "nemotron": "nvidia/nemotron-3-ultra-550b-a55b",
    "gemma":    "google/gemma-4-31b-it",
    "qwen":     "qwen/qwen3-next-80b-a3b-instruct",
}

# Head-room above the measured maximum prompt, covering the output budget the
# runner may escalate to plus a margin for instrument edits.
OUTPUT_BUDGET = 16384
MARGIN = 8192

# Quantization pin, per judge. The same slug served at a different quantization
# is a different function, so a judge that is allowed to drift between fp4 and
# bf16 between cells is not one rater. Each pin is chosen for the widest
# fallback pool at a single quantization, because pinning to a quantization
# served by one provider trades a numerical confound for an availability
# failure. Asserted against the live endpoint table by --build and --verify.
PINNED_QUANT = {
    "nemotron": "fp4",   # DeepInfra x2 + BaseTen x2; fp8 is Venice alone
    "gemma":    "fp8",   # DeepInfra x3 + Parasail + SiliconFlow
    "qwen":     "fp8",   # DeepInfra x3 + Parasail
}
MIN_PROVIDERS_AT_PIN = 2


def load_env() -> None:
    env = HERE / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "judges.py"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def model_table() -> dict:
    """Model-level attributes, keyed by slug."""
    out = {}
    for m in get(f"{API}/models")["data"]:
        if m["id"] in JUDGES.values():
            tp = m.get("top_provider") or {}
            out[m["id"]] = {
                "context_length": m.get("context_length"),
                "tokenizer": (m.get("architecture") or {}).get("tokenizer"),
                "top_provider_context_length": tp.get("context_length"),
                "top_provider_max_completion_tokens": tp.get("max_completion_tokens"),
            }
    return out


def provider_table(slug: str) -> list[dict]:
    """Every provider serving this slug, with its own context and quantization."""
    data = get(f"{API}/models/{slug}/endpoints").get("data") or {}
    rows = []
    for e in data.get("endpoints") or []:
        rows.append({
            "provider": e.get("provider_name"),
            "context_length": e.get("context_length"),
            "max_completion_tokens": e.get("max_completion_tokens"),
            "quantization": e.get("quantization"),
        })
    return rows


def largest_stimulus() -> tuple[str, str]:
    """(paper_id, text) of the largest paper IN THE ACTUAL STIMULUS FILE.

    Read from the stimulus extract.py produced, not re-derived with some other
    tool. An earlier version of this function shelled out to pdftotext, which
    yields text 0.7% smaller than the pinned PyMuPDF engine the stimulus is
    built with - so the context floor was being derived from a document the
    raters would never see. The floor has to come from the bytes actually sent.
    """
    stim = HERE / "stimulus_fulltext.json"
    if not stim.exists():
        sys.exit(f"FATAL: {stim.name} not found; run: python3 extract.py --build")
    recs = json.loads(stim.read_text())
    best = max(recs, key=lambda r: len(r.get("text") or ""))
    return best["paper_id"], best["text"]


def measure(slug: str, prompt: str, pin: str | None = None) -> dict:
    """True prompt_tokens for this slug's own tokenizer, via a 1-token call.

    Measured THROUGH the pin, so the figure comes from a provider the run will
    actually use rather than from whichever endpoint routing happened to pick.
    """
    from openai import OpenAI
    cli = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=API,
                 timeout=300.0, max_retries=0)
    extra = {"provider": {"quantizations": [pin]}} if pin else None
    r = cli.chat.completions.create(model=slug, max_tokens=1, temperature=0,
                                    messages=[{"role": "user", "content": prompt}],
                                    extra_body=extra)
    return {
        "prompt_tokens": r.usage.prompt_tokens,
        "served_by": getattr(r, "provider", None),
    }


def build() -> None:
    load_env()
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("FATAL: OPENROUTER_API_KEY is not set (env or experiment/.env).")

    print("resolving largest stimulus in the corpus ...")
    paper_id, text = largest_stimulus()
    print(f"  {paper_id}: {len(text)} chars\n")

    # The instrument wraps the stimulus, so the requirement must include it.
    sys.path.insert(0, str(HERE))
    import second_rater as sr
    instrument = sr.Instrument("full text")
    prompt = instrument.user_prompt(paper_id, "(title)", text)
    overhead = len(prompt) - len(text)
    print(f"instrument overhead: {overhead} chars "
          f"(sha {instrument.sha})\n")

    models = model_table()
    judges = {}
    for name, slug in JUDGES.items():
        print(f"{name} ({slug})")
        attrs = dict(models.get(slug, {}))
        providers = provider_table(slug)
        ctxs = [p["context_length"] for p in providers if p["context_length"]]
        quants = sorted({p["quantization"] for p in providers if p["quantization"]})
        pin = PINNED_QUANT.get(name)
        at_pin = [p for p in providers if p["quantization"] == pin]
        # Endpoint count overstates resilience: one provider often serves
        # several endpoints (DeepInfra serves three of gemma's five at fp8) and
        # they fail together. Distinct provider names are the real pool.
        distinct = sorted({p["provider"] for p in at_pin if p["provider"]})
        m = measure(slug, prompt, pin)
        attrs.update({
            "pinned_quantization": pin,
            "endpoints_at_pin": at_pin,
            "n_endpoints_at_pin": len(at_pin),
            "providers_at_pin": distinct,
            "n_providers_at_pin": len(distinct),
            "slug": slug,
            "providers": providers,
            "n_providers": len(providers),
            "provider_context_min": min(ctxs) if ctxs else None,
            "provider_context_max": max(ctxs) if ctxs else None,
            "quantizations_offered": quants,
            "measured": {
                "largest_paper": paper_id,
                "prompt_chars": len(prompt),
                "prompt_tokens": m["prompt_tokens"],
                "chars_per_token": round(len(prompt) / m["prompt_tokens"], 3),
                "served_by": m["served_by"],
            },
        })
        judges[name] = attrs
        print(f"  model context      : {attrs.get('context_length')}")
        print(f"  providers          : {len(providers)} "
              f"(context {attrs['provider_context_min']}-{attrs['provider_context_max']})")
        print(f"  quantizations      : {', '.join(quants) or 'n/a'}")
        print(f"  pinned quantization: {pin or '(none)'} -> "
              f"{len(at_pin)} endpoint(s) across {len(distinct)} provider(s)"
              + (f" ({', '.join(distinct)})" if distinct else ""))
        print(f"  measured prompt    : {m['prompt_tokens']} tokens "
              f"via {m['served_by']}\n")
        if pin and len(distinct) < MIN_PROVIDERS_AT_PIN:
            print(f"  !! only {len(distinct)} provider serves {pin}; "
                  f"an outage will stall this judge\n")

    worst = max(j["measured"]["prompt_tokens"] for j in judges.values())
    required = worst + OUTPUT_BUDGET + MARGIN

    below = []
    for name, j in judges.items():
        for p in j["providers"]:
            if p["context_length"] and p["context_length"] < required:
                below.append({"judge": name, "provider": p["provider"],
                              "context_length": p["context_length"],
                              "quantization": p["quantization"]})

    manifest = {
        "schema": "knosys-judge-manifest/1",
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "instrument_sha": instrument.sha,
        "stimulus_generation": "full document (bibliography retained)",
        "stimulus_source": "stimulus_fulltext.json via extract.py + corpus.py "
                           f"(pymupdf {corpus.PINNED_PYMUPDF}/MuPDF "
                           f"{corpus.PINNED_MUPDF})",
        "judges": judges,
        "replication": {
            "min_context_tokens_required": required,
            "derivation": (
                f"largest stimulus ({paper_id}) measured at {worst} prompt tokens "
                f"under the most token-dense of the three tokenizers, "
                f"+ {OUTPUT_BUDGET} output budget + {MARGIN} margin"
            ),
            "requirement": (
                "Each judge must be served with a context window of at least "
                f"{required} tokens. A smaller window truncates the largest "
                "stimuli silently and the run is not a replication. Quantization "
                "must also match: the same slug served at a different "
                "quantization is a different function."
            ),
            "providers_below_requirement": below,
            "pinned_quantizations": {k: v for k, v in PINNED_QUANT.items()},
            "quantization_requirement": (
                "Each judge must be served at the quantization pinned here. "
                "The same slug at a different quantization is a different "
                "function, and OpenRouter's routing is not deterministic, so "
                "an unpinned run samples several numerically distinct raters "
                "under one name."
            ),
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("-" * 64)
    print(f"min context window required : {required} tokens")
    print(f"providers below requirement : {len(below)}")
    for b in below:
        print(f"   {b['judge']:9} {b['provider']:14} "
              f"ctx={b['context_length']} quant={b['quantization']}")
    print(f"\nwrote {MANIFEST.name}")


def verify() -> None:
    if not MANIFEST.exists():
        sys.exit(f"FATAL: {MANIFEST.name} not found; run --build first.")
    man = json.loads(MANIFEST.read_text())
    required = man["replication"]["min_context_tokens_required"]
    print(f"frozen manifest built {man['built_utc']}")
    print(f"requirement: >= {required} tokens of context per judge\n")

    import second_rater as sr
    sys.path.insert(0, str(HERE))
    drift = []
    if set(sr.MODELS.items()) != set(JUDGES.items()):
        drift.append(f"MODELS in second_rater.py differs from JUDGES here: "
                     f"{sr.MODELS} vs {JUDGES}")

    models = model_table()
    for name, slug in JUDGES.items():
        rec = man["judges"].get(name)
        if not rec:
            drift.append(f"{name}: absent from manifest")
            continue
        live_ctx = models.get(slug, {}).get("context_length")
        if live_ctx != rec.get("context_length"):
            drift.append(f"{name}: model context changed "
                         f"{rec.get('context_length')} -> {live_ctx}")
        live = provider_table(slug)
        lq = sorted({p["quantization"] for p in live if p["quantization"]})
        if lq != rec["quantizations_offered"]:
            drift.append(f"{name}: quantizations changed "
                         f"{rec['quantizations_offered']} -> {lq}")
        pin = rec.get("pinned_quantization")
        if pin:
            at_pin = [p for p in live if p["quantization"] == pin]
            dist = sorted({p["provider"] for p in at_pin if p["provider"]})
            if len(dist) < MIN_PROVIDERS_AT_PIN:
                drift.append(f"{name}: only {len(dist)} provider(s) now serve "
                             f"{pin} (was {rec.get('n_providers_at_pin')})")
            if dist != rec.get("providers_at_pin"):
                drift.append(f"{name}: provider pool at {pin} changed "
                             f"{rec.get('providers_at_pin')} -> {dist}")
            thin = [p for p in at_pin
                    if p["context_length"] and p["context_length"] < required]
            if thin:
                drift.append(f"{name}: {len(thin)} endpoint(s) at {pin} are "
                             f"below the {required} token requirement")
        ctxs = [p["context_length"] for p in live if p["context_length"]]
        bad = [p for p in live if p["context_length"] and p["context_length"] < required]
        status = "OK" if not bad else f"{len(bad)} provider(s) BELOW requirement"
        print(f"  {name:9} live providers={len(live):2}  "
              f"context {min(ctxs) if ctxs else '?'}-{max(ctxs) if ctxs else '?'}  {status}")
        for p in bad:
            print(f"      - {p['provider']} ctx={p['context_length']} "
                  f"quant={p['quantization']}")

    print()
    if drift:
        print("DRIFT AGAINST FROZEN MANIFEST:")
        for d in drift:
            print(f"  ! {d}")
        sys.exit(1)
    print("no drift: live judge attributes match the frozen manifest")


def show() -> None:
    if not MANIFEST.exists():
        sys.exit(f"FATAL: {MANIFEST.name} not found; run --build first.")
    man = json.loads(MANIFEST.read_text())
    print(f"judge manifest  built {man['built_utc']}")
    print(f"stimulus        {man['stimulus_generation']}")
    print(f"instrument_sha  {man['instrument_sha']}\n")
    hdr = (f"{'judge':10} {'model ctx':>10} {'provs':>6} {'prov ctx min':>13} "
           f"{'quant':>6} {'provs@pin':>8} {'meas tok':>9}")
    print(hdr); print("-" * len(hdr))
    for name, j in man["judges"].items():
        print(f"{name:10} {j.get('context_length'):>10} {j['n_providers']:>6} "
              f"{j['provider_context_min']:>13} "
              f"{str(j.get('pinned_quantization')):>6} "
              f"{j.get('n_providers_at_pin', 0):>8} "
              f"{j['measured']['prompt_tokens']:>9}")
    r = man["replication"]
    print(f"\nminimum context window required: {r['min_context_tokens_required']} tokens")
    print(f"  {r['derivation']}")
    if r["providers_below_requirement"]:
        print("\nproviders that must NOT be used:")
        for b in r["providers_below_requirement"]:
            print(f"  {b['judge']:9} {b['provider']:14} ctx={b['context_length']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--build", action="store_true")
    g.add_argument("--verify", action="store_true")
    g.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    elif a.verify:
        verify()
    else:
        show()
