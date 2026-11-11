"""
jev_eval.py — JEV (OpenRouter Decisions API) vs local Gemma as the routing classifier.

Usage:
    python -m router.jev_eval [--model typesafe/jev-1.14] [--max-usd 0.25] [--workers 8]
    python -m router.jev_eval --from-cache   # rebuild the report from
                                             # data/jev_verdicts_1000.jsonl, no API calls

The key is read from OPENROUTER_API_KEY in the environment or in this repo's
.env (gitignored); it is never printed or logged.

JEV is a decision model: given a state and a `choice` question it returns a
probability per option and a confidence. Here the state is the query and the
question is "small or big", with the label definition from router/seed_data.py
(the same rules the Gemma classifier prompt encodes) — nothing is fitted to
Eval-1000. The orchestrator's rule is applied to JEV exactly as to Gemma: only
a "small" verdict with confidence >= CONFIDENCE_THRESHOLD stays small.

Compared on Eval-1000 (FINDINGS.md §11):
    1. JEV vs the cached Gemma 4 E4B verdicts (data/classifier_verdicts_1000.jsonl),
       each alone on all 1,000 queries and each as the cascade's fallback
       behind router v1 (queries with ML confidence < 0.65)
    2. JEV vs the ML routers themselves (v1 and v0), including on the 103
       queries where the word-count rule is wrong

Calls the paid API: ~1,000 short requests, about a cent at JEV's input price.
Stops at --max-usd.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
from scipy.stats import binomtest

from config import CONFIDENCE_THRESHOLD, ML_ROUTER_CONFIDENCE_THRESHOLD, ML_ROUTER_PATH
from router.eval_data import load_cases
from router.ml_router import MLRouter

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
HERE = os.path.dirname(__file__)
GEMMA_VERDICTS = os.path.join(HERE, "..", "data", "classifier_verdicts_1000.jsonl")
OUT_VERDICTS = os.path.join(HERE, "..", "data", "jev_verdicts_1000.jsonl")
RESULTS_PATH = os.path.join(HERE, "jev_results.json")

QUESTION = {
    "route": {
        "type": "choice",
        "instructions": (
            "A user sent this message to an AI assistant. Decide which model should "
            "answer it: a small, cheap model or a large, capable one."
        ),
        "criteria": {
            "small": ("Cheap, one-shot, low risk: greetings and small talk, factual lookups, "
                      "trivia, definitions, unit conversions, simple arithmetic, light code, "
                      "short simple creative requests. A small model answers it well."),
            "big": ("Needs multi-step reasoning, synthesis, proofs, code or system design, "
                    "debugging, analysis, multi-constraint planning or research-style "
                    "answers, or is sensitive: anything where a small model's quality is "
                    "likely to degrade."),
        },
    }
}


class Jev:
    def __init__(self, key: str, model: str, max_usd: float) -> None:
        self.client = httpx.Client(timeout=30, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "X-OpenRouter-Title": "dual-llm-system routing eval"})
        self.model, self.max_usd = model, max_usd
        self.cost = 0.0
        self._lock = threading.Lock()

    def ask(self, query: str) -> dict:
        if self.cost >= self.max_usd:
            raise RuntimeError(f"spend cap ${self.max_usd} reached")
        body = {"model": self.model, "state": query, "questions": QUESTION}
        for attempt in range(4):
            started = time.perf_counter()
            r = self.client.post(ENDPOINT, json=body)
            latency = (time.perf_counter() - started) * 1000
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(0.5 * 2 ** attempt)
                continue
            if r.status_code != 200:
                raise RuntimeError(f"JEV HTTP {r.status_code}: {r.text[:200]}")
            data = r.json()
            a = data["answers"]["route"]
            cost = float((data.get("usage") or {}).get("cost") or 0.0)
            with self._lock:
                self.cost += cost
            small = a["choice"] == "small" and a["confidence"] >= CONFIDENCE_THRESHOLD
            return {"query": query, "choice": a["choice"], "confidence": a["confidence"],
                    "probabilities": a["probabilities"], "final": "small" if small else "big",
                    "latency_ms": round(latency, 1), "cost_usd": cost,
                    "input_tokens": (data.get("usage") or {}).get("input_tokens")}
        raise AssertionError("unreachable")


def _key_from_dotenv(path: str = os.path.join(HERE, "..", ".env")) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                name, _, value = line.strip().partition("=")
                if name.strip() == "OPENROUTER_API_KEY":
                    return value.strip().strip("'\"")
    except OSError:
        pass
    return ""


def mcnemar(a_ok: np.ndarray, b_ok: np.ndarray) -> tuple[int, int, float]:
    """(b fixes, b breaks, exact two-sided p) for b relative to a."""
    fixes, breaks = int((~a_ok & b_ok).sum()), int((a_ok & ~b_ok).sum())
    p = binomtest(fixes, fixes + breaks, 0.5).pvalue if fixes + breaks else 1.0
    return fixes, breaks, float(p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--model", default="typesafe/jev-1.14")
    ap.add_argument("--max-usd", type=float, default=0.25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--from-cache", action="store_true",
                    help="rebuild the report from the saved JEV verdicts; no API calls")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    queries = [q for q, _, _ in load_cases("eval_1000.jsonl")]
    if args.from_cache:
        with open(OUT_VERDICTS, encoding="utf-8") as f:
            cached = {r["query"]: r for r in map(json.loads, f)}
        rows = [cached[q] for q in queries]
        with open(RESULTS_PATH, encoding="utf-8") as f:
            previous = json.load(f)
        args.model, wall = previous.get("jev_model", args.model), previous.get("wall_s")
    else:
        key = os.environ.get("OPENROUTER_API_KEY") or _key_from_dotenv()
        if not key:
            raise SystemExit("Set OPENROUTER_API_KEY in the environment or in .env "
                             "(an OpenRouter key with a little credit).")
        jev = Jev(key, args.model, args.max_usd)
        print(f"\nAsking {args.model} about {len(queries)} queries ({args.workers} at a time)...")
        started = time.time()
        rows = []
        with ThreadPoolExecutor(args.workers) as pool:
            for i, row in enumerate(pool.map(jev.ask, queries), 1):
                rows.append(row)
                if i % 100 == 0:
                    print(f"  {i}/{len(queries)}  (${jev.cost:.4f} so far)")
        wall = round(time.time() - started, 1)
        with open(OUT_VERDICTS, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    report(rows, args.model, wall)


def _fmt_ci(ci: list[float]) -> str:
    return f"[{ci[0]:.3f},{ci[1]:.3f}]"


def _ci(ok: np.ndarray) -> list[float]:
    means = np.random.default_rng(0).choice(ok.astype(float), (2000, len(ok))).mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def report(rows: list[dict], model: str, wall: float | None) -> None:
    cases = load_cases("eval_1000.jsonl")
    queries = [q for q, _, _ in cases]
    truth = np.array([gt for _, gt, _ in cases])
    with open(GEMMA_VERDICTS, encoding="utf-8") as f:
        gemma = {r["query"]: r["final"] for r in map(json.loads, f)}
    words = np.array([len(re.findall(r"\w+", q)) for q in queries])
    length_misleads = np.where(words <= 8, "small", "big") != truth

    jev_final = np.array([r["final"] for r in rows])
    gemma_final = np.array([gemma[q] for q in queries])
    jev_ok, gemma_ok = jev_final == truth, gemma_final == truth

    routers = {}
    for name in ("router_v1.joblib", "router_v0.joblib"):
        r = MLRouter.load(os.path.join(HERE, "models", name))
        t0 = time.perf_counter()
        preds = [r.predict(q) for q in queries]
        ms = (time.perf_counter() - t0) * 1000 / len(queries)
        routers[name] = (np.array([p.decision for p in preds]),
                         np.array([p.confidence for p in preds]), ms)
    ml, conf, _ = routers[os.path.basename(ML_ROUTER_PATH)]
    deferred = conf < ML_ROUTER_CONFIDENCE_THRESHOLD
    casc_gemma = np.where(deferred, gemma_final, ml) == truth
    casc_jev = np.where(deferred, jev_final, ml) == truth

    lat = [r["latency_ms"] for r in rows]
    cost = sum(r.get("cost_usd") or 0.0 for r in rows)
    alone, casc = mcnemar(gemma_ok, jev_ok), mcnemar(casc_gemma, casc_jev)
    vs_ml = {}
    for name, (dec, _, ms) in routers.items():
        ok = dec == truth
        vs_ml[name] = {
            "accuracy": float(ok.mean()), "ci": _ci(ok), "ms_per_query": round(ms, 3),
            "where_length_misleads": float(ok[length_misleads].mean()),
            "jev_vs_router_fixes_breaks_p": mcnemar(ok, jev_ok),
            "jev_vs_router_where_length_misleads": mcnemar(ok[length_misleads], jev_ok[length_misleads]),
            "agreement": float((dec == jev_final).mean()),
        }
    res = {
        "jev_model": model,
        "router": os.path.basename(ML_ROUTER_PATH),
        "classifier_alone": {"jev": float(jev_ok.mean()), "jev_ci": _ci(jev_ok),
                             "gemma": float(gemma_ok.mean()),
                             "jev_vs_gemma_fixes_breaks_p": alone},
        "cascade": {"jev": float(casc_jev.mean()), "gemma": float(casc_gemma.mean()),
                    "deferred": float(deferred.mean()),
                    "jev_on_deferred": float(jev_ok[deferred].mean()),
                    "gemma_on_deferred": float(gemma_ok[deferred].mean()),
                    "jev_vs_gemma_fixes_breaks_p": casc},
        "jev_where_length_misleads": float(jev_ok[length_misleads].mean()),
        "vs_ml_router": vs_ml,
        "large_share": {"jev_alone": float((jev_final == "big").mean()),
                        "gemma_alone": float((gemma_final == "big").mean())},
        "agreement_jev_gemma": float((jev_final == gemma_final).mean()),
        "jev_latency_ms": {"median": statistics.median(lat),
                           "p95": sorted(lat)[int(0.95 * (len(lat) - 1))]},
        "jev_cost_usd": round(cost, 6),
        "jev_upgraded_low_confidence": sum(r["choice"] == "small" and r["final"] == "big" for r in rows),
        "wall_s": wall,
    }

    print(f"\nJEV ({model}) vs Gemma 4 E4B as the routing classifier\n")
    print(f"{'':38}{'JEV':>10}{'Gemma 4 E4B':>14}")
    print("─" * 62)
    print(f"{'Classifier alone (1,000)':<38}{jev_ok.mean():>10.1%}{gemma_ok.mean():>14.1%}")
    print(f"{'On the queries the router defers on':<38}{jev_ok[deferred].mean():>10.1%}{gemma_ok[deferred].mean():>14.1%}")
    print(f"{'Cascade behind ' + res['router']:<38}{casc_jev.mean():>10.1%}{casc_gemma.mean():>14.1%}")
    print(f"JEV vs Gemma alone: JEV fixes {alone[0]}, breaks {alone[1]} (exact McNemar p = {alone[2]:.3f}); "
          f"in the cascade: {casc[0]} vs {casc[1]} (p = {casc[2]:.3f})")

    print("\nJEV vs the ML router\n")
    print(f"{'':22}{'Accuracy':>9}{'95% CI':>17}{'Length misleads':>17}{'ms/query':>10}")
    print("─" * 75)
    for name, v in vs_ml.items():
        print(f"{name:<22}{v['accuracy']:>9.1%}{_fmt_ci(v['ci']):>17}"
              f"{v['where_length_misleads']:>17.1%}{v['ms_per_query']:>10.2f}")
    print(f"{'JEV':<22}{jev_ok.mean():>9.1%}{_fmt_ci(res['classifier_alone']['jev_ci']):>17}"
          f"{res['jev_where_length_misleads']:>17.1%}{res['jev_latency_ms']['median']:>10.0f}")
    for name, v in vs_ml.items():
        b, a, p = v["jev_vs_router_fixes_breaks_p"]
        bh, ah, ph = v["jev_vs_router_where_length_misleads"]
        print(f"{name}: router right & JEV wrong {a}, reverse {b} (p = {p:.3f}); "
              f"where length misleads {ah} vs {bh} (p = {ph:.3f}); agree {v['agreement']:.1%}")
    print(f"\nJEV cost ${cost:.4f} for {len(rows)} calls; median latency "
          f"{res['jev_latency_ms']['median']:.0f} ms (network round trip).")
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Saved {os.path.normpath(RESULTS_PATH)}")


if __name__ == "__main__":
    main()
