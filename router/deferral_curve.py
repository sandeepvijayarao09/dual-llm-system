"""
deferral_curve.py — Cascade accuracy as a function of how much traffic defers.

Usage:
    python -m router.deferral_curve      # table + router/deferral_results.json
                                         # + docs/deferral_curve.svg

The cascade has one knob: the ML router's confidence threshold. Every query
below it goes to the LLM classifier (~0.9 s each on local Gemma 4 E4B).
Sweeping it traces accuracy against the share of queries deferred — the
deferral curve used to compare cascades (Gupta et al., ICLR 2024) — instead
of reporting the single 0.65 operating point. Uses the cached classifier
verdicts in data/classifier_verdicts_1000.jsonl, so it makes no model calls.

Baselines for router v1, at every deferral share f:
    random  defer a random f of queries:  (1 - f) * ML accuracy + f * classifier accuracy
    oracle  defer the queries where the ML router is wrong and the classifier
            right first, the ones where deferring hurts last

This is a diagnostic on the test set: the threshold stays at 0.65; choosing
it from this curve would tune on Eval-1000.
"""

from __future__ import annotations

import json
import os

import numpy as np

from config import ML_ROUTER_CONFIDENCE_THRESHOLD
from router.eval_data import load_cases
from router.ml_router import MLRouter

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
VERDICTS_PATH = os.path.join(ROOT, "data", "classifier_verdicts_1000.jsonl")
RESULTS_PATH = os.path.join(HERE, "deferral_results.json")
SVG_PATH = os.path.join(ROOT, "docs", "deferral_curve.svg")
WORD_RULE_ACC = 0.897
ROUTERS = {"v1": "router_v1.joblib", "v0": "router_v0.joblib"}


def curve(conf: np.ndarray, ml_ok: np.ndarray, clf_ok: np.ndarray) -> list[tuple[float, float, float]]:
    """(threshold, deferred share, cascade accuracy) at every distinct threshold."""
    points = []
    for t in np.unique(np.concatenate([[0.5], conf, [1.0001]])):
        deferred = conf < t
        acc = np.where(deferred, clf_ok, ml_ok).mean()
        points.append((float(min(t, 1.0)), float(deferred.mean()), float(acc)))
    return points


def at(t: float, conf: np.ndarray, ml_ok: np.ndarray, clf_ok: np.ndarray) -> tuple[float, float]:
    """(deferred share, cascade accuracy) at exactly threshold t."""
    deferred = conf < t
    return float(deferred.mean()), float(np.where(deferred, clf_ok, ml_ok).mean())


def oracle(ml_ok: np.ndarray, clf_ok: np.ndarray) -> list[tuple[float, float]]:
    gain = clf_ok.astype(int) - ml_ok.astype(int)           # +1 helps, 0 neutral, -1 hurts
    order = np.sort(gain)[::-1]
    n = len(gain)
    acc = ml_ok.mean() + np.concatenate([[0], np.cumsum(order)]) / n
    return [(k / n, float(a)) for k, a in enumerate(acc)]


def auc(xy: list[tuple[float, float]]) -> float:
    x, y = np.array([p[0] for p in xy]), np.array([p[1] for p in xy])
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2))


def main() -> None:
    cases = load_cases("eval_1000.jsonl")
    truth = np.array([gt for _, gt, _ in cases])
    with open(VERDICTS_PATH, encoding="utf-8") as f:
        verdict = {r["query"]: r["final"] for r in map(json.loads, f)}
    clf_ok = np.array([verdict[q] for q, _, _ in cases]) == truth

    out: dict = {"classifier_accuracy": float(clf_ok.mean()), "routers": {}}
    for name, file in ROUTERS.items():
        router = MLRouter.load(os.path.join(HERE, "models", file))
        preds = [router.predict(q) for q, _, _ in cases]
        conf = np.array([p.confidence for p in preds])
        ml_ok = np.array([p.decision for p in preds]) == truth
        pts = curve(conf, ml_ok, clf_ok)
        op = at(ML_ROUTER_CONFIDENCE_THRESHOLD, conf, ml_ok, clf_ok)
        out["routers"][name] = {
            "ml_accuracy": float(ml_ok.mean()),
            "auc": auc([(d, a) for _, d, a in pts]),
            "operating_point": {"threshold": ML_ROUTER_CONFIDENCE_THRESHOLD,
                                "deferred": op[0], "accuracy": op[1]},
            "curve": pts,
            "table": {f"{t:.2f}": at(t, conf, ml_ok, clf_ok)
                      for t in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 0.95, 1.0001)},
        }
        if name == "v1":
            orc = oracle(ml_ok, clf_ok)
            a0, a1 = ml_ok.mean(), clf_ok.mean()
            out["baselines_v1"] = {
                "random_auc": float((a0 + a1) / 2),
                "oracle_auc": auc(orc),
                "oracle_peak": max(a for _, a in orc),
                "oracle_peak_deferred": min(d for d, a in orc if a == max(b for _, b in orc)),
                "oracle": orc,
            }

    print(f"\nDeferral curve on Eval-1000 (cached Gemma 4 E4B verdicts; classifier alone "
          f"{out['classifier_accuracy']:.1%}). Threshold stays {ML_ROUTER_CONFIDENCE_THRESHOLD}.\n")
    print(f"{'Threshold':>10}  {'v1 deferred':>11} {'v1 acc':>7}   {'v0 deferred':>11} {'v0 acc':>7}")
    for key in out["routers"]["v1"]["table"]:
        (d1, a1), (d0, a0) = (out["routers"][n]["table"][key] for n in ("v1", "v0"))
        label = "1.00" if key == "1.00" else key
        print(f"{label:>10}  {d1:>11.1%} {a1:>7.1%}   {d0:>11.1%} {a0:>7.1%}")
    b = out["baselines_v1"]
    print(f"\nArea under the curve (mean accuracy over deferral 0-100%): "
          f"v1 {out['routers']['v1']['auc']:.3f}, v0 {out['routers']['v0']['auc']:.3f}, "
          f"random {b['random_auc']:.3f}, oracle {b['oracle_auc']:.3f}")
    print(f"Oracle peak: {b['oracle_peak']:.1%} at {b['oracle_peak_deferred']:.1%} deferred")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    with open(SVG_PATH, "w", encoding="utf-8") as f:
        f.write(svg(out))
    print(f"Saved {os.path.normpath(RESULTS_PATH)} and {os.path.normpath(SVG_PATH)}")


# ── SVG (static, light surface; colours from the validated categorical order) ──

W, H = 760, 420
L, R, T, B = 64, 150, 92, 52                  # plot margins
Y0, Y1 = 0.88, 1.00
SURFACE, INK, INK2, GRID, REF = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#8a8983"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def _x(d: float) -> float:
    return L + d * (W - L - R)


def _y(a: float) -> float:
    return T + (Y1 - a) / (Y1 - Y0) * (H - T - B)


def _path(xy, step: bool = False) -> str:
    pts = []
    for i, (d, a) in enumerate(xy):
        if step and i:
            pts.append(f"{_x(d):.1f},{_y(xy[i - 1][1]):.1f}")
        pts.append(f"{_x(d):.1f},{_y(a):.1f}")
    return "M" + " L".join(pts)


def svg(out: dict) -> str:
    v1, v0 = out["routers"]["v1"], out["routers"]["v0"]
    b = out["baselines_v1"]
    acc_cls = out["classifier_accuracy"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
        f'<text x="24" y="30" font-size="17" font-weight="600" fill="{INK}">'
        "Deferral curve: cascade accuracy vs share of queries sent to the LLM classifier</text>",
        f'<text x="24" y="50" font-size="12.5" fill="{INK2}">Eval-1000, cached Gemma 4 E4B '
        f"verdicts. Each deferred query costs ~0.9 s of local inference.</text>",
        f'<text x="24" y="68" font-size="12.5" fill="{INK2}">Dots mark the shipped threshold '
        f"(0.65). Area under curve: v1 {v1['auc']:.3f} · v0 {v0['auc']:.3f} · random "
        f"{b['random_auc']:.3f} · oracle {b['oracle_auc']:.3f}</text>",
    ]
    for a in np.arange(Y0, Y1 + 1e-9, 0.02):
        lines.append(f'<line x1="{L}" y1="{_y(a):.1f}" x2="{W - R}" y2="{_y(a):.1f}" stroke="{GRID}" stroke-width="1"/>')
        lines.append(f'<text x="{L - 8}" y="{_y(a) + 4:.1f}" font-size="11.5" fill="{INK2}" text-anchor="end">{a:.0%}</text>')
    for d in np.arange(0, 1.0001, 0.2):
        lines.append(f'<text x="{_x(d):.1f}" y="{H - B + 18}" font-size="11.5" fill="{INK2}" text-anchor="middle">{d:.0%}</text>')
    lines.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 12}" font-size="12" fill="{INK2}" text-anchor="middle">Share of queries deferred to the LLM classifier</text>')
    lines.append(f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="{REF}" stroke-width="1"/>')

    # Word-count rule reference line
    lines.append(f'<line x1="{L}" y1="{_y(WORD_RULE_ACC):.1f}" x2="{W - R}" y2="{_y(WORD_RULE_ACC):.1f}" stroke="{REF}" stroke-width="1.5"/>')
    lines.append(f'<text x="{W - R + 8}" y="{_y(WORD_RULE_ACC) + 4:.1f}" font-size="12" fill="{INK2}">≤8-words rule {WORD_RULE_ACC:.1%}</text>')

    # Random baseline (grey, de-emphasised), oracle, v0, v1
    a0 = v1["ml_accuracy"]
    lines.append(f'<path d="{_path([(0, a0), (1, acc_cls)])}" fill="none" stroke="{REF}" stroke-width="2" stroke-linecap="round"/>')
    series = [
        (b["oracle"], AQUA, "Oracle deferral (v1)"),
        ([(d, a) for _, d, a in v0["curve"]], ORANGE, "Router v0"),
        ([(d, a) for _, d, a in v1["curve"]], BLUE, "Router v1"),
    ]
    for xy, colour, _ in series:
        xy = sorted(xy)
        lines.append(f'<path d="{_path(xy, step=True)}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    for r, colour in ((v0, ORANGE), (v1, BLUE)):
        op = r["operating_point"]
        lines.append(f'<circle cx="{_x(op["deferred"]):.1f}" cy="{_y(op["accuracy"]):.1f}" r="5" fill="{colour}" stroke="{SURFACE}" stroke-width="2"/>')

    # Direct labels at the right edge, nudged apart; text stays in ink colours
    ends = [
        (max(a for _, a in b["oracle"]), AQUA, "Oracle deferral (v1)"),
        (v1["operating_point"]["accuracy"], BLUE, f"Router v1 · {v1['operating_point']['accuracy']:.1%} at 0.65"),
        (v0["operating_point"]["accuracy"], ORANGE, f"Router v0 · {v0['operating_point']['accuracy']:.1%} at 0.65"),
        (acc_cls - 0.006, REF, "Random deferral (v1)"),
    ]
    used: list[float] = []
    for a, colour, label in ends:
        y = _y(a)
        while any(abs(y - u) < 16 for u in used):
            y += 16
        used.append(y)
        lines.append(f'<line x1="{W - R + 8}" y1="{y:.1f}" x2="{W - R + 20}" y2="{y:.1f}" stroke="{colour}" stroke-width="3" stroke-linecap="round"/>')
        lines.append(f'<text x="{W - R + 24}" y="{y + 4:.1f}" font-size="11.5" fill="{INK}">{label.split(" · ")[0]}</text>')
        if " · " in label:
            lines.append(f'<text x="{W - R + 24}" y="{y + 17:.1f}" font-size="10.5" fill="{INK2}">{label.split(" · ")[1]}</text>')
            used.append(y + 13)
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
