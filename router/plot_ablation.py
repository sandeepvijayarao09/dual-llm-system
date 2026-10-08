"""Draw docs/router_vs_baselines.svg from router/ablation_results.json.

Accuracy on Eval-1000 with 95% bootstrap intervals, for the full router and the
baselines in FINDINGS.md section 3. Plain SVG text, no plotting dependency.

    python -m router.plot_ablation
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (name in ablation_results.json, label shown)
ROWS = [
    ("all-big", "Always GPT-4o"),
    ("random (p=0.5)", "Random"),
    ("all-small", "Always GPT-4o-mini"),
    ("tag heuristic (rules only, no learning)", "Tag rules, no learning"),
    ("LR + TF-IDF 1-3 gram, no tags", "TF-IDF + LR, no tags"),
    ("length threshold (<=8 words -> small)", "One line: ≤ 8 words → mini"),
    ("LR + TF-IDF 1-3 gram, + tags", "Full ML router (TF-IDF + tags + LR)"),
]

W, ROW_H, TOP, LEFT, RIGHT = 760, 34, 84, 270, 70
LO, HI = 0.40, 1.00
INK, MUTED, GRID, ACCENT, BASE = "#1f2937", "#6b7280", "#e5e7eb", "#1d4ed8", "#9ca3af"


def x(v: float) -> float:
    return LEFT + (v - LO) / (HI - LO) * (W - LEFT - RIGHT)


def main() -> None:
    results = {r["name"]: r for r in json.loads((ROOT / "router/ablation_results.json").read_text())}
    h = TOP + ROW_H * len(ROWS) + 46
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" '
        f'font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
        f'<rect width="{W}" height="{h}" fill="#ffffff"/>',
        f'<text x="24" y="30" font-size="17" font-weight="600" fill="{INK}">'
        "Routing accuracy on Eval-1000, with 95% bootstrap intervals</text>",
        f'<text x="24" y="50" font-size="12.5" fill="{MUTED}">'
        "The full router beats a one-line word-count rule by 2.9 points, and the "
        "intervals overlap.</text>",
        f'<text x="24" y="68" font-size="12.5" fill="{MUTED}">'
        "Word count and label correlate at r = 0.766 on this benchmark (FINDINGS.md).</text>",
    ]
    for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        out.append(f'<line x1="{x(t):.1f}" y1="{TOP - 6}" x2="{x(t):.1f}" '
                   f'y2="{TOP + ROW_H * len(ROWS)}" stroke="{GRID}"/>')
        out.append(f'<text x="{x(t):.1f}" y="{TOP + ROW_H * len(ROWS) + 18}" font-size="11.5" '
                   f'fill="{MUTED}" text-anchor="middle">{t:.0%}</text>')
    for i, (name, label) in enumerate(ROWS):
        r = results[name]
        y = TOP + i * ROW_H + ROW_H / 2
        color = ACCENT if "+ tags" in name else INK if "length" in name else BASE
        weight = "600" if color != BASE else "400"
        out.append(f'<line x1="{x(r["acc_lo"]):.1f}" y1="{y:.1f}" x2="{x(r["acc_hi"]):.1f}" '
                   f'y2="{y:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round" opacity="0.45"/>')
        out.append(f'<circle cx="{x(r["accuracy"]):.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
        out.append(f'<text x="{LEFT - 14}" y="{y + 4.5:.1f}" font-size="13" fill="{INK}" '
                   f'font-weight="{weight}" text-anchor="end">{label}</text>')
        out.append(f'<text x="{x(r["acc_hi"]) + 10:.1f}" y="{y + 4.5:.1f}" font-size="12" '
                   f'fill="{INK}" font-weight="{weight}">{r["accuracy"]:.1%}</text>')
    out.append(f'<text x="24" y="{h - 10}" font-size="11" fill="{MUTED}">'
               "python -m router.ablation, then python -m router.plot_ablation</text>")
    out.append("</svg>")
    path = ROOT / "docs" / "router_vs_baselines.svg"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
