# Final Report — Dual-LLM System

NeurIPS 2026-format paper describing the dual-LLM system implemented in this
repository.

**Authors:** Zichen Qi, Yalin Sun, Sandeep Vijayarao (Northeastern University)

## Files

- `main.tex` — full paper (uses the `[preprint]` option to un-anonymize)
- `checklist.tex` — completed 16-item NeurIPS checklist (input by `main.tex`)
- `neurips_2026.sty` — official NeurIPS 2026 style file (v2026-01-29)

## Build

Requires a TeX distribution (TeX Live, MacTeX, MiKTeX) with `pdflatex` and
`bibtex` (or just `natbib` from the `neurips_2026` package — we use an
inline `thebibliography`, so `bibtex` is not required).

```bash
cd paper
pdflatex main.tex
pdflatex main.tex   # second pass resolves cross-references
```

This produces `main.pdf`.

## Notes on the track option

The paper is compiled with `\usepackage[preprint]{neurips_2026}`, which:

- un-anonymizes the author block (required for course submission)
- adds "Preprint. Work in progress." in the footer
- is the correct option per the NeurIPS 2026 style guide when the paper is
  not being submitted for double-blind review

For the actual NeurIPS submission (hypothetically), swap the first
`\usepackage` line for either:

- `\usepackage{neurips_2026}` (default, double-blind Main Track), or
- `\usepackage[main, final]{neurips_2026}` (camera-ready for an accepted
  Main Track paper).
