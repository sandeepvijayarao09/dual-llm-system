# Paper

NeurIPS 2026-format paper for the dual-LLM system in this repository, including
the controlled ablation from [`../FINDINGS.md`](../FINDINGS.md) (the 89.7%
word-count baseline and the length confound).

- `main.tex` / `main.pdf`: the current paper
- `checklist.tex`: NeurIPS checklist, `\input` by `main.tex`
- `refs.bib`: bibliography
- `neurips_2026.sty`: official NeurIPS 2026 style file (v2026-01-29)
- `archive/`: the original course final report (Zichen Qi, Yalin Sun and
  Sandeep Vijayarao, Northeastern University) that this paper extends. It
  predates the ablation, so its claims about the semantic feature tags and its
  seed-set size are superseded by `main.tex` and FINDINGS.md.

## Build

With [Tectonic](https://tectonic-typesetting.github.io) (fetches packages and
runs BibTeX itself):

```bash
cd paper
tectonic main.tex
```

Or with a TeX distribution:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The archived report is kept as source. It uses an inline numeric bibliography
that natbib's author-year mode rejects, so build it with `pdflatex` (which
continues past that error) from `archive/` with the style file on the path:
`TEXINPUTS=..: pdflatex -interaction=nonstopmode course-report.tex`.

The paper uses `\usepackage[preprint]{neurips_2026}`, which un-anonymizes the
author block and adds "Preprint. Work in progress." to the footer.
