/**
 * Prepare model output for remark-math:
 *
 * 1. Models write LaTeX with either delimiter style: $…$ / $$…$$ (which
 *    remark-math parses) or \(…\) / \[…\] (which it doesn't). Rewrite the
 *    second style into the first.
 * 2. "$5 and $10" would otherwise parse as math between the two dollar
 *    signs. A "$" followed by an amount ("$5", "$0.75", "$1,200", "$3k")
 *    and then whitespace, punctuation or the end is currency: escape it.
 *    "$5$", "$2^n$" and "$2k$" stay math, because the amount runs straight
 *    into "$", "^" or another math character.
 *
 * Fenced and inline code are left untouched.
 */
const CURRENCY = /(?<![\\$])\$(?=\d[\d,]*(?:\.\d+)?(?:[kKmMbB]n?)?(?:[\s.,;:!?)\]]|$))/g;

export function normalizeMathDelimiters(markdown: string): string {
  return markdown
    .split(/(```[\s\S]*?```|`[^`\n]*`)/g)
    .map((part, i) =>
      i % 2 === 1
        ? part // code: keep verbatim
        : part
            .replace(/\\\[([\s\S]+?)\\\]/g, (_, tex: string) => `\n$$\n${tex.trim()}\n$$\n`)
            .replace(/\\\(([\s\S]+?)\\\)/g, (_, tex: string) => `$${tex.trim()}$`)
            .replace(CURRENCY, "\\$"),
    )
    .join("");
}
