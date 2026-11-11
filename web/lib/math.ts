/**
 * Models write LaTeX with either delimiter style: $…$ / $$…$$ (which
 * remark-math parses) or \(…\) / \[…\] (which it doesn't). Rewrite the
 * second style into the first, leaving fenced and inline code untouched.
 */
export function normalizeMathDelimiters(markdown: string): string {
  return markdown
    .split(/(```[\s\S]*?```|`[^`\n]*`)/g)
    .map((part, i) =>
      i % 2 === 1
        ? part // code: keep verbatim
        : part
            .replace(/\\\[([\s\S]+?)\\\]/g, (_, tex: string) => `\n$$\n${tex.trim()}\n$$\n`)
            .replace(/\\\(([\s\S]+?)\\\)/g, (_, tex: string) => `$${tex.trim()}$`),
    )
    .join("");
}
