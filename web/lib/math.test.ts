// Run with `npm test` (Node's built-in runner; Node 24+ strips the types).
import assert from "node:assert/strict";
import { test } from "node:test";
import { normalizeMathDelimiters as norm } from "./math.ts";

test("currency is escaped so it isn't parsed as math", () => {
  assert.equal(norm("It costs $5 and $10."), "It costs \\$5 and \\$10.");
  assert.equal(norm("$0.75 / $3.75 per 1M"), "\\$0.75 / \\$3.75 per 1M");
  assert.equal(norm("about $1,200, or $3k"), "about \\$1,200, or \\$3k");
  assert.equal(norm("Total: $42"), "Total: \\$42");
});

test("inline math that starts with a digit stays math", () => {
  for (const s of ["$5$", "$2^n$", "$2k$", "$2k^2 = q^2$", "$10\\%$"]) assert.equal(norm(s), s);
});

test("ordinary inline and display math is unchanged", () => {
  for (const s of ["$x^2$", "$$\\sqrt{2}$$", "where $p$ and $q$ are coprime"]) {
    assert.equal(norm(s), s);
  }
});

test("\\( \\) and \\[ \\] become dollar delimiters", () => {
  assert.equal(norm("so \\(p^2 = 2q^2\\) holds"), "so $p^2 = 2q^2$ holds");
  assert.equal(norm("\\[ \\sqrt{2} \\notin \\mathbb{Q} \\]"), "\n$$\n\\sqrt{2} \\notin \\mathbb{Q}\n$$\n");
});

test("already-escaped dollars and code are left alone", () => {
  assert.equal(norm("\\$5 already escaped"), "\\$5 already escaped");
  assert.equal(norm("run `echo $5` then"), "run `echo $5` then");
  assert.equal(norm("```\nprice=$5\n```"), "```\nprice=$5\n```");
});
