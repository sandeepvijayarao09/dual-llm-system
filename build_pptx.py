"""
Build a native PowerPoint (.pptx) version of the dual-LLM presentation.
Mirrors the content of presentation.html — 13 slides, same theme.

Run: python build_pptx.py
Output: presentation.pptx
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

# ── Brand palette ────────────────────────────────────────────────
RED     = RGBColor(0xC0, 0x39, 0x2B)
BLACK   = RGBColor(0x11, 0x11, 0x11)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
GRAY    = RGBColor(0x55, 0x55, 0x55)
LIGHT   = RGBColor(0xF4, 0xF4, 0xF4)
BORDER  = RGBColor(0xE0, 0xE0, 0xE0)
GREEN   = RGBColor(0x27, 0xAE, 0x60)
ORANGE  = RGBColor(0xF3, 0x9C, 0x12)
BLUE1   = RGBColor(0x1A, 0x52, 0x76)
BLUE2   = RGBColor(0x1F, 0x61, 0x8D)
NAVY    = RGBColor(0x2C, 0x3E, 0x50)
PURPLE  = RGBColor(0x6C, 0x34, 0x83)
BROWN   = RGBColor(0x78, 0x42, 0x12)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# ── Presentation setup ──────────────────────────────────────────
prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H
blank = prs.slide_layouts[6]  # blank layout

# ── Helpers ──────────────────────────────────────────────────────
def add_bg(slide, color):
    """Fill entire slide background with solid color."""
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    bg.shadow.inherit = False
    return bg

def add_left_bar(slide, color=RED):
    """Thin red bar on the left edge (brand accent)."""
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.08), SLIDE_H)
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()

def add_text(slide, left, top, width, height, text, *,
             size=14, bold=False, color=BLACK, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, font="Segoe UI"):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    return box

def add_title(slide, text):
    """Section title with underline."""
    add_text(slide, Inches(0.5), Inches(0.25), Inches(12.3), Inches(0.6),
             text.upper(), size=28, bold=True, color=BLACK)
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                  Inches(0.5), Inches(0.95),
                                  Inches(12.3), Inches(0.025))
    line.fill.solid()
    line.fill.fore_color.rgb = RED
    line.line.fill.background()

def add_bullets(slide, left, top, width, height, items, *, size=14, color=BLACK):
    """Red-square bulleted list."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
        p.alignment = PP_ALIGN.LEFT
        r1 = p.add_run()
        r1.text = "■  "
        r1.font.size = Pt(size - 2)
        r1.font.color.rgb = RED
        r1.font.bold = True
        r1.font.name = "Segoe UI"
        r2 = p.add_run()
        r2.text = item
        r2.font.size = Pt(size)
        r2.font.color.rgb = color
        r2.font.name = "Segoe UI"
    return box

def add_box(slide, left, top, width, height, fill_color, *,
            text=None, text_size=12, text_color=WHITE, text_bold=False,
            subtext=None, subtext_size=10, subtext_color=None,
            align=PP_ALIGN.CENTER, rounded=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    box = slide.shapes.add_shape(shape_type, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = fill_color
    box.line.fill.background()
    if text:
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = Inches(0.1)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = align
        r = p.add_run()
        r.text = text
        r.font.size = Pt(text_size)
        r.font.bold = text_bold
        r.font.color.rgb = text_color
        r.font.name = "Segoe UI"
        if subtext:
            p2 = tf.add_paragraph()
            p2.alignment = align
            r2 = p2.add_run()
            r2.text = subtext
            r2.font.size = Pt(subtext_size)
            r2.font.color.rgb = subtext_color or text_color
            r2.font.name = "Segoe UI"
    return box

def add_callout(slide, left, top, width, height, text, *,
                bg=RGBColor(0xFD, 0xF0, 0xF0), accent=RED, text_color=BLACK):
    """Left-accent callout box."""
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = bg
    box.line.fill.background()
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 left, top, Inches(0.05), height)
    bar.fill.solid()
    bar.fill.fore_color.rgb = accent
    bar.line.fill.background()
    tb = slide.shapes.add_textbox(left + Inches(0.15), top,
                                  width - Inches(0.2), height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    # Split on bold markers
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        r = p.add_run()
        r.text = part
        r.font.size = Pt(11)
        r.font.color.rgb = accent if i % 2 == 1 else text_color
        r.font.bold = (i % 2 == 1)
        r.font.name = "Segoe UI"

def add_stat_box(slide, left, top, width, height, num, lbl, *,
                 num_color=RED, bg=BLACK):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                 left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = bg
    box.line.fill.background()
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_top = tf.margin_bottom = Inches(0.05)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = num
    r.font.size = Pt(36)
    r.font.bold = True
    r.font.color.rgb = num_color
    r.font.name = "Segoe UI"
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = lbl
    r2.font.size = Pt(10)
    r2.font.color.rgb = RGBColor(0xBB, 0xBB, 0xBB)
    r2.font.name = "Segoe UI"

def add_arrow(slide, left, top, width, height, color=RGBColor(0x99, 0x99, 0x99)):
    arr = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, left, top, width, height)
    arr.fill.solid()
    arr.fill.fore_color.rgb = color
    arr.line.fill.background()

# ══════════════════════════════════════════════════════════════════
# SLIDE 1 — TITLE
# ══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
add_bg(s, BLACK)

add_text(s, Inches(1), Inches(1.8), Inches(11.3), Inches(1.0),
         "A Dual-LLM System with", size=54, bold=True, color=WHITE,
         align=PP_ALIGN.CENTER)
add_text(s, Inches(1), Inches(2.8), Inches(11.3), Inches(1.0),
         "Cascade Routing", size=54, bold=True, color=RED,
         align=PP_ALIGN.CENTER)
add_text(s, Inches(1), Inches(4.0), Inches(11.3), Inches(0.4),
         "Machine-Learned Intelligence & Bounded Memory",
         size=20, color=RGBColor(0xBB, 0xBB, 0xBB), align=PP_ALIGN.CENTER)

add_text(s, Inches(1), Inches(5.0), Inches(11.3), Inches(0.35),
         "Zichen Qi  ·  Yalin Sun  ·  Sandeep Vijayarao",
         size=16, color=RGBColor(0xDD, 0xDD, 0xDD), align=PP_ALIGN.CENTER)
add_text(s, Inches(1), Inches(5.4), Inches(11.3), Inches(0.35),
         "Northeastern University  |  CS Final Project",
         size=14, color=RGBColor(0xBB, 0xBB, 0xBB), align=PP_ALIGN.CENTER)
add_text(s, Inches(1), Inches(5.85), Inches(11.3), Inches(0.3),
         "github.com/sandeepvijayarao09/dual-llm-system",
         size=12, color=RGBColor(0xAA, 0xAA, 0xAA), align=PP_ALIGN.CENTER)
add_text(s, Inches(1), Inches(6.2), Inches(11.3), Inches(0.3),
         "April 2026",
         size=11, color=RGBColor(0x88, 0x88, 0x88), align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════════
# SLIDE 2 — COST OF INTELLIGENCE
# ══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
add_left_bar(s)
add_title(s, "The Cost of Intelligence")

add_bullets(s, Inches(0.6), Inches(1.5), Inches(6.5), Inches(4),
            [
                "Frontier models cost 25× more per token than compact models.",
                "Most chat traffic includes greetings or simple lookups.",
                "Advanced reasoning models are over-utilized unnecessarily.",
                "Linear context growth leads to massive latency spikes.",
                "Efficiency requires separating simple tasks from complex ones.",
            ], size=15)

# Cost comparison bars
add_text(s, Inches(7.5), Inches(1.5), Inches(5.0), Inches(0.35),
         "Cost Comparison (per 1M Tokens)", size=14, bold=True)
add_text(s, Inches(7.5), Inches(2.2), Inches(2.0), Inches(0.4),
         "GPT-4o (Large)", size=12, bold=True,
         anchor=MSO_ANCHOR.MIDDLE)
add_box(s, Inches(9.6), Inches(2.2), Inches(2.7), Inches(0.5),
        RED, text="$15.00", text_size=14, text_bold=True)

add_text(s, Inches(7.5), Inches(3.0), Inches(2.0), Inches(0.4),
         "GPT-4o-mini (Small)", size=12, bold=True,
         anchor=MSO_ANCHOR.MIDDLE)
add_box(s, Inches(9.6), Inches(3.0), Inches(0.18), Inches(0.5),
        GRAY, text="")
add_text(s, Inches(9.9), Inches(3.0), Inches(1.0), Inches(0.5),
         "$0.60", size=12, bold=True, anchor=MSO_ANCHOR.MIDDLE)

add_callout(s, Inches(7.5), Inches(4.0), Inches(5.0), Inches(0.8),
            "**25× cost disparity** based on current OpenAI pricing.")

# ══════════════════════════════════════════════════════════════════
# SLIDE 3 — PROJECT CONTRIBUTIONS
# ══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
add_left_bar(s)
add_title(s, "Project Contributions")

def card(slide, l, t, w, h, icon, title, desc):
    c = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    c.fill.solid(); c.fill.fore_color.rgb = WHITE
    c.line.color.rgb = BORDER; c.line.width = Pt(1)
    add_text(slide, l + Inches(0.2), t + Inches(0.15), Inches(0.6), Inches(0.5),
             icon, size=22, color=RED)
    add_text(slide, l + Inches(0.2), t + Inches(0.7), w - Inches(0.4), Inches(0.4),
             title, size=13, bold=True)
    add_text(slide, l + Inches(0.2), t + Inches(1.15), w - Inches(0.4), h - Inches(1.2),
             desc, size=10, color=GRAY)

# Top row - 3 cards
card(s, Inches(0.5), Inches(1.3), Inches(4.1), Inches(2.3),
     "🔀", "Three-Layer Cascade",
     "Pre-flight + offline TF-IDF ML router + LLM-based JSON routing fallback — 85.3% of queries handled at zero API cost.")
card(s, Inches(4.8), Inches(1.3), Inches(4.1), Inches(2.3),
     "🪄", "Personalization-by-Compression",
     "4-stage pipeline preserving privacy: the Large LLM never sees the user profile directly.")
card(s, Inches(9.1), Inches(1.3), Inches(4.1), Inches(2.3),
     "🧮", "Bounded Memory",
     "Verbatim N-turn buffer with O(1) rolling summarization per turn. Amortized constant cost.")

# Bottom row - 2 cards
card(s, Inches(2.0), Inches(3.9), Inches(4.5), Inches(2.3),
     "🗄️", "SQLite Logging & Self-Improvement",
     "Every routing decision logged → export high-confidence labels → retrain. Automated self-supervised dataset generation.")
card(s, Inches(6.8), Inches(3.9), Inches(4.5), Inches(2.3),
     "</>", "Open Source + 38-Test Suite",
     "Full implementation with comprehensive pytest coverage: 17 router + 9 memory + 12 integration tests = 38 tests.")

# ══════════════════════════════════════════════════════════════════
# SLIDE 4 — LANDSCAPE & DIFFERENTIATION
# ══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
add_left_bar(s)
add_title(s, "Landscape & Differentiation")

# Table
rows = [
    ("Prior Work", "Core Approach", "Our Differentiation"),
    ("FrugalGPT (2023)", "Sequential LLM cascades.", "Added offline ML pre-stage — zero API cost for 85% of queries."),
    ("RouteLLM (2024)", "Pairwise preference training.", "TF-IDF speed + prompt logic — no preference data needed."),
    ("LangChain Memory", "Standard summary buffers.", "Fixed O(1) incremental update with no-LLM fallback."),
    ("LaMP (2024)", "RAG-based personalization.", "Privacy-first summary routing — Large LLM never sees profile."),
]
tbl = s.shapes.add_table(len(rows), 3,
                         Inches(0.5), Inches(1.4),
                         Inches(12.3), Inches(3.8)).table
tbl.columns[0].width = Inches(2.8)
tbl.columns[1].width = Inches(3.8)
tbl.columns[2].width = Inches(5.7)

for i, row in enumerate(rows):
    for j, val in enumerate(row):
        cell = tbl.cell(i, j)
        cell.text = ""
        p = cell.text_frame.paragraphs[0]
        r = p.add_run()
        r.text = val
        r.font.size = Pt(12 if i == 0 else 11)
        r.font.bold = (i == 0 or j == 0)
        r.font.name = "Segoe UI"
        if i == 0:
            r.font.color.rgb = WHITE
            cell.fill.solid(); cell.fill.fore_color.rgb = BLACK
        else:
            r.font.color.rgb = BLACK
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if i % 2 == 1 else RGBColor(0xE8, 0xEC, 0xF0)

add_callout(s, Inches(0.5), Inches(5.5), Inches(12.3), Inches(0.9),
            "**Key insight:** No prior system combines offline ML routing + privacy-preserving personalization + bounded O(1) memory in a single deployable API-only stack.")

# ══════════════════════════════════════════════════════════════════
# SLIDE 5 — SYSTEM ARCHITECTURE
# ══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
add_left_bar(s)
add_title(s, "System Architecture")

# Left: diagram
add_text(s, Inches(0.5), Inches(1.2), Inches(7.0), Inches(0.3),
         "ROUTING — decide which model handles the query",
         size=10, bold=True, color=RED)

add_box(s, Inches(0.5), Inches(1.5), Inches(7.0), Inches(0.6), NAVY,
        text="Layer 0 — Pre-flight Length Check", text_size=12, text_bold=True,
        subtext="query > 2,000 words → skip to Large LLM instantly",
        subtext_size=9, subtext_color=RGBColor(0xAA, 0xBB, 0xCC),
        rounded=True)
add_arrow(s, Inches(3.8), Inches(2.2), Inches(0.3), Inches(0.18))

add_box(s, Inches(0.5), Inches(2.45), Inches(7.0), Inches(0.7), BLUE1,
        text="Layer 1 — ML Router  (~0 ms, no API call)",
        text_size=12, text_bold=True,
        subtext="TF-IDF (1–3 grams, 5k feats) + 16 feature tags + Logistic Regression",
        subtext_size=9, subtext_color=RGBColor(0xAA, 0xCC, 0xE5),
        rounded=True)
add_arrow(s, Inches(3.8), Inches(3.25), Inches(0.3), Inches(0.18))
add_text(s, Inches(4.2), Inches(3.25), Inches(3), Inches(0.2),
         "conf < 0.65 → cascade", size=8, color=GRAY)

add_box(s, Inches(0.5), Inches(3.5), Inches(7.0), Inches(0.7), BLUE2,
        text="Layer 2 — LLM Classifier  (GPT-4o-mini, JSON mode)",
        text_size=12, text_bold=True,
        subtext="routing_decision · profile_relevant · confidence",
        subtext_size=9, subtext_color=RGBColor(0xAA, 0xCC, 0xE5),
        rounded=True)
add_arrow(s, Inches(3.8), Inches(4.3), Inches(0.3), Inches(0.18))

add_box(s, Inches(0.5), Inches(4.55), Inches(3.3), Inches(0.75),
        GREEN, text="Small LLM",
        text_size=14, text_bold=True,
        subtext="GPT-4o-mini · SimpleAnswerer",
        subtext_size=9, subtext_color=RGBColor(0xDD, 0xFF, 0xDD),
        rounded=True)
add_box(s, Inches(4.0), Inches(4.55), Inches(3.5), Inches(0.75),
        RED, text="Large LLM",
        text_size=14, text_bold=True,
        subtext="GPT-4o · Reasoner pipeline",
        subtext_size=9, subtext_color=RGBColor(0xFF, 0xDD, 0xDD),
        rounded=True)

add_text(s, Inches(0.5), Inches(5.5), Inches(7.0), Inches(0.3),
         "BACKGROUND — runs every turn automatically",
         size=10, bold=True, color=PURPLE)
add_box(s, Inches(0.5), Inches(5.8), Inches(3.3), Inches(0.55), PURPLE,
        text="ConversationBuffer",
        text_size=11, text_bold=True,
        subtext="bounded memory",
        subtext_size=8, subtext_color=RGBColor(0xDD, 0xBB, 0xDD),
        rounded=True)
add_box(s, Inches(4.0), Inches(5.8), Inches(3.5), Inches(0.55), BROWN,
        text="Routing Log (SQLite)",
        text_size=11, text_bold=True,
        subtext="training data for next router",
        subtext_size=8, subtext_color=RGBColor(0xE0, 0xC0, 0xA0),
        rounded=True)

# Right: bullets
add_bullets(s, Inches(8.0), Inches(1.5), Inches(5.0), Inches(5.5),
            [
                "Pre-flight length check bypasses short/long extremes.",
                "ML Router uses TF-IDF and Logistic Regression — ~0 ms.",
                "LLM Classifier acts as high-accuracy fallback — only 14.7% of queries.",
                "Reasoner (Large LLM) handles core logic without profiles.",
                "Personalizer (Small LLM) assembles the final user-facing response.",
                "SQLite Logger captures every decision for self-supervised retraining.",
            ], size=13)

# ══════════════════════════════════════════════════════════════════
# SLIDE 6 — CASCADE ROUTER
# ══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
add_left_bar(s)
add_title(s, "Stage 1: Cascade Router")

# Left: pipeline visualization
pipe_bg = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                             Inches(0.5), Inches(1.3),
                             Inches(7.3), Inches(5.2))
pipe_bg.fill.solid(); pipe_bg.fill.fore_color.rgb = RGBColor(0xF4, 0xF0, 0xFF)
pipe_bg.line.fill.background()

add_text(s, Inches(0.7), Inches(1.45), Inches(7), Inches(0.35),
         "ML Pipeline — How One Query Becomes a Routing Decision",
         size=13, bold=True)

# Feature extraction box
fx = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                        Inches(0.75), Inches(1.95),
                        Inches(3.3), Inches(2.0))
fx.fill.solid(); fx.fill.fore_color.rgb = WHITE
fx.line.color.rgb = PURPLE; fx.line.width = Pt(1.5)
add_text(s, Inches(0.9), Inches(2.05), Inches(3.0), Inches(0.3),
         "① Feature Extraction", size=11, bold=True, color=PURPLE)
add_text(s, Inches(0.9), Inches(2.35), Inches(3.0), Inches(1.2),
         'encode("why is ML hard?")\n→ LEN_SHORT HAS_REASONING_KW\n   Q_SINGLE why is ML hard?',
         size=9, color=GRAY, font="Courier New")
add_text(s, Inches(0.9), Inches(3.45), Inches(3.0), Inches(0.5),
         "TF-IDF Vectorizer\n1–3 grams · 5,000 features",
         size=9, color=GRAY)

# Logistic regression box
lr = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                        Inches(4.2), Inches(1.95),
                        Inches(3.3), Inches(2.0))
lr.fill.solid(); lr.fill.fore_color.rgb = BLUE1
lr.line.fill.background()
add_text(s, Inches(4.35), Inches(2.05), Inches(3.0), Inches(0.3),
         "② Logistic Regression", size=11, bold=True, color=WHITE)
add_text(s, Inches(4.35), Inches(2.4), Inches(3.0), Inches(1.1),
         "Trained on 280 seed examples\nBalanced class weights\nC=2.0, LBFGS solver",
         size=10, color=RGBColor(0xAA, 0xCC, 0xE5))
add_text(s, Inches(4.35), Inches(3.45), Inches(3.0), Inches(0.5),
         "Output: P(small), P(big)",
         size=10, bold=True, color=RGBColor(0x90, 0xFF, 0xCC))

# Dispatch rule strip
add_box(s, Inches(0.75), Inches(4.2), Inches(6.75), Inches(0.55),
        GREEN, text="max(P) ≥ 0.65 → Trust ML  |  max(P) < 0.65 → LLM Classifier fallback",
        text_size=12, text_bold=True, rounded=True)

# Self-improving loop note
add_text(s, Inches(0.75), Inches(4.95), Inches(6.75), Inches(0.3),
         "Every decision → SQLite log → retrain next router version",
         size=10, color=GRAY, align=PP_ALIGN.CENTER)

# Formula box
formula_bg = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(0.75), Inches(5.4),
                                Inches(6.75), Inches(1.0))
formula_bg.fill.solid(); formula_bg.fill.fore_color.rgb = WHITE
formula_bg.line.color.rgb = BORDER
add_text(s, Inches(0.85), Inches(5.5), Inches(6.55), Inches(0.3),
         "Dispatch Rule (τ = 0.65)", size=11, bold=True, align=PP_ALIGN.CENTER)
add_text(s, Inches(0.85), Inches(5.8), Inches(6.55), Inches(0.5),
         "route(q) = ŷ_ML   if c_ML ≥ τ\n              f_LLM(q)   otherwise",
         size=12, bold=True, align=PP_ALIGN.CENTER, font="Courier New")

# Right: bullets
add_text(s, Inches(8.0), Inches(1.3), Inches(5.0), Inches(0.4),
         "Logistic Regression Pipeline", size=14, bold=True)
add_bullets(s, Inches(8.0), Inches(1.75), Inches(5.0), Inches(4.5),
            [
                "16 hand-crafted tag signals + TF-IDF weights.",
                "Reasoning triggers: 'why', 'derive', 'analyze', 'prove'.",
                "TF-IDF Vectorization: 1–3 ngrams, 5,000 features.",
                "Model serialized via Joblib for < 5 ms latency.",
                "Never worse than an LLM-only baseline — conservative fallback.",
            ], size=12)

# ══════════════════════════════════════════════════════════════════
# SLIDE 7 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
add_left_bar(s)
add_title(s, "Feature Engineering — 16 Routing Tags")

add_text(s, Inches(0.5), Inches(1.15), Inches(12.3), Inches(0.3),
         "Each tag fires when its pattern matches, prepended to the raw query before TF-IDF encoding.",
         size=11, color=GRAY)

tags = [
    ("LEN_SHORT",        "≤ 8 words",                          "→ small", GREEN),
    ("LEN_MEDIUM",       "9–40 words",                         "neutral", GRAY),
    ("LEN_LONG",         "> 40 words",                         "→ big", RED),
    ("HAS_CODE",         "```, def, class, import",            "→ big", RED),
    ("HAS_MATH",         "digits+ops, \\int, \\sum",           "→ big", RED),
    ("HAS_REASONING_KW", "why, how, compare, prove, analyze",  "→ big", RED),
    ("HAS_FACTUAL_KW",   "what is, who is, when did, define",  "→ small", GREEN),
    ("HAS_GREETING_KW",  "hi, hello, thanks, bye",             "→ small", GREEN),
    ("HAS_CREATIVE_KW",  "joke, poem, story, haiku, pun",      "→ small", GREEN),
    ("HAS_OPINION_KW",   "best, vs, should I, pros and cons",  "→ big", RED),
    ("HAS_DEEP_WHAT",    "\"what is\" + > 5 words (abstract)", "→ big", RED),
    ("HAS_IMPACT_KW",    "impact, effect, consequence",        "→ big", RED),
    ("HAS_RESEARCH_KW",  "research, literature, studies show", "→ big", RED),
    ("HAS_SELF_REF",     "my name, who am I, about me",        "→ inject", GRAY),
    ("Q_MULTI",          "≥ 2 question marks",                 "→ big", RED),
    ("MULTI_SENTENCE",   "≥ 3 sentence-ending chars",          "→ big", RED),
]

left_tags  = tags[:8]
right_tags = tags[8:]

def draw_tag_row(slide, left, top, tag, sig, hint, color):
    # tag box
    tb = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                left, top, Inches(2.3), Inches(0.33))
    tb.fill.solid(); tb.fill.fore_color.rgb = RGBColor(0xF0, 0xF0, 0xF0)
    tb.line.fill.background()
    tf = tb.text_frame; tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = tag
    r.font.size = Pt(9); r.font.bold = True
    r.font.name = "Courier New"; r.font.color.rgb = RED
    # signal
    add_text(slide, left + Inches(2.4), top, Inches(2.6), Inches(0.33),
             sig, size=9, color=GRAY, anchor=MSO_ANCHOR.MIDDLE)
    # hint
    add_text(slide, left + Inches(5.05), top, Inches(1.0), Inches(0.33),
             hint, size=9, bold=True, color=color, anchor=MSO_ANCHOR.MIDDLE)

for i, (tag, sig, hint, color) in enumerate(left_tags):
    draw_tag_row(s, Inches(0.5), Inches(1.55 + i * 0.4), tag, sig, hint, color)
for i, (tag, sig, hint, color) in enumerate(right_tags):
    draw_tag_row(s, Inches(7.0), Inches(1.55 + i * 0.4), tag, sig, hint, color)

add_callout(s, Inches(0.5), Inches(6.4), Inches(12.3), Inches(0.7),
            'Example: encode("why is the sky blue?") → LEN_SHORT HAS_REASONING_KW Q_SINGLE why is the sky blue?')

# ══════════════════════════════════════════════════════════════════
# SLIDE 8 — PERSONALIZATION-BY-COMPRESSION
# ══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
add_left_bar(s)
add_title(s, "Personalization-by-Compression")

# 4-step pipeline
steps = [
    ("Reason",    "Large LLM writes deep core\nanswer. No user profile."),
    ("Summarize", "Small LLM compresses core\nto one sentence (≤80 tokens)."),
    ("Add Context", "Small LLM injects user profile\n— intro + closing wrapper."),
    ("Assemble",  "Final output:\nintro + core + \"---\" + closing"),
]

step_w = Inches(2.9)
step_gap = Inches(0.25)
start_x = Inches(0.5)
step_y = Inches(1.7)

for i, (title, desc) in enumerate(steps):
    x = start_x + (step_w + step_gap) * i
    # red circle
    circle = s.shapes.add_shape(MSO_SHAPE.OVAL,
                                x + Inches(1.15), step_y,
                                Inches(0.6), Inches(0.6))
    circle.fill.solid(); circle.fill.fore_color.rgb = RED
    circle.line.fill.background()
    # title
    add_text(s, x, step_y + Inches(0.75), step_w, Inches(0.35),
             title, size=16, bold=True, align=PP_ALIGN.CENTER)
    # description
    add_text(s, x, step_y + Inches(1.15), step_w, Inches(1.1),
             desc, size=11, color=GRAY, align=PP_ALIGN.CENTER)
    # arrow
    if i < len(steps) - 1:
        arr = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                 x + step_w + Inches(0.02),
                                 step_y + Inches(0.2),
                                 step_gap - Inches(0.04), Inches(0.25))
        arr.fill.solid(); arr.fill.fore_color.rgb = RED
        arr.line.fill.background()

# Privacy Win + Cost Win callouts
add_callout(s, Inches(0.5), Inches(4.6), Inches(6.15), Inches(1.5),
            "**🔒 Privacy Win:**  The Large LLM never sees the user profile. "
            "Profile context only enters through the Small LLM wrapper — "
            "limiting exposure of personal data to the most powerful model.")
add_callout(s, Inches(6.85), Inches(4.6), Inches(6.0), Inches(1.5),
            "**💰 Cost Win:**  Large LLM output is never re-injected into context "
            "windows. One-sentence summary prevents exponential token growth "
            "across the personalization chain.",
            bg=RGBColor(0xF0, 0xFF, 0xF4), accent=GREEN)

add_text(s, Inches(0.5), Inches(6.3), Inches(12.3), Inches(0.5),
         "Profile injection is conditional — the LLM classifier's profile_relevant flag decides per-query. "
         "Self-referential queries (\"what is my name\") always inject via hard override.",
         size=10, color=GRAY, align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════════
# SLIDE 9 — BOUNDED CONVERSATIONAL MEMORY
# ══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
add_left_bar(s)
add_title(s, "Bounded Conversational Memory")

add_text(s, Inches(0.5), Inches(1.3), Inches(6.5), Inches(0.4),
         "The ConversationBuffer", size=15, bold=True)
add_bullets(s, Inches(0.5), Inches(1.75), Inches(6.5), Inches(3.5),
            [
                "Keeps last N = 10 turn-pairs verbatim in context.",
                "Summarizer fires on EVERY turn — rolling summary (≤ 80 tokens) always covers full history.",
                "Eviction from verbatim window is loss-free (already summarized).",
                "Assistant turns truncated to 400 characters before buffering.",
                "No-LLM fallback: if summarization fails, truncation-only summary used.",
            ], size=13)

add_callout(s, Inches(0.5), Inches(4.6), Inches(6.5), Inches(2.2),
            "**Context sent to model each turn:**\n"
            "[System]: \"Earlier: Alex asked about recursion, REST APIs…\"\n"
            "[User/Asst]: last 10 turns verbatim\n"
            "[User]: current query")

# Right: memory diagram
diag_bg = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                             Inches(7.3), Inches(1.3),
                             Inches(5.5), Inches(5.5))
diag_bg.fill.solid(); diag_bg.fill.fore_color.rgb = RGBColor(0xF8, 0xF8, 0xF8)
diag_bg.line.fill.background()

add_text(s, Inches(7.5), Inches(1.45), Inches(5.1), Inches(0.35),
         "Sliding Window Visualization", size=13, bold=True)

# Turn 4 evicted
add_text(s, Inches(7.5), Inches(2.0), Inches(0.8), Inches(0.35),
         "Turn 4", size=11, bold=True, color=GRAY)
add_box(s, Inches(8.4), Inches(2.0), Inches(1.5), Inches(0.35),
        RGBColor(0xDD, 0xDD, 0xDD), text="evicted ✗",
        text_size=10, text_color=GRAY)
add_text(s, Inches(10.0), Inches(2.0), Inches(2.8), Inches(0.35),
         "→ compressed into summary", size=9, color=GRAY,
         anchor=MSO_ANCHOR.MIDDLE)

# Rolling summary
add_box(s, Inches(8.4), Inches(2.5), Inches(4.2), Inches(0.4),
        RED, text="Rolling Summary (≤ 80 tokens)",
        text_size=10, text_bold=True, rounded=True)

# Turns 5, 6, 7
for i, turn in enumerate([5, 6, 7]):
    y = Inches(3.2 + i * 0.55)
    add_text(s, Inches(7.5), y, Inches(0.8), Inches(0.4),
             f"Turn {turn}", size=11, bold=True, color=GRAY)
    label = "verbatim (current)" if turn == 7 else "verbatim"
    add_box(s, Inches(8.4), y, Inches(4.2), Inches(0.4),
            BLUE1, text=label, text_size=11, text_bold=True,
            rounded=True)

# ══════════════════════════════════════════════════════════════════
# SLIDE 10 — EMPIRICAL PERFORMANCE
# ══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
add_left_bar(s)
add_title(s, "Empirical Performance")

# Big stats row
stats_y = Inches(1.3)
stats_w = Inches(2.95)
gap = Inches(0.15)
for i, (num, lbl) in enumerate([
    ("92.6%", "Routing Accuracy\n(1,000-case benchmark)"),
    ("85.3%", "ML-only rate\n(zero API cost)"),
    ("14.7%", "LLM Cascade rate\n(down from 30.8%)"),
    ("< 5ms", "ML router latency\n(local CPU, no GPU)"),
]):
    x = Inches(0.5) + (stats_w + gap) * i
    add_stat_box(s, x, stats_y, stats_w, Inches(1.3), num, lbl)

# Router accuracy table
add_text(s, Inches(0.5), Inches(2.9), Inches(5.5), Inches(0.3),
         "Router Accuracy", size=12, bold=True)
acc_rows = [
    ("Metric", "Value"),
    ("Seed Dataset Size", "280 examples"),
    ("Eval Set Size", "1,000 queries (21 categories)"),
    ("Held-out Accuracy (v0)", "95.5% (N=22 split)"),
    ("Benchmark Accuracy (v1)", "92.6% (1k benchmark)"),
    ("F1 Score", "0.922"),
    ("Recall (big class)", "0.963"),
]
at = s.shapes.add_table(len(acc_rows), 2,
                        Inches(0.5), Inches(3.25),
                        Inches(6.0), Inches(3.5)).table
at.columns[0].width = Inches(3.2); at.columns[1].width = Inches(2.8)
for i, row in enumerate(acc_rows):
    for j, val in enumerate(row):
        c = at.cell(i, j); c.text = ""
        p = c.text_frame.paragraphs[0]
        r = p.add_run(); r.text = val
        r.font.size = Pt(11); r.font.name = "Segoe UI"
        r.font.bold = (i == 0 or (i > 0 and j == 1 and "280" in val or "1,000" in val or "92.6" in val or "0.922" in val or "0.963" in val))
        if i == 0:
            r.font.color.rgb = WHITE; r.font.bold = True
            c.fill.solid(); c.fill.fore_color.rgb = BLACK
        else:
            r.font.color.rgb = BLACK
            c.fill.solid()
            c.fill.fore_color.rgb = WHITE if i % 2 == 1 else RGBColor(0xE8, 0xEC, 0xF0)

# Test suite on right
add_text(s, Inches(7.0), Inches(2.9), Inches(5.8), Inches(0.3),
         "System Robustness — 38 Tests Passed", size=12, bold=True)

test_data = [
    ("17 / 17", "Features &\nRouter Tests"),
    ("9 / 9",   "Memory Buffer\nTests"),
    ("12 / 12", "Integration &\nLogic Tests"),
]
for i, (num, lbl) in enumerate(test_data):
    x = Inches(7.0) + (Inches(1.95) + Inches(0.1)) * i
    add_stat_box(s, x, Inches(3.25), Inches(1.95), Inches(1.4),
                 num, lbl, num_color=GREEN)

add_callout(s, Inches(7.0), Inches(4.9), Inches(5.8), Inches(1.5),
            "**Result:**  Confident \"Small\" routes skip the LLM classifier entirely — ~85% of queries incur zero API cost for routing. F1 = 0.922 on 1,000-case benchmark.")

# ══════════════════════════════════════════════════════════════════
# SLIDE 11 — PER-CATEGORY RESULTS
# ══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
add_left_bar(s)
add_title(s, "Per-Category Results — 21 Categories")

cats_left = [
    ("analysis", "100%", GREEN),
    ("career_learning", "100%", GREEN),
    ("comparison_technical ★", "100%", GREEN),
    ("compound_questions", "100%", GREEN),
    ("conversions", "100%", GREEN),
    ("definitions", "100%", GREEN),
    ("domain_specific", "100%", GREEN),
    ("math_simple", "100%", GREEN),
    ("multi_constraint", "100%", GREEN),
    ("proofs", "100%", GREEN),
    ("real_world_planning", "100%", GREEN),
]
cats_right = [
    ("research_synthesis", "100%", GREEN),
    ("short_creative", "100%", GREEN),
    ("system_design", "100%", GREEN),
    ("factual", "96.1%", ORANGE),
    ("yes_no", "95.6%", ORANGE),
    ("greetings", "93.8%", ORANGE),
    ("ambiguous", "90.0%", ORANGE),
    ("conversational ★", "80.0%", ORANGE),
    ("ethical_philosophical ★", "75.0%", ORANGE),
    ("debug_simple ★ ← next target", "33.3%", RED),
]

def draw_cat_row(slide, left, top, name, acc, color, highlight=False):
    if highlight:
        hl = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                    left - Inches(0.05), top,
                                    Inches(5.9), Inches(0.4))
        hl.fill.solid(); hl.fill.fore_color.rgb = RGBColor(0xFD, 0xF0, 0xF0)
        hl.line.fill.background()
    add_text(slide, left, top, Inches(4.5), Inches(0.4),
             name, size=11,
             bold=(highlight), color=(RED if highlight else BLACK),
             anchor=MSO_ANCHOR.MIDDLE)
    add_text(slide, left + Inches(4.5), top, Inches(1.3), Inches(0.4),
             acc, size=11, bold=True, color=color,
             align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    # underline
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                  left, top + Inches(0.38),
                                  Inches(5.8), Inches(0.008))
    line.fill.solid(); line.fill.fore_color.rgb = RGBColor(0xF0, 0xF0, 0xF0)
    line.line.fill.background()

for i, (name, acc, color) in enumerate(cats_left):
    draw_cat_row(s, Inches(0.5), Inches(1.3 + i * 0.42), name, acc, color)
for i, (name, acc, color) in enumerate(cats_right):
    highlight = (i == len(cats_right) - 1)
    draw_cat_row(s, Inches(6.9), Inches(1.3 + i * 0.42), name, acc, color,
                 highlight=highlight)

add_callout(s, Inches(0.5), Inches(6.1), Inches(12.3), Inches(1.0),
            "★ New categories in Eval-1000  |  **14 of 21 categories at 100%**  |  "
            "debug_simple fails because \"why doesn't my CSS apply?\" triggers "
            "HAS_REASONING_KW — fix: add HAS_DEBUG_SIMPLE tag + 30 seed examples.")

# ══════════════════════════════════════════════════════════════════
# SLIDE 12 — BEFORE / AFTER
# ══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
add_left_bar(s)
add_title(s, "Before / After — v0 → v1 Improvement")

# v0 box
v0 = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                        Inches(0.5), Inches(1.3),
                        Inches(5.5), Inches(4.6))
v0.fill.solid(); v0.fill.fore_color.rgb = RGBColor(0xF8, 0xF8, 0xF8)
v0.line.fill.background()
add_text(s, Inches(0.7), Inches(1.45), Inches(5.1), Inches(0.4),
         "v0 — BASELINE", size=14, bold=True, color=GRAY)

v0_data = [
    ("Training examples", "120"),
    ("ngram range", "1–2 grams"),
    ("TF-IDF features", "3,000"),
    ("Feature tags", "8 tags"),
    ("Accuracy", "88.2%"),
    ("Cascade rate", "30.8%"),
    ("ML-only rate", "69.2%"),
]
for i, (k, v) in enumerate(v0_data):
    y = Inches(2.0 + i * 0.48)
    add_text(s, Inches(0.7), y, Inches(3.5), Inches(0.4),
             k, size=12, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, Inches(4.0), y, Inches(1.8), Inches(0.4),
             v, size=12, bold=True, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    # divider
    if i < len(v0_data) - 1:
        line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                  Inches(0.7), y + Inches(0.44),
                                  Inches(5.1), Inches(0.008))
        line.fill.solid(); line.fill.fore_color.rgb = BORDER
        line.line.fill.background()

# arrow
arr = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                         Inches(6.15), Inches(3.3),
                         Inches(0.7), Inches(0.6))
arr.fill.solid(); arr.fill.fore_color.rgb = RED
arr.line.fill.background()

# v1 box
v1 = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                        Inches(7.05), Inches(1.3),
                        Inches(5.8), Inches(4.6))
v1.fill.solid(); v1.fill.fore_color.rgb = RGBColor(0xFD, 0xF0, 0xF0)
v1.line.color.rgb = RED; v1.line.width = Pt(2)
add_text(s, Inches(7.25), Inches(1.45), Inches(5.4), Inches(0.4),
         "v1 — IMPROVED", size=14, bold=True, color=RED)

v1_data = [
    ("Training examples", "280 (+133%)"),
    ("ngram range", "1–3 grams"),
    ("TF-IDF features", "5,000"),
    ("Feature tags", "16 tags"),
    ("Accuracy", "92.6% (+4.4 pp)"),
    ("Cascade rate", "14.7% (−16.1 pp)"),
    ("ML-only rate", "85.3% (+16.1 pp)"),
]
for i, (k, v) in enumerate(v1_data):
    y = Inches(2.0 + i * 0.48)
    add_text(s, Inches(7.25), y, Inches(3.5), Inches(0.4),
             k, size=12, anchor=MSO_ANCHOR.MIDDLE)
    add_text(s, Inches(10.7), y, Inches(2.0), Inches(0.4),
             v, size=12, bold=True, color=RED, align=PP_ALIGN.RIGHT,
             anchor=MSO_ANCHOR.MIDDLE)
    if i < len(v1_data) - 1:
        line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                  Inches(7.25), y + Inches(0.44),
                                  Inches(5.4), Inches(0.008))
        line.fill.solid(); line.fill.fore_color.rgb = RGBColor(0xF3, 0xD0, 0xD0)
        line.line.fill.background()

# Bottom callouts
add_callout(s, Inches(0.5), Inches(6.15), Inches(6.15), Inches(1.0),
            "**Biggest wins:** short_creative cascade 100% → 11.1% after adding "
            "HAS_CREATIVE_KW + 25 examples. debug_simple is the remaining gap (33.3%).")
add_callout(s, Inches(6.85), Inches(6.15), Inches(6.0), Inches(1.0),
            "**Self-improving loop:** Every routing decision is logged. "
            "Export high-confidence labels → retrain → redeploy in minutes.",
            bg=RGBColor(0xF0, 0xFF, 0xF4), accent=GREEN)

# ══════════════════════════════════════════════════════════════════
# SLIDE 13 — SUMMARY & FUTURE WORK
# ══════════════════════════════════════════════════════════════════
s = prs.slides.add_slide(blank)
add_left_bar(s)
add_title(s, "Summary & Future Work")

# Limitations
add_text(s, Inches(0.5), Inches(1.3), Inches(6.0), Inches(0.4),
         "Limitations", size=15, bold=True)
add_bullets(s, Inches(0.5), Inches(1.75), Inches(6.0), Inches(4),
            [
                "Small held-out set in v0 (N=22); v1 uses 1,000-case benchmark.",
                "Binary labels only (Small vs Big) — fine-grained routing not yet.",
                "Uncalibrated τ threshold (0.65 empirical, not optimized).",
                "Residual profile leakage — name stub still sent on impersonal queries.",
                "Relies on proprietary OpenAI API — model deprecation risk.",
                "debug_simple at 33.3% — immediate next fix.",
            ], size=12)

# Next Steps
add_text(s, Inches(6.8), Inches(1.3), Inches(6.0), Inches(0.4),
         "Next Steps", size=15, bold=True)
add_bullets(s, Inches(6.8), Inches(1.75), Inches(6.0), Inches(4),
            [
                "Add HAS_DEBUG_SIMPLE tag + 30 seed examples.",
                "Export agreement-only logs → retrain Router v2.",
                "Expand to 4+ routing classes (code, math, reasoning, factual).",
                "Online A/B testing against LLM-only baseline.",
                "Port to open-weight models (Llama-3, Mistral).",
                "Energy-efficient edge deployment (Raspberry Pi / mobile).",
            ], size=12)

# Closing dark callout
close = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                           Inches(0.5), Inches(5.9),
                           Inches(12.3), Inches(1.3))
close.fill.solid(); close.fill.fore_color.rgb = BLACK
close.line.fill.background()

add_text(s, Inches(0.5), Inches(6.0), Inches(12.3), Inches(0.5),
         "Questions?", size=22, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_text(s, Inches(0.5), Inches(6.55), Inches(12.3), Inches(0.35),
         "Code:  github.com/sandeepvijayarao09/dual-llm-system",
         size=13, color=RGBColor(0xFF, 0xA0, 0xA0), align=PP_ALIGN.CENTER,
         font="Courier New")
add_text(s, Inches(0.5), Inches(6.9), Inches(12.3), Inches(0.3),
         "Zichen Qi  ·  Yalin Sun  ·  Sandeep Vijayarao   ·   April 2026",
         size=11, color=RGBColor(0xBB, 0xBB, 0xBB), align=PP_ALIGN.CENTER)

# ── Save ──────────────────────────────────────────────────────────
out_path = "presentation.pptx"
prs.save(out_path)
print(f"Saved: {out_path}  ({len(prs.slides)} slides)")
