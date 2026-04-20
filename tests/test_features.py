"""Tests for router.features.encode_text."""

from router.features import encode_text


def test_short_query_gets_len_short_tag():
    assert "LEN_SHORT" in encode_text("Hi there")


def test_long_query_gets_len_long_tag():
    long_q = " ".join(["word"] * 50)
    assert "LEN_LONG" in encode_text(long_q)


def test_reasoning_keywords():
    for kw in ("why", "explain", "prove", "derive", "analyze"):
        tagged = encode_text(f"{kw} something happens")
        assert "HAS_REASONING_KW" in tagged, f"Missing tag for {kw}"


def test_factual_keywords():
    assert "HAS_FACTUAL_KW" in encode_text("What is the capital of France?")
    assert "HAS_FACTUAL_KW" in encode_text("Who is Obama?")


def test_greeting_keywords():
    assert "HAS_GREETING_KW" in encode_text("hi, how are you?")
    assert "HAS_GREETING_KW" in encode_text("Thanks a lot")
    assert "HAS_GREETING_KW" in encode_text("Good morning")


def test_code_tag():
    assert "HAS_CODE" in encode_text("```python\nprint('hi')\n```")
    assert "HAS_CODE" in encode_text("def foo(x): return x")


def test_math_tag():
    assert "HAS_MATH" in encode_text("prove integral of x dx")
    assert "HAS_MATH" in encode_text("what is 2 + 2")


def test_multiple_question_marks():
    assert "Q_MULTI" in encode_text("why? how? when?")


def test_empty_input_doesnt_crash():
    # Should return a reasonable tag string, not raise
    out = encode_text("")
    assert "LEN_SHORT" in out


def test_none_and_whitespace():
    assert "LEN_SHORT" in encode_text("   ")
