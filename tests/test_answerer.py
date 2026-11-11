"""SimpleAnswerer prompt selection — which system prompt a profile gets."""

from modules.answerer import ANSWERER_SYSTEM_WITH_PROFILE, SimpleAnswerer


def _sent(fake_small_llm, profile):
    SimpleAnswerer(fake_small_llm).answer("Hello, how's it going?", None, profile)
    return fake_small_llm.calls[-1]


def test_name_only_stub_gets_the_plain_prompt(fake_small_llm):
    """The tailoring prompt made Gemma 4 E4B invent a background 6/6 times
    when it was given only a name; the plain prompt plus the name doesn't."""
    call = _sent(fake_small_llm, {"name": "Alex"})
    assert "domain, background, and interests" not in call["system"]
    assert "never guess or invent" in call["system"]
    assert call["user"].endswith("[User's name: Alex]")


def test_real_profile_gets_the_tailoring_prompt(fake_small_llm):
    call = _sent(fake_small_llm, {"name": "Alex", "domain": "Computer Science",
                                  "interests": ["algorithms"]})
    assert call["system"].startswith(ANSWERER_SYSTEM_WITH_PROFILE.split("\n")[0])
    assert "Use ONLY the facts in [User context]" in call["system"]
    assert "Domain: Computer Science" in call["user"]


def test_no_profile_sends_the_query_unchanged(fake_small_llm):
    call = _sent(fake_small_llm, None)
    assert call["user"] == "Hello, how's it going?"
