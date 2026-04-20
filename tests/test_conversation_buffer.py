"""Tests for memory.conversation_buffer.ConversationBuffer."""

from memory import ConversationBuffer


def test_buffer_starts_empty():
    buf = ConversationBuffer(window_size=3)
    assert buf.recent_turns == []
    assert buf.rolling_summary == ""
    assert buf.build_history() == []


def test_single_turn_kept_verbatim():
    buf = ConversationBuffer(window_size=3)
    buf.add_turn("hello", "hi there")
    history = buf.build_history()
    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    assert buf.rolling_summary == ""


def test_window_eviction_degrades_without_llm():
    """With no small_llm wired up the buffer still respects window_size."""
    buf = ConversationBuffer(window_size=2)   # 2 turn-pairs = 4 messages

    buf.add_turn("q1", "a1")
    buf.add_turn("q2", "a2")
    assert len(buf.recent_turns) == 4         # at capacity, nothing evicted

    buf.add_turn("q3", "a3")
    # q1/a1 should be evicted into the summary
    assert len(buf.recent_turns) == 4
    contents = [m["content"] for m in buf.recent_turns]
    assert "q1" not in contents
    assert "q3" in contents
    assert "q1" in buf.rolling_summary        # degraded summarizer appended it


def test_build_history_includes_summary_when_present():
    buf = ConversationBuffer(window_size=1)

    buf.add_turn("old q", "old a")
    buf.add_turn("new q", "new a")            # evicts the first pair

    history = buf.build_history()
    assert history[0]["role"] == "system"
    assert "summary" in history[0]["content"].lower()
    # Remaining entries are the verbatim recent turns
    assert history[1]["content"] == "new q"
    assert history[2]["content"] == "new a"


def test_clear_resets_everything():
    buf = ConversationBuffer(window_size=1)
    buf.add_turn("a", "b")
    buf.add_turn("c", "d")
    assert buf.rolling_summary != ""

    buf.clear()
    assert buf.recent_turns == []
    assert buf.rolling_summary == ""
    assert buf.build_history() == []


def test_summarizer_called_on_eviction(fake_small_llm):
    """When a small_llm is wired up, it drives the summary update."""
    fake_small_llm.default_response = "Summary: user greeted, asked about foo."
    buf = ConversationBuffer(small_llm=fake_small_llm, window_size=1)

    buf.add_turn("hi", "hello")
    assert len(fake_small_llm.calls) == 0      # no eviction yet

    buf.add_turn("what about foo?", "foo is bar")
    assert len(fake_small_llm.calls) == 1      # eviction triggered summarize
    assert buf.rolling_summary == "Summary: user greeted, asked about foo."


def test_multiple_evictions_pass_previous_summary(fake_small_llm):
    """Rolling summary is incremental: previous summary is fed into next call."""
    responses = iter(["SUM1", "SUM2"])
    fake_small_llm.script(lambda _s, _u: True, "")  # placeholder
    fake_small_llm.scripts = []                      # reset

    def _resp(_s, _u):
        return next(responses, "FINAL")
    fake_small_llm.script(lambda s, u: True, "")     # won't actually be used
    # Override complete directly to yield a different value each call
    call_log: list[str] = []
    original_complete = fake_small_llm.complete
    def complete(system, user_message, **kwargs):
        call_log.append(user_message)
        return next(responses, "FINAL")
    fake_small_llm.complete = complete               # type: ignore[assignment]

    buf = ConversationBuffer(small_llm=fake_small_llm, window_size=1)
    buf.add_turn("q1", "a1")
    buf.add_turn("q2", "a2")    # evicts q1/a1 → produces SUM1
    assert buf.rolling_summary == "SUM1"

    buf.add_turn("q3", "a3")    # evicts q2/a2 → produces SUM2
    assert buf.rolling_summary == "SUM2"

    # Second summarize call must include previous summary "SUM1" as input
    assert "SUM1" in call_log[1]


def test_assistant_truncation(fake_small_llm):
    """Very long assistant answers are truncated before feeding the summarizer."""
    long_answer = "x" * 10_000
    received_user_msg: list[str] = []

    def complete(system, user_message, **kwargs):
        received_user_msg.append(user_message)
        return "S"

    fake_small_llm.complete = complete   # type: ignore[assignment]

    buf = ConversationBuffer(
        small_llm=fake_small_llm, window_size=1, max_assistant_chars=400,
    )
    buf.add_turn("q1", long_answer)
    buf.add_turn("q2", "a2")             # evict q1/long_answer

    # The user_message fed to the summarizer must be far shorter than 10k
    # (it should only contain ~400 chars of the assistant text plus framing)
    assert len(received_user_msg[0]) < 2000
    assert "…[truncated]" in received_user_msg[0]


def test_snapshot_shape():
    buf = ConversationBuffer(window_size=2)
    buf.add_turn("q", "a")
    snap = buf.snapshot()
    assert set(snap.keys()) == {
        "window_size", "recent_turn_count", "has_summary", "summary_chars",
    }
    assert snap["window_size"] == 2
    assert snap["recent_turn_count"] == 2
    assert snap["has_summary"] is False
