"""
Dual LLM System — Interactive CLI Demo

Usage:
  python main.py              # interactive chat
  python main.py --demo       # run predefined demo queries
  python main.py --verbose    # show routing decisions in real-time
"""

import argparse
import sys
from orchestrator import Orchestrator

# ── Demo queries ──────────────────────────────────────────────────────────────
DEMO_QUERIES = [
    # Simple → Small LLM
    ("Hi! How are you?",
     {"tone": "casual", "length": "brief"}),

    # Simple factual → Small LLM
    ("What is the capital of Japan?",
     None),

    # Complex reasoning → Large LLM
    ("Explain how transformer self-attention works and why it scales better than RNNs.",
     None),

    # Complex + personalise → Large LLM → Personalizer
    ("What are the main differences between TCP and UDP protocols?",
     {"tone": "casual", "length": "brief", "format": "bullets", "expertise": "novice"}),

    # Math → complex → Large LLM
    ("Prove that the square root of 2 is irrational.",
     None),
]


# ── CLI helper ────────────────────────────────────────────────────────────────

def print_response(resp, verbose: bool = False) -> None:
    print("\n" + "─" * 60)
    print(resp.answer)
    print("─" * 60)

    if verbose:
        icon = "🏠" if "small" in resp.routing_decision else "☁️"
        personalised = " + personalised" if resp.personalizer_applied else ""
        print(
            f"{icon} Routed to: {resp.routing_decision}{personalised} | "
            f"Model: {resp.model_used} | "
            f"Latency: {resp.latency_ms:.0f} ms"
        )
        if resp.classification:
            clf = resp.classification
            print(
                f"   Classifier → complexity={clf.get('complexity')} | "
                f"intent={clf.get('intent')} | "
                f"confidence={clf.get('confidence')}"
            )
    print()


# ── Demo mode ─────────────────────────────────────────────────────────────────

def run_demo(orch: Orchestrator, verbose: bool) -> None:
    print("=" * 60)
    print("  Dual LLM System — Demo Mode")
    print("=" * 60)

    for i, (query, profile) in enumerate(DEMO_QUERIES, 1):
        print(f"\n[{i}/{len(DEMO_QUERIES)}] Query: {query}")
        if profile:
            print(f"         Profile: {profile}")
        resp = orch.process(query, user_profile=profile, verbose=verbose)
        print_response(resp, verbose)


# ── Interactive mode ──────────────────────────────────────────────────────────

def run_interactive(orch: Orchestrator, verbose: bool) -> None:
    print("=" * 60)
    print("  Dual LLM System — Interactive Chat")
    print("  Type 'quit' to exit | 'profile' to set user preferences")
    print("=" * 60 + "\n")

    history: list[dict] = []
    user_profile: dict = {}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if user_input.lower() == "profile":
            user_profile = set_profile()
            print(f"Profile set: {user_profile}\n")
            continue

        if user_input.lower() == "clear":
            history.clear()
            print("Conversation history cleared.\n")
            continue

        resp = orch.process(
            user_input,
            history=history if history else None,
            user_profile=user_profile or None,
            verbose=verbose,
        )

        print_response(resp, verbose)

        # Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": resp.answer})

        # Keep last 10 turns
        if len(history) > 20:
            history = history[-20:]


def set_profile() -> dict:
    print("\nSet your preferences (press Enter to skip):")
    tone = input("  Tone [casual/formal/technical]: ").strip() or None
    length = input("  Length [brief/medium/detailed]: ").strip() or None
    fmt = input("  Format [prose/bullets/plain]: ").strip() or None
    expertise = input("  Expertise [novice/intermediate/expert]: ").strip() or None
    return {k: v for k, v in {
        "tone": tone, "length": length, "format": fmt, "expertise": expertise
    }.items() if v}


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Dual LLM System")
    parser.add_argument("--demo", action="store_true", help="Run predefined demo queries")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show routing metadata")
    args = parser.parse_args()

    try:
        orch = Orchestrator()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)

    if args.demo:
        run_demo(orch, args.verbose)
    else:
        run_interactive(orch, args.verbose)


if __name__ == "__main__":
    main()
