"""
Dual LLM System — Interactive CLI Demo

Usage:
  python main.py                        # interactive chat (guest user)
  python main.py --user sandeep         # chat as a named user (profile persists)
  python main.py --demo                 # run predefined demo queries
  python main.py --verbose              # show routing decisions in real-time
  python main.py --user sandeep --show-profile   # print current saved profile
"""

import argparse
import json
import logging
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

    # Complex + profile → Large LLM personalized
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
        print(
            f"{icon} Routed to: {resp.routing_decision} | "
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
        if resp.context_note:
            print(f"   Context note added ✅")
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

def run_interactive(orch: Orchestrator, user_id: str, verbose: bool) -> None:
    print("=" * 60)
    print("  Dual LLM System — Interactive Chat")
    print(f"  User: {user_id}")
    print("  Commands: 'quit' | 'profile' | 'clear' | 'show profile'")
    print("=" * 60 + "\n")

    # Load persisted profile for this user
    user_profile = orch.load_profile(user_id)
    if user_profile.get("interaction_count", 0) > 0:
        print(f"  Welcome back! Session #{user_profile['interaction_count'] + 1}")
        print(f"  Known profile: expertise={user_profile.get('expertise')} | "
              f"domain={user_profile.get('domain') or 'not set'} | "
              f"interests={user_profile.get('interests') or []}\n")
    else:
        print("  Welcome! Your profile will adapt as we chat.\n")

    history: list[dict] = []
    session_queries: list[str] = []    # track queries for profile update

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            _end_session(orch, user_id, session_queries, verbose)
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            _end_session(orch, user_id, session_queries, verbose)
            print("Goodbye!")
            break

        if user_input.lower() == "show profile":
            print(f"\nCurrent profile:\n{json.dumps(user_profile, indent=2)}\n")
            continue

        if user_input.lower() == "profile":
            overrides = _set_profile_manual()
            user_profile.update(overrides)
            print(f"Profile updated: {overrides}\n")
            continue

        if user_input.lower() == "clear":
            history.clear()
            print("Conversation history cleared.\n")
            continue

        # Track query for session-end profile update
        session_queries.append(user_input)

        resp = orch.process(
            user_input,
            history=history if history else None,
            user_profile=user_profile,
            verbose=verbose,
        )

        print_response(resp, verbose)

        # Update conversation history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": resp.answer})

        # Keep last 10 turns
        if len(history) > 20:
            history = history[-20:]


def _end_session(
    orch: Orchestrator,
    user_id: str,
    session_queries: list[str],
    verbose: bool,
) -> None:
    """Trigger dynamic profile update at end of session."""
    if session_queries and user_id != "guest":
        print("\n  📊 Updating your profile based on this session...")
        updated = orch.end_session(user_id, session_queries)
        print(f"  ✅ Profile updated — "
              f"interests: {updated.get('interests')} | "
              f"expertise: {updated.get('expertise')}\n")


def _set_profile_manual() -> dict:
    print("\nOverride preferences (press Enter to skip):")
    tone      = input("  Tone [casual/formal/technical]: ").strip() or None
    length    = input("  Length [brief/medium/detailed]: ").strip() or None
    fmt       = input("  Format [prose/bullets/plain]: ").strip() or None
    expertise = input("  Expertise [novice/intermediate/expert]: ").strip() or None
    domain    = input("  Domain [e.g. AI, finance, medicine]: ").strip() or None
    return {k: v for k, v in {
        "tone": tone, "length": length, "format": fmt,
        "expertise": expertise, "domain": domain,
    }.items() if v}


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Dual LLM System")
    parser.add_argument("--demo",         action="store_true", help="Run predefined demo queries")
    parser.add_argument("--verbose", "-v",action="store_true", help="Show routing metadata")
    parser.add_argument("--user",         type=str, default="guest", help="User ID for persistent profile")
    parser.add_argument("--show-profile", action="store_true", help="Print saved profile and exit")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        orch = Orchestrator()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)

    if args.show_profile:
        profile = orch.load_profile(args.user)
        print(f"\nProfile for '{args.user}':")
        print(json.dumps(profile, indent=2))
        return

    if args.demo:
        run_demo(orch, args.verbose)
    else:
        run_interactive(orch, user_id=args.user, verbose=args.verbose)


if __name__ == "__main__":
    main()
