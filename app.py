"""
Dual LLM System — Streamlit Demo UI

Run:  streamlit run app.py
"""

import json
import streamlit as st
from orchestrator import Orchestrator

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dual LLM System", page_icon="🧠", layout="wide")

# ── Init orchestrator (cached so it only loads once) ──────────────────────────
@st.cache_resource
def get_orchestrator():
    return Orchestrator()

orch = get_orchestrator()

# ── Preset profiles ──────────────────────────────────────────────────────────
PRESET_PROFILES = {
    "cs_student": "🎓 Alex — CS College Student",
    "nasa_engineer": "🚀 Dr. Sarah Chen — NASA Engineer",
    "high_school": "📚 Jordan — High School Student",
    "guest": "👤 Guest (blank profile)",
}

# ── Sidebar: User profile ────────────────────────────────────────────────────
st.sidebar.title("👤 Switch User")

# Profile switcher
selected_key = st.sidebar.selectbox(
    "Select a persona",
    options=list(PRESET_PROFILES.keys()),
    format_func=lambda k: PRESET_PROFILES[k],
)

# Track profile switches to clear chat
if "current_user" not in st.session_state:
    st.session_state.current_user = selected_key
if st.session_state.current_user != selected_key:
    orch.clear_buffer(st.session_state.current_user)
    st.session_state.current_user = selected_key
    st.session_state.messages = []
    st.session_state.session_queries = []
    st.rerun()

user_id = selected_key

# Load profile from DB
profile = orch.load_profile(user_id)

st.sidebar.markdown("---")
st.sidebar.subheader("Profile Details")

# Display profile info (read-only for presets)
st.sidebar.markdown(f"**Name:** {profile.get('name') or 'Guest'}")
st.sidebar.markdown(f"**Expertise:** {profile.get('expertise', 'unknown')}")
st.sidebar.markdown(f"**Tone:** {profile.get('tone', 'unknown')}")
st.sidebar.markdown(f"**Domain:** {profile.get('domain') or 'not set'}")

interests = profile.get("interests", [])
if interests:
    st.sidebar.markdown(f"**Interests:** {', '.join(interests)}")

background = profile.get("background", "")
if background:
    st.sidebar.markdown("---")
    st.sidebar.caption("Background")
    st.sidebar.markdown(f"*{background}*")

st.sidebar.markdown("---")
st.sidebar.caption(f"Sessions: {profile.get('interaction_count', 0)}")
topics = profile.get("topics_discussed", [])
if topics:
    st.sidebar.caption(f"Topics discussed: {', '.join(topics)}")

# ── Main area: Chat ──────────────────────────────────────────────────────────
st.title("🧠 Dual LLM System")

# Show who is active
col_name, col_badge = st.columns([3, 1])
with col_name:
    st.caption(f"Chatting as **{profile.get('name') or 'Guest'}** — {profile.get('domain') or 'no domain'} — {profile.get('expertise', 'unknown')} level")
with col_badge:
    if profile.get("expertise") == "expert":
        st.markdown("🔴 Expert")
    elif profile.get("expertise") == "intermediate":
        st.markdown("🟡 Intermediate")
    else:
        st.markdown("🟢 Novice")

st.markdown("---")

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_queries" not in st.session_state:
    st.session_state.session_queries = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("metadata"):
            with st.expander("🔍 Routing Details"):
                st.json(msg["metadata"])

# Chat input
st.caption("💡 Slash commands: `/large <query>` — force Large LLM &nbsp;|&nbsp; `/small <query>` — force Small LLM")
if prompt := st.chat_input("Ask anything... (or use /large /small to force a model)"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Track query for profile update
    st.session_state.session_queries.append(prompt)

    # Process with orchestrator — pass user_id so the per-user
    # ConversationBuffer (sliding window + rolling summary) kicks in.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = orch.process(
                query=prompt,
                user_id=user_id,
                user_profile=profile,
                verbose=False,
            )

        st.markdown(resp.answer)

        # Routing badge
        rd = resp.routing_decision
        if rd == "forced:large":
            icon = "⚡ Forced → Large LLM (GPT-4o)"
        elif rd == "forced:small":
            icon = "⚡ Forced → Small LLM (GPT-4o Mini)"
        elif "small" in rd:
            icon = "🏠 Small LLM (GPT-4o Mini)"
        else:
            icon = "☁️ Large LLM (GPT-4o)"

        metadata = {
            "routing": f"{icon}",
            "decision": resp.routing_decision,
            "routed_by": resp.routed_by,
            "model": resp.model_used,
            "latency": f"{resp.latency_ms:.0f} ms",
            "context_note": resp.context_note or "(none)",
            "ml_prediction": resp.ml_prediction or "(not used)",
            "classification": resp.classification or "(not used)",
            "buffer": resp.buffer_state or "(no buffer)",
        }

        with st.expander("🔍 Routing Details"):
            st.json(metadata)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": resp.answer,
        "metadata": metadata,
    })

# ── Bottom bar: Session controls ──────────────────────────────────────────────
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_queries = []
        orch.clear_buffer(user_id)
        st.rerun()

with col2:
    if st.button("📊 Update Profile from Session"):
        if st.session_state.session_queries and user_id != "guest":
            updated = orch.end_session(
                user_id,
                st.session_state.session_queries,
            )
            st.success(
                f"Profile updated! "
                f"Interests: {updated.get('interests')} | "
                f"Expertise: {updated.get('expertise')}"
            )
            st.session_state.session_queries = []
        else:
            st.warning("No queries to analyze or using guest mode.")

with col3:
    if st.button("🔄 Reset Profile"):
        if user_id != "guest":
            orch.profile_db.delete(user_id)
            st.success(f"Profile for '{user_id}' reset.")
            st.rerun()
