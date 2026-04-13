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

# ── Sidebar: User profile ────────────────────────────────────────────────────
st.sidebar.title("👤 User Profile")

user_id = st.sidebar.text_input("User ID", value="guest")

# Load profile from DB
profile = orch.load_profile(user_id)

st.sidebar.markdown("---")
st.sidebar.subheader("Current Profile")

# Editable profile fields
profile["name"] = st.sidebar.text_input("Name", value=profile.get("name", ""))
profile["expertise"] = st.sidebar.selectbox(
    "Expertise",
    ["novice", "intermediate", "expert"],
    index=["novice", "intermediate", "expert"].index(profile.get("expertise", "intermediate")),
)
profile["tone"] = st.sidebar.selectbox(
    "Tone",
    ["casual", "formal", "technical"],
    index=["casual", "formal", "technical"].index(profile.get("tone", "casual")),
)
profile["domain"] = st.sidebar.text_input("Domain", value=profile.get("domain", ""))

interests_str = ", ".join(profile.get("interests", []))
interests_input = st.sidebar.text_input("Interests (comma separated)", value=interests_str)
profile["interests"] = [i.strip() for i in interests_input.split(",") if i.strip()]

profile["preferred_format"] = st.sidebar.selectbox(
    "Format",
    ["prose", "bullets", "plain"],
    index=["prose", "bullets", "plain"].index(profile.get("preferred_format", "prose")),
)

# Save profile button
if st.sidebar.button("💾 Save Profile"):
    orch.profile_db.save(user_id, profile)
    st.sidebar.success("Profile saved!")

st.sidebar.markdown("---")
st.sidebar.caption(f"Sessions: {profile.get('interaction_count', 0)}")
st.sidebar.caption(f"Topics: {profile.get('topics_discussed', [])}")

# ── Main area: Chat ──────────────────────────────────────────────────────────
st.title("🧠 Dual LLM System")
st.caption("Two models. One experience. Built around you.")

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
            with st.expander("🔍 Details"):
                st.json(msg["metadata"])

# Chat input
if prompt := st.chat_input("Ask anything..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Track query for profile update
    st.session_state.session_queries.append(prompt)

    # Process with orchestrator
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            resp = orch.process(
                query=prompt,
                user_profile=profile,
                verbose=False,
            )

        st.markdown(resp.answer)

        # Show routing metadata
        icon = "🏠 Small LLM" if "small" in resp.routing_decision else "☁️ Large LLM"
        metadata = {
            "routing": f"{icon} ({resp.routing_decision})",
            "model": resp.model_used,
            "latency": f"{resp.latency_ms:.0f} ms",
            "context_note": resp.context_note or "(none)",
            "classification": resp.classification,
        }

        with st.expander("🔍 Details"):
            st.json(metadata)

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": resp.answer,
        "metadata": metadata,
    })

# ── Bottom bar: Session controls ──────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_queries = []
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
            st.rerun()
        else:
            st.warning("No queries to analyze or using guest mode.")

with col3:
    if st.button("🔄 Reset Profile"):
        if user_id != "guest":
            orch.profile_db.delete(user_id)
            st.success(f"Profile for '{user_id}' reset.")
            st.rerun()
