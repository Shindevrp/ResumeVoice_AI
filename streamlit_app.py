from __future__ import annotations

import json
from urllib.request import urlopen

import streamlit as st

st.set_page_config(page_title="ResumeVoice AI", layout="wide")

BACKEND = "http://localhost:8000"


def fetch_sessions() -> list[dict]:
    try:
        with urlopen(f"{BACKEND}/sessions", timeout=2) as resp:
            return json.load(resp).get("sessions", [])
    except Exception:
        return []


def latest_session() -> dict | None:
    sessions = fetch_sessions()
    return sessions[-1] if sessions else None


st.markdown(
    """
<style>
    .stApp { background: radial-gradient(ellipse at 50% 0%, #0B1A2E 0%, #020617 70%); }
    .stApp > header { display: none; }
    #MainMenu, footer { display: none; }
    .block-container { padding: 1rem; max-width: 100% !important; }
    iframe { border: none; width: 100%; height: 80vh; }
</style>
""",
    unsafe_allow_html=True,
)


@st.fragment(run_every="2s")
def session_status() -> None:
    session = latest_session()
    st.subheader("Live Session")
    if session is None:
        st.write("Topic:", "—")
        st.write("Intent:", "—")
        st.write("State: offline")
        return
    st.write("Topic:", session.get("topic") or "—")
    st.write("Intent:", session.get("intent") or "—")
    st.write("State:", session.get("state") or "—")
    st.write(
        "Engagement:",
        f"{session.get('engagement_score', 0):.2f}",
    )
    st.write("User turns:", session.get("total_user_turns", 0))
    st.write("Session:", session.get("session_id", "—"))


session_status()

st.iframe(f"{BACKEND}/ui", height=800)
