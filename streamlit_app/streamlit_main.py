"""
streamlit_main.py — Main Streamlit UI for the SMS Recruitment Chatbot.

Flow:
    1. Candidate types a message in the chat input.
    2. The message is passed to get_bot_response() (app/main.py), which
       runs the multi-agent pipeline (Main Agent + advisors).
    3. The agent returns an action ("continue" / "schedule" / "end")
       and a response message.
    4. The response is displayed in the chat and the action is shown
       in the debug sidebar.
    5. When action is "end", the input is disabled and the user can
       reset to start a new conversation.

Session state keys:
    messages          : list of {"role": "user"|"assistant", "content": str}
    conversation_ended: bool — True once the agent returns action="end"
    last_result       : dict — raw output from get_bot_response() for debugging

Run locally:
    streamlit run streamlit_app/streamlit_main.py
"""

import sys
import os

# Ensure the project root is on sys.path so app.* imports resolve correctly
# regardless of the directory Streamlit is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from app.main import get_bot_response

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="SMS Recruitment Chatbot", page_icon="💬", layout="centered")
st.title("💬 SMS Recruitment Chatbot")
st.caption("Python Developer Position")

# ---------------------------------------------------------------------------
# Session state — persists across Streamlit reruns within the same session
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_ended" not in st.session_state:
    st.session_state.conversation_ended = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ---------------------------------------------------------------------------
# DEBUG SIDEBAR — shows the agent's last action and full raw response.
# Remove before production.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Debug")
    if st.session_state.last_result:
        action = st.session_state.last_result.get("action")
        action_icon = {"continue": "🟡", "schedule": "🟢", "end": "🔴"}
        st.write(f"**Action:** {action_icon.get(action, '⚪')} `{action}`")
        st.divider()
        st.write("**Full agent response:**")
        st.json(st.session_state.last_result)
    else:
        st.write("No response yet.")
# --- END DEBUG SIDEBAR ---

# ---------------------------------------------------------------------------
# Chat history — re-rendered on every rerun from session state
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------------------------------------------------------------------------
# Input / response
# ---------------------------------------------------------------------------
if st.session_state.conversation_ended:
    # Lock the UI once the agent has ended the conversation
    st.info("This conversation has ended.")
    if st.button("Start New Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_ended = False
        st.session_state.last_result = None
        st.rerun()
else:
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # 1. Add candidate message to history and display it immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # 2. Call the main agent with the full conversation history
        with st.spinner("..."):
            result = get_bot_response(st.session_state.messages)

        # 3. Unpack the agent response
        st.session_state.last_result = result
        bot_message = result.get("response", "")
        action = result.get("action", "continue")

        # 4. Add recruiter reply to history and display it
        st.session_state.messages.append({"role": "assistant", "content": bot_message})
        with st.chat_message("assistant"):
            st.write(bot_message)

        # 5. If the agent decided to end, lock the UI on next rerun
        if action == "end":
            st.session_state.conversation_ended = True

        # 6. Rerun so the debug sidebar reflects the latest action
        st.rerun()