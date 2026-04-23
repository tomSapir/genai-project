import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from app.main import get_bot_response

st.set_page_config(page_title="SMS Recruitment Chatbot", page_icon="💬", layout="centered")
st.title("💬 SMS Recruitment Chatbot")
st.caption("Python Developer Position")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_ended" not in st.session_state:
    st.session_state.conversation_ended = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# --- DEBUG SIDEBAR (remove before production) ---
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

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input / response
if st.session_state.conversation_ended:
    st.info("This conversation has ended.")
    if st.button("Start New Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_ended = False
        st.rerun()
else:
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Add and display candidate message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Call the main agent
        with st.spinner("..."):
            result = get_bot_response(st.session_state.messages)

        st.session_state.last_result = result
        bot_message = result.get("response", "")
        action = result.get("action", "continue")

        # Add and display recruiter reply
        st.session_state.messages.append({"role": "assistant", "content": bot_message})
        with st.chat_message("assistant"):
            st.write(bot_message)

        if action == "end":
            st.session_state.conversation_ended = True

        st.rerun()