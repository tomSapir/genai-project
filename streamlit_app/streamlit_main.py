"""
streamlit_main.py — Main Streamlit UI for the SMS Recruitment Chatbot.

Flow:
    1. Candidate types a message in the chat input.
    2. The message is passed to get_bot_response() (app/main.py), which
       runs the multi-agent pipeline (Main Agent + advisors).
    3. The agent returns an action ("continue" / "schedule" / "end")
       and a response message.
    4. The response is appended to history and a rerun re-renders the
       chat from session state.
    5. When action is "end", the input is disabled and the user can
       reset to start a new conversation.

Session state keys:
    messages          : list of {"role": "user"|"assistant", "content": str}
    conversation_ended: bool — True once the agent returns action="end"

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

st.set_page_config(page_title="SMS Recruitment Chatbot", page_icon="💬", layout="centered")
st.title("💬 SMS Recruitment Chatbot")
st.caption("Python Developer Position")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_ended" not in st.session_state:
    st.session_state.conversation_ended = False

# Chat history — re-rendered on every rerun from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.conversation_ended:
    # Lock the UI once the agent has ended the conversation
    st.info("This conversation has ended.")
    if st.button("Start New Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_ended = False
        st.rerun()
else:
    user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        try:
            with st.spinner("..."):
                result = get_bot_response(st.session_state.messages)
        except Exception:
            # Roll back the user message so they can retry cleanly.
            # We don't surface the raw exception — a JSON parser failure
            # would otherwise dump a traceback into the chat UI.
            st.session_state.messages.pop()
            st.error("Sorry, something went wrong processing your message. Please try again.")
            st.stop()

        bot_message = result.get("response", "")
        action = result.get("action", "continue")
        st.session_state.messages.append({"role": "assistant", "content": bot_message})

        if action == "end":
            st.session_state.conversation_ended = True

        st.rerun()
