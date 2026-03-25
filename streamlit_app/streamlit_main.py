import streamlit as st

st.title("SMS Recruitment Chatbot")

# Store chat history:
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages:
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

# User input:
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to chat history:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message:
    with st.chat_message("user"):
        st.write(user_input)
    
    # Here you would typically call your chatbot's response function
    # TODO: For demonstration, we'll just echo the user's message:" 
    # TODO: Add bot response to chat history:   
    # TODO: Display bot response