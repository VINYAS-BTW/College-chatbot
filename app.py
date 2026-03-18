import streamlit as st
from backend import ask_question

st.set_page_config(page_title="College Chatbot", page_icon="🎓")
st.title("🎓 RVITM College Chatbot")
st.caption("Ask me anything about CSE, AI/ML, or Admissions!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input at the bottom
user_input = st.chat_input("Ask something...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get and show bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_question(user_input)
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})