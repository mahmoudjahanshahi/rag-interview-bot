# pip install streamlit

import streamlit as st
from chatbot import chatbot

# Set up Streamlit page configuration
st.set_page_config(page_title="Interview Assistant", layout="wide")
st.title("Interview Assistant Chatbot")
st.caption("Ask questions for your job interview preparation.")

# Sidebar settings
st.sidebar.header("Settings")
neighbors = st.sidebar.slider("Number of context chunks", 1, 5, 3)

# Display information about the app
st.sidebar.markdown("[View source on GitHub](https://github.com/mahmoudjahanshahi/rag-interview-bot)", unsafe_allow_html=True)

# Chat interface
user_input = st.text_input("Your question:", placeholder="Am I a good fit for this job?")

# Process user input and generate response
if user_input:
    with st.spinner("Thinking..."):
        answer = chatbot(user_input, neighbors=neighbors)
    st.markdown("### Answer")
    st.markdown(answer)
