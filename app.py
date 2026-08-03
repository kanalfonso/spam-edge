import streamlit as st
from langchain_ollama import ChatOllama
import random
import time


st.title('Spam @ The Edge')

llm = ChatOllama(
    model="qwen2.5:7b"
)

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])




# prompt = st.chat_input("Say something")

# if prompt:
#     with st.chat_message("human"):
#         st.markdown(prompt)


#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})


#     with st.chat_message("assistant"):
#         response = llm.invoke(prompt)

#         st.markdown(response.content)

#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response.content})




