import streamlit as st
import random
import time
from ai.stream_response import get_response_stream
from langchain_core.messages import HumanMessage

st.title('Spam @ The Edge')

THREAD_ID = "thread-1"
URL = "http://127.0.0.1:8000/generate_response"


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_chat_disabled" not in st.session_state:
    st.session_state.is_chat_disabled = False

if "chat_placeholder_msg" not in st.session_state:
    st.session_state.chat_placeholder_msg = "What is up?"


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



def on_user_submit():
    """Logic when user inputs prompt to chat"""
    st.session_state.is_chat_disabled = True
    st.session_state.chat_placeholder_msg = "Chat momentarily disabled. Click button first!"


def on_btn_submit():
    """Logic when user chooses a feedback button"""
    st.session_state.is_chat_disabled = False
    st.session_state.chat_placeholder_msg = "What is up?"


# When user sends a message
if prompt := st.chat_input(
    placeholder=st.session_state.chat_placeholder_msg, 
    key="chat_input", 
    disabled=st.session_state.is_chat_disabled,
    on_submit=on_user_submit    # disables chat after user submits a prompt
    ):
    

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # response = st.write("Hello")


        ### API call that generates LLM response ###
        PAYLOAD = {'thread_id': THREAD_ID, 'messages': prompt}

        ai_stream_generator = get_response_stream(
            url=URL, 
            payload=PAYLOAD
        )


        response = st.write_stream(ai_stream_generator)

        ### API call that generates LLM response ###

    col_text, col1, col2 = st.columns([2, 1, 1])

    with col_text:
        st.write("**Verify response:**")

    with col1:
        yes_feedback_btn = st.button("Yes", on_click=on_btn_submit)


    with col2:
        no_feedback_btn = st.button("No", on_click=on_btn_submit)


    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


st.write(st.session_state)

