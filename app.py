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


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



# When user sends a message
if prompt := st.chat_input("What is up?"):
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})


    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        
        PAYLOAD = {'thread_id': THREAD_ID, 'messages': prompt}

        ai_stream_generator = get_response_stream(
            url=URL, 
            payload=PAYLOAD
        )


        response = st.write_stream(ai_stream_generator)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})


st.write(st.session_state)

