import requests
import streamlit as st

base_url ="http://127.0.0.1:8000"

if st.button('Click for GET request'):
    get_request = requests.get(f"{base_url}/todos")
    st.write(get_request.json())


