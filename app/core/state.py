# core/state.py
import streamlit as st

UPLOADED_FILE_KEY = "uploaded_excel"


def init_session_state() -> None:
    if UPLOADED_FILE_KEY not in st.session_state:
        st.session_state[UPLOADED_FILE_KEY] = None


def set_uploaded_file(uploaded_file) -> None:
    st.session_state[UPLOADED_FILE_KEY] = uploaded_file


def get_uploaded_file():
    return st.session_state.get(UPLOADED_FILE_KEY)
