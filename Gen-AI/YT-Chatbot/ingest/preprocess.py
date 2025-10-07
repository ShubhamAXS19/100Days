import streamlit as st

if "transcript" in st.session_state and st.session_state["transcript"]:
    all_text = [snippet.text for snippet in st.session_state["transcript"]]
else:
    all_text = []
