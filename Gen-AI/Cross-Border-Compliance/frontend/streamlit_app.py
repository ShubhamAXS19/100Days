# Streamlit entrypoint
import streamlit as st


with st.spinner("Uploading Files..."):
    uploaded_files = st.file_uploader(
        "Upload data", accept_multiple_files=True
    )
    
