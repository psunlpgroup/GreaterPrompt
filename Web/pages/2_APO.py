import streamlit as st


st.markdown("<h1 style='text-align: center; white-space: nowrap;'>🤩 Optimize with APO</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a jsonl input file", type=("jsonl"))
