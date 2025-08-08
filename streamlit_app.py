import streamlit as st
from code_explainer import CodeExplainer

st.set_page_config(page_title="Code Explainer", page_icon="🐍", layout="wide")

st.title("🐍 Code Explainer")

ex = CodeExplainer()
code = st.text_area("Paste Python code", height=300)
if st.button("Explain") and code.strip():
    with st.spinner("Generating explanation..."):
        out = ex.explain_code(code)
    st.subheader("Explanation")
    st.write(out)
