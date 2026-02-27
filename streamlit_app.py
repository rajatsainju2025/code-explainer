import streamlit as st

st.set_page_config(page_title="Code Explainer", page_icon="🐍", layout="wide")

st.title("🐍 Code Explainer")


@st.cache_resource
def get_explainer():
    """Cache the CodeExplainer instance to avoid repeated initialization."""
    from code_explainer import CodeExplainer
    return CodeExplainer()


# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Settings")
    strategy = st.selectbox(
        "Explanation Strategy",
        ["vanilla", "ast_augmented", "retrieval_augmented", "execution_trace", "enhanced_rag"],
        index=0,
        help="Select the prompting strategy for generating explanations.",
    )
    max_length = st.slider("Max Output Length", 128, 2048, 512, step=64)

    st.divider()
    st.caption("Model loads on first request. This may take a moment.")

# Main area
try:
    ex = get_explainer()
    model_ready = True
except Exception as e:
    st.error(f"Failed to initialize model: {e}")
    model_ready = False

code = st.text_area("Paste Python code", height=300, placeholder="def fibonacci(n):\n    ...")

if st.button("Explain", disabled=not model_ready, type="primary") and code.strip():
    with st.spinner("Generating explanation..."):
        try:
            out = ex.explain_code(code, max_length=max_length, strategy=strategy)
            st.subheader("📝 Explanation")
            st.markdown(out)
        except Exception as e:
            st.error(f"Explanation failed: {e}")
elif not code.strip() and st.session_state.get("_button_clicked"):
    st.warning("Please enter some code to explain.")
