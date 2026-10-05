"""DriveWise Streamlit demo.  Run from the project root:  streamlit run app/app.py"""

import sys
from pathlib import Path

# Make the project root importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.pipeline import DriveWise

st.set_page_config(page_title="DriveWise", page_icon="🚗", layout="wide")


@st.cache_resource
def get_assistant() -> DriveWise:
    """Load models, vector store and Gemini client once per server session."""
    return DriveWise()


st.title("🚗 DriveWise")
st.caption("Brochure-Grounded Conversational AI for Cars")

try:
    assistant = get_assistant()
except Exception as exc:
    st.error(f"Could not start DriveWise: {exc}")
    st.stop()

brand = st.selectbox("Select Brand", assistant.store.brands())
model = st.selectbox("Select Model", assistant.store.models(brand))
question = st.text_input(
    "Ask a question about this vehicle",
    placeholder="e.g. How many airbags does this car have?",
)

if st.button("Ask DriveWise"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Searching brochure..."):
        response = assistant.ask(question.strip(), brand, model)

    st.subheader("Answer")
    st.write(response.answer)

    st.subheader("Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Retrieval", f"{response.retrieval_time * 1000:.1f} ms")
    col2.metric("Reranking", f"{response.rerank_time * 1000:.1f} ms")
    col3.metric("Generation", f"{response.generation_time * 1000:.1f} ms")

    if response.sources:
        st.subheader("Sources")
        for src in response.sources:
            st.markdown(
                f"- **{src['document']}** | Section: **{src['section']}** | Page **{src['page']}**"
            )
