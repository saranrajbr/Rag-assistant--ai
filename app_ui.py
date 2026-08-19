import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Document Q&A App",
    page_icon="📄"
)

st.title("📄 Ask Your Document")

st.write(
    "This app answers questions using the document in the data/ folder."
)

# ---------------------------------------------------------------------------
# Sidebar with instructions
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📖 How to use")

    st.write("""
    1. Make sure your document is inside the `data/` folder.
    2. Type a question about the document.
    3. Click **Get Answer**.
    4. Check the **Sources** section to see where the answer came from.
    """)

    st.divider()

    st.write("### 🤖 Local AI")
    st.write("LLM: llama3.2")
    st.write("Embedding: nomic-embed-text")

# ---------------------------------------------------------------------------
# Configure local models
# ---------------------------------------------------------------------------
Settings.llm = Ollama(
    model="llama3.2",
    request_timeout=120.0
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text"
)

# ---------------------------------------------------------------------------
# Build the index once and cache it
# ---------------------------------------------------------------------------
@st.cache_resource
def load_index():
    documents = SimpleDirectoryReader("data").load_data()

    index = VectorStoreIndex.from_documents(documents)

    return index

# ---------------------------------------------------------------------------
# Load index
# ---------------------------------------------------------------------------
try:
    with st.spinner("Loading document and building index..."):
        index = load_index()

    query_engine = index.as_query_engine(
        similarity_top_k=5
    )

except Exception as e:
    st.error("❌ Could not load the document or build the index.")
    st.error(f"Error: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Question input
# ---------------------------------------------------------------------------
question = st.text_input(
    "Ask a question about your document:"
)

# ---------------------------------------------------------------------------
# Get Answer button
# ---------------------------------------------------------------------------
if st.button("Get Answer"):

    # Check for empty question
    if not question.strip():

        st.warning("⚠️ Please enter a question first.")

    else:

        try:
            with st.spinner("Thinking..."):

                response = query_engine.query(question)

            # ---------------------------------------------------------------
            # Display answer
            # ---------------------------------------------------------------
            st.markdown("### 💡 Answer")

            st.write(str(response))

            # ---------------------------------------------------------------
            # Display sources
            # ---------------------------------------------------------------
            st.markdown("### 📚 Sources")

            if response.source_nodes:

                for i, node in enumerate(response.source_nodes, 1):

                    score = node.score

                    st.write(
                        f"**Source {i}** — Similarity score: "
                        f"{score:.4f}"
                    )

                    st.write(
                        node.node.get_content()
                    )

                    st.divider()

            else:

                st.info("No source information was returned.")

        except Exception as e:

            st.error(
                "❌ Something went wrong while generating the answer."
            )

            st.warning(
                "Make sure Ollama is running and the required models "
                "are available."
            )

            st.code(str(e))

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()

st.caption(
    "Running locally with Ollama — no API keys and no cloud API required."
)