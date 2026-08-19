import os
import re
import ast
import operator
import streamlit as st

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Intelligent AI Assistant",
    page_icon="🤖",
    layout="centered",
)

# ============================================================
# TITLE
# ============================================================
st.title("🤖 Intelligent AI Assistant")

st.caption(
    "Tool use + RAG document knowledge + conversation memory"
)

# ============================================================
# CONFIGURATION
# ============================================================
OLLAMA_API_KEY = st.secrets.get(
    "OLLAMA_API_KEY",
    ""
)

if not OLLAMA_API_KEY:
    st.error(
        "Ollama API key is not configured."
    )
    st.info(
        """
Add your key to Streamlit Secrets:

OLLAMA_API_KEY = "your-new-key"
"""
    )
    st.stop()

# ============================================================
# OLLAMA CLOUD LLM
# ============================================================
llm = OpenAILike(
    model="gpt-oss:120b",
    api_base="https://ollama.com/v1",
    api_key=OLLAMA_API_KEY,
    is_chat_model=True,
    context_window=32768,
    max_tokens=2048,
)

# ============================================================
# HUGGING FACE EMBEDDING MODEL
# ============================================================
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=50)

# ============================================================
# CALCULATOR TOOL
# ============================================================
def safe_calculate(expression: str) -> str:
    """Safely evaluate mathematical expression string."""
    try:
        # Avoid eval() vulnerabilities, use simple ast evaluation
        allowed_operators = {
            ast.Add: operator.add, ast.Sub: operator.sub,
            ast.Mult: operator.mul, ast.Div: operator.truediv,
            ast.Pow: operator.pow, ast.BitXor: operator.xor,
            ast.USub: operator.neg
        }

        def eval_expr(node):
            if isinstance(node, ast.Num): 
                return node.n
            elif isinstance(node, ast.BinOp):
                return allowed_operators[type(node.op)](eval_expr(node.left), eval_expr(node.right))
            elif isinstance(node, ast.UnaryOp):
                return allowed_operators[type(node.op)](eval_expr(node.operand))
            else:
                raise TypeError(node)

        parsed_expr = ast.parse(expression, mode='eval').body
        result = eval_expr(parsed_expr)
        return f"{result}"
    except Exception as e:
        return f"Error: {str(e)}"

def detect_math(query: str):
    # Regex to capture basic mathematical expressions like "calculate 125 * 48"
    match = re.search(r'(?:calculate|what is|compute)?\s*([\d\.\s\+\-\*\/\(\)\^]+)', query.lower())
    if match:
        expr = match.group(1).strip()
        if len(expr) > 2 and re.match(r'^[\d\.\s\+\-\*\/\(\)\^]+$', expr):
            return expr
    return None

# ============================================================
# DOCUMENT RAG / SEARCH TOOL
# ============================================================
@st.cache_resource
def load_rag_index():
    if not os.path.isdir("data") or not os.listdir("data"):
        return None
    
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine(similarity_top_k=3)

query_engine = load_rag_index()

def document_search(query: str) -> str:
    if query_engine is None:
        return "No documents uploaded. Please upload a document to the data folder."
    response = query_engine.query(query)
    return str(response)

def detect_rag(query: str) -> bool:
    rag_keywords = ['document', 'doc', 'context', 'file', 'read', 'explain', 'what is the main topic']
    return any(keyword in query.lower() for keyword in rag_keywords)

# ============================================================
# CHAT MEMORY
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================
# USER INPUT & LLM ROUTING
# ============================================================
user_input = st.chat_input("Ask me anything...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            math_expr = detect_math(user_input)
            
            if math_expr:
                result = safe_calculate(math_expr)
                answer = f"The result is {result}"
                
            elif detect_rag(user_input):
                # Context augmented generation
                answer = document_search(user_input)
                
            else:
                # Normal generation with history
                from llama_index.core.llms import ChatMessage
                
                chat_history = []
                for msg in st.session_state.messages:
                    role_type = "user" if msg["role"] == "user" else "assistant"
                    chat_history.append(ChatMessage(role=role_type, content=msg["content"]))
                
                response = llm.chat(chat_history)
                answer = str(response.message.content)

            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("🤖 Intelligent AI Assistant")
    st.write("### Features")
    st.write("🧮 Calculator")
    st.write("📄 Document RAG")
    st.write("💬 Conversation Memory")
    st.write("🧠 General AI")
    st.divider()
    st.write("### Backend")
    st.write("☁️ Ollama Cloud")
    st.write("🔤 Hugging Face Embeddings")
    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()