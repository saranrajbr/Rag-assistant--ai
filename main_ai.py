import os
os.environ['NLTK_DATA'] = '/tmp/nltk_data'
os.environ['LLAMA_INDEX_CACHE_DIR'] = '/tmp/llama_index_cache'
import asyncio
import streamlit as st

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)

from llama_index.core.tools import (
    FunctionTool,
    QueryEngineTool,
)

from llama_index.core.agent.workflow import FunctionAgent

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.core.workflow import Context


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Intelligent AI Assistant",
    page_icon="🤖",
)

st.title("🤖 Intelligent AI Assistant")

st.caption(
    "Tool use + RAG document knowledge + conversation memory"
)


# ============================================================
# OLLAMA API KEY
# ============================================================

try:

    OLLAMA_API_KEY = st.secrets["OLLAMA_API_KEY"]

except Exception:

    st.error("OLLAMA_API_KEY is missing.")

    st.info(
        """
Create:

.streamlit/secrets.toml

and add:

OLLAMA_API_KEY = "YOUR_NEW_API_KEY"
"""
    )

    st.stop()


# ============================================================
# OLLAMA CLOUD LLM
# ============================================================

Settings.llm = Ollama(
    model="gpt-oss:120b-cloud",
    request_timeout=120.0,
    base_url="https://ollama.com",
    headers={
        "Authorization": f"Bearer {OLLAMA_API_KEY}"
    },
)


# ============================================================
# LOCAL EMBEDDING MODEL
# ============================================================

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text"
)


# ============================================================
# CALCULATOR TOOL
# ============================================================

def calculator(expression: str) -> str:

    """
    Calculate a mathematical expression.
    """

    try:

        result = eval(
            expression,
            {
                "__builtins__": {}
            }
        )

        return f"The result is {result}"

    except Exception as e:

        return f"Could not calculate that: {e}"


calculator_tool = FunctionTool.from_defaults(
    fn=calculator,
    name="calculator",
    description=(
        "Use this tool for mathematical calculations. "
        "Examples: 25 * 4, 100 / 5, "
        "(10 + 5) * 2."
    ),
)


# ============================================================
# BUILD DOCUMENT RAG
# ============================================================

@st.cache_resource
def create_query_engine():

    documents = SimpleDirectoryReader(
        "data"
    ).load_data()

    if not documents:

        raise ValueError(
            "No documents found in the data folder."
        )

    index = VectorStoreIndex.from_documents(
        documents
    )

    query_engine = index.as_query_engine(
        similarity_top_k=5
    )

    return query_engine


# ============================================================
# LOAD DOCUMENT
# ============================================================

try:

    with st.spinner(
        "Loading document knowledge..."
    ):

        query_engine = create_query_engine()

except Exception as e:

    st.error(
        "Could not load the documents."
    )

    st.exception(e)

    st.stop()


# ============================================================
# DOCUMENT TOOL
# ============================================================

document_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="document_search",
    description=(
        "Use this tool to answer questions about "
        "the uploaded document. "
        "Always use this tool when the user asks "
        "about information from the document."
    ),
)


# ============================================================
# CREATE AGENT
# ============================================================

@st.cache_resource
def create_agent():

    return FunctionAgent(
        tools=[
            calculator_tool,
            document_tool,
        ],

        llm=Settings.llm,

        system_prompt=(
            "You are an intelligent AI assistant.\n\n"

            "RULE 1:\n"
            "For mathematical questions, always use "
            "the calculator tool.\n\n"

            "RULE 2:\n"
            "For questions about the uploaded document, "
            "always use the document_search tool.\n\n"

            "RULE 3:\n"
            "Never invent information from the document.\n\n"

            "RULE 4:\n"
            "If the requested information is not in the "
            "document, clearly say that it is not "
            "available in the document.\n\n"

            "RULE 5:\n"
            "For general questions, answer normally.\n\n"

            "RULE 6:\n"
            "Give clear and concise answers."
        ),
    )


agent = create_agent()


# ============================================================
# CREATE ONE PERSISTENT EVENT LOOP
# ============================================================

if "event_loop" not in st.session_state:

    st.session_state.event_loop = asyncio.new_event_loop()


loop = st.session_state.event_loop


# ============================================================
# CREATE AGENT CONTEXT
# ============================================================

if "agent_context" not in st.session_state:

    st.session_state.agent_context = Context(
        agent
    )


# ============================================================
# CHAT MEMORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# ASYNC AGENT FUNCTION
# ============================================================

async def ask_agent(
    question,
    context,
):

    response = await agent.run(
        question,
        ctx=context,
    )

    return str(response)


# ============================================================
# USER INPUT
# ============================================================

user_input = st.chat_input(
    "Ask me anything..."
)


if user_input:

    # --------------------------------------------------------
    # SHOW USER
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            user_input
        )


    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )


    # --------------------------------------------------------
    # ASK AI
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            try:

                # IMPORTANT:
                # Do NOT use asyncio.run()
                #
                # We reuse the same event loop.

                answer = loop.run_until_complete(
                    ask_agent(
                        user_input,
                        st.session_state.agent_context,
                    )
                )

                st.markdown(
                    answer
                )

            except Exception as e:

                answer = (
                    "Error while processing "
                    f"your request: {e}"
                )

                st.error(
                    answer
                )


    # --------------------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🤖 Intelligent AI Assistant"
    )

    st.write(
        "### Features"
    )

    st.write(
        "🧮 Calculator"
    )

    st.write(
        "📄 Document RAG"
    )

    st.write(
        "💬 Conversation Memory"
    )

    st.write(
        "🧠 General AI"
    )

    st.divider()

    st.write(
        "### Backend"
    )

    st.write(
        "☁️ Ollama Cloud"
    )

    st.write(
        "🔤 Local nomic-embed-text"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        # Create a fresh context
        st.session_state.agent_context = Context(
            agent
        )

        st.rerun()