import asyncio
import threading
import streamlit as st

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
)

from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Intelligent AI Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Intelligent AI Assistant")

st.caption(
    "Tool use + RAG document knowledge + conversation memory"
)


# ============================================================
# PERSISTENT ASYNC WORKER
# ============================================================

class AgentRunner:
    """
    Keeps ONE asyncio event loop alive.

    This is important because Ollama/LlamaIndex objects can
    retain async resources. Creating a new event loop for
    every Streamlit message can result in:

        RuntimeError: Event loop is closed
    """

    def __init__(self):

        self.loop = asyncio.new_event_loop()

        self.thread = threading.Thread(
            target=self._start_loop,
            daemon=True,
        )

        self.ready = threading.Event()

        self.thread.start()

        # Wait until the event loop is running.
        self.ready.wait()

        # Initialize everything inside the same event loop
        # thread that will execute the agent.
        self.submit(
            self._initialize()
        ).result()


    def _start_loop(self):

        asyncio.set_event_loop(self.loop)

        self.ready.set()

        self.loop.run_forever()


    def submit(self, coroutine):

        return asyncio.run_coroutine_threadsafe(
            coroutine,
            self.loop,
        )


    async def _initialize(self):

        # ----------------------------------------------------
        # MODELS
        # ----------------------------------------------------

        Settings.llm = Ollama(
            model="llama3.2",
            request_timeout=120.0,
        )

        Settings.embed_model = OllamaEmbedding(
            model_name="nomic-embed-text"
        )


        # ----------------------------------------------------
        # LOAD DOCUMENTS
        # ----------------------------------------------------

        documents = SimpleDirectoryReader(
            "data"
        ).load_data()


        # ----------------------------------------------------
        # BUILD VECTOR INDEX
        # ----------------------------------------------------

        self.index = VectorStoreIndex.from_documents(
            documents
        )


        # ----------------------------------------------------
        # CREATE QUERY ENGINE
        # ----------------------------------------------------

        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5
        )


        # ----------------------------------------------------
        # CALCULATOR TOOL
        # ----------------------------------------------------

        def calculator(expression: str) -> str:
            """
            Calculate a basic mathematical expression.

            Example:
                25 * 16
                450 / 9
                (10 + 5) / 3
            """

            try:

                result = eval(
                    expression,
                    {
                        "__builtins__": {}
                    },
                )

                return f"The result is {result}"

            except Exception as e:

                return (
                    f"Could not calculate that: {e}"
                )


        self.calculator_tool = FunctionTool.from_defaults(
            fn=calculator,
            name="calculator",
            description=(
                "Use this tool for mathematical calculations. "
                "Always use this tool instead of calculating "
                "math yourself."
            ),
        )


        # ----------------------------------------------------
        # DOCUMENT SEARCH TOOL
        # ----------------------------------------------------

        def document_search(
            question: str
        ) -> str:
            """
            Search the uploaded document.

            Use this whenever the user asks a question
            about the document.
            """

            try:

                response = self.query_engine.query(
                    question
                )

                return str(response)

            except Exception as e:

                return (
                    "Document search failed: "
                    f"{e}"
                )


        self.document_tool = FunctionTool.from_defaults(
            fn=document_search,
            name="document_search",
            description=(
                "Search the uploaded document to answer "
                "questions about its contents. "
                "Always use this tool when the user asks "
                "about information from the document."
            ),
        )


        # ----------------------------------------------------
        # CREATE AGENT
        # ----------------------------------------------------

        self.agent = AgentWorkflow.from_tools_or_functions(
            [
                self.calculator_tool,
                self.document_tool,
            ],

            llm=Settings.llm,

            system_prompt=(
                "You are an intelligent AI assistant. "

                "RULE 1: For mathematics, ALWAYS use "
                "the calculator tool. "

                "RULE 2: For questions about the uploaded "
                "document, ALWAYS use document_search. "

                "RULE 3: Never invent information from "
                "the document. "

                "RULE 4: For general questions, answer "
                "using your general knowledge. "

                "RULE 5: Give clear and concise answers."
            ),
        )


        # ----------------------------------------------------
        # CONVERSATION HISTORY
        # ----------------------------------------------------

        self.chat_history = []


    async def _ask(self, question):

        response = await self.agent.run(
            user_msg=question,
            chat_history=self.chat_history,
        )

        answer = str(response)


        # ----------------------------------------------------
        # SAVE MEMORY
        # ----------------------------------------------------

        self.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        self.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return answer


    def ask(self, question):

        future = self.submit(
            self._ask(question)
        )

        return future.result()


    def clear_memory(self):

        future = self.submit(
            self._clear_memory()
        )

        future.result()


    async def _clear_memory(self):

        self.chat_history = []


# ============================================================
# CREATE PERSISTENT RUNNER
# ============================================================

@st.cache_resource
def get_agent_runner():

    return AgentRunner()


# ============================================================
# INITIALIZE AI
# ============================================================

try:

    with st.spinner(
        "Loading AI assistant and document..."
    ):

        runner = get_agent_runner()

except Exception as e:

    st.error(
        "Failed to initialize the AI assistant."
    )

    st.exception(e)

    st.stop()


# ============================================================
# STREAMLIT CHAT MEMORY
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
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Ask me anything..."
)


if user_input:

    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(user_input)


    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )


    # --------------------------------------------------------
    # ASK AGENT
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                answer = runner.ask(
                    user_input
                )

                st.markdown(answer)

            except Exception as e:

                answer = (
                    f"Error while processing your request: "
                    f"{e}"
                )

                st.error(answer)


    # --------------------------------------------------------
    # SAVE ASSISTANT MESSAGE
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

    st.header("🤖 AI Assistant")

    st.write(
        "### Capabilities"
    )

    st.write("🧮 Calculator")
    st.write("📄 Document RAG")
    st.write("💬 Conversation Memory")
    st.write("🧠 General Questions")

    st.divider()

    st.write(
        "### Local Models"
    )

    st.write("🦙 llama3.2")
    st.write("🔤 nomic-embed-text")

    st.divider()

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []

        runner.clear_memory()

        st.rerun()