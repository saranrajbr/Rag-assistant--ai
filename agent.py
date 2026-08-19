"""
Day 3, Module 2 Demo: Building an Intelligent AI Assistant
(Tool use + Document knowledge + Conversation memory)

Before running:
    pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama pypdf

This assistant can:
  1. Answer general questions using its own reasoning
  2. Do math using a real calculator tool (not "guessed" math)
  3. Answer questions about a document (reusing Day 2's RAG knowledge)
  4. Remember earlier turns in the same conversation

Folder structure expected (same as Day 2):
    agent_demo/
        agent_app.py
        data/
            your_document.txt (or .pdf)
"""

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import asyncio

# ---------------------------------------------------------------------------
# STEP 1: Configure local models (same as Day 2)
# ---------------------------------------------------------------------------
Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# ---------------------------------------------------------------------------
# STEP 2: Define a TOOL - a plain Python function the agent can choose to call
# ---------------------------------------------------------------------------
# This is the key new concept for today: the agent doesn't do math itself
# (LLMs are often wrong at arithmetic) - it calls this real function instead.
def calculator(expression: str) -> str:
    """Evaluates a basic math expression, e.g. '25 * 4' or '(10 + 5) / 3'."""
    try:
        # NOTE for instructor: eval() is used here ONLY for a controlled
        # classroom demo with trusted input. Mention to students this is
        # NOT safe for real-world/production apps handling untrusted input.
        result = eval(expression, {"__builtins__": {}})
        return f"The result is {result}"
    except Exception as e:
        return f"Could not calculate that: {e}"

calculator_tool = FunctionTool.from_defaults(
    fn=calculator,
    name="calculator",
    description="Use this to evaluate math expressions like '12 * 8' or '450 / 9'."
)

# ---------------------------------------------------------------------------
# STEP 3: Turn Day 2's RAG pipeline into a TOOL too
# ---------------------------------------------------------------------------
# This is the nice "aha" moment: everything students built on Day 2 becomes
# just ONE tool inside today's agent.
print("Loading document and building knowledge index...")
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=5)

document_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="document_search",
    description="Use this to answer questions about the content of the uploaded document."
)

# ---------------------------------------------------------------------------
# STEP 4: Build the AGENT - give it the tools and let it decide when to use them
# ---------------------------------------------------------------------------
agent = FunctionAgent(
    tools=[calculator_tool, document_tool],
    llm=Settings.llm,
    system_prompt=(
        "You are a helpful assistant. Use the calculator tool for any math. "
        "Use the document_search tool for questions about the uploaded document. "
        "For general questions, answer directly using your own knowledge."
    ),
)

# ---------------------------------------------------------------------------
# STEP 5: Chat loop WITH conversation memory
# ---------------------------------------------------------------------------
# ctx keeps track of the conversation so the agent remembers earlier turns.
from llama_index.core.workflow import Context

async def main():
    ctx = Context(agent)
    print("\nIntelligent Assistant ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break

        response = await agent.run(user_input, ctx=ctx)
        print(f"\nAssistant: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())
