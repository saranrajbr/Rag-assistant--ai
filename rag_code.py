from pathlib import Path

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


# Tell LlamaIndex which local models to use
Settings.llm = Ollama(
    model="llama3.2",
    request_timeout=120.0
)

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text"
)


# Find the data folder relative to this Python file
DATA_DIR = Path(__file__).parent / "data"

print(f"Loading documents from: {DATA_DIR}")

documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()

print(f"Loaded {len(documents)} document(s)")


# Build the searchable index
index = VectorStoreIndex.from_documents(documents)


# Create the question-answering engine
query_engine = index.as_query_engine()


# Ask questions in a loop
print("\nYour document Q&A app is ready! Type 'exit' to quit.\n")

while True: 
    question = input("Ask a question: ")

    if question.strip().lower() == "exit":
        break

    response = query_engine.query(question)

    print("\nAnswer:", response)
    print("-" * 50)