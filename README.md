🤖 Intelligent AI Assistant

An AI-powered document assistant built with Python, Streamlit, LlamaIndex, and Ollama. The application combines RAG (Retrieval-Augmented Generation), AI tool usage, and conversation memory to provide a more useful and context-aware chat experience.

Instead of simply asking an LLM questions, the assistant can search your own documents, perform calculations using a dedicated tool, and maintain context throughout a conversation.

✨ Features
📄 Document Question Answering

Upload or place your documents inside the data/ directory and ask questions about their content.

The application uses RAG to:

Load the document
Split and process the content
Generate embeddings
Store the document knowledge in a vector index
Retrieve relevant information when a question is asked
Generate an answer using the LLM

This allows the assistant to answer questions based on your own data rather than relying only on its general knowledge.

🧮 Calculator Tool

The assistant has a dedicated calculator tool for mathematical operations.

For example:

What is 125 × 48?

Instead of relying on the LLM to calculate the result, the agent can use the calculator function and return the actual result.

💬 Conversation Memory

The assistant maintains the conversation context during the current session.

For example:

User: My name is Saran.


Assistant: Nice to meet you, Saran.


User: What is my name?


Assistant: Your name is Saran.
🧠 General AI Questions

For questions unrelated to the uploaded documents or calculations, the assistant can respond using the capabilities of the configured LLM.

☁️ Ollama Cloud

The application uses Ollama Cloud for the language model, allowing the application to access a cloud-hosted model through an API.

🔤 Local Embeddings

Document embeddings are generated using the local:

nomic-embed-text

model through Ollama.

This keeps the document embedding process separate from the cloud LLM.

🏗️ Architecture
                         ┌──────────────────┐
                         │      User        │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │    Streamlit     │
                         │       UI         │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   AI Agent       │
                         │   LlamaIndex     │
                         └───────┬──────────┘
                                 │
                  ┌──────────────┼──────────────┐
                  │              │              │
                  ▼              ▼              ▼
           ┌────────────┐ ┌─────────────┐ ┌──────────────┐
           │ Calculator │ │ Document    │ │ General      │
           │   Tool     │ │ Search Tool │ │ Question     │
           └────────────┘ └──────┬──────┘ └──────────────┘
                                 │
                                 ▼
                         ┌──────────────────┐
                         │ Vector Index     │
                         │   LlamaIndex     │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ nomic-embed-text │
                         │ Local Embedding  │
                         └──────────────────┘


                                  │
                                  ▼


                         ┌──────────────────┐
                         │  Ollama Cloud    │
                         │      LLM         │
                         └──────────────────┘
🛠️ Tech Stack
Technology	Purpose
Python	Core application
Streamlit	Web interface
LlamaIndex	RAG and AI agent framework
Ollama	LLM and embedding model interface
Ollama Cloud	Cloud-based LLM
nomic-embed-text	Document embeddings
PyPDF	PDF document processing
📁 Project Structure
rag/
│
├── app_ui.py
│
├── requirements.txt
│
├── README.md
│
├── data/
│   └── your_document.pdf
│
└── .streamlit/
    └── secrets.toml
⚙️ Installation
1. Clone the repository
git clone https://github.com/yourusername/intelligent-ai-assistant.git

Navigate into the project:

cd intelligent-ai-assistant
2. Create a virtual environment

Windows:

python -m venv .venv

Activate it:

.venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
🔐 Configure API Key

Create the following file:

.streamlit/secrets.toml

Add your Ollama API key:

OLLAMA_API_KEY = "YOUR_API_KEY"

Never commit secrets.toml to GitHub.

Add this to .gitignore:

.streamlit/secrets.toml
.venv/
__pycache__/

If an API key is accidentally exposed, revoke it and generate a new one.

🔤 Install the Embedding Model

The application uses nomic-embed-text for local document embeddings.

Make sure Ollama is installed, then run:

ollama pull nomic-embed-text

Verify:

ollama list
▶️ Run the Application

Start Streamlit:

python -m streamlit run app_ui.py

Streamlit will provide a local URL, normally similar to:

http://localhost:8501

Open the URL in your browser.

📄 Adding Documents

Place your documents inside:

data/

For example:

data/
├── cybersecurity.pdf
├── networking.pdf
└── notes.txt

The application loads the documents from this directory and creates the vector index automatically.

You can then ask questions such as:

What is the main topic of this document?

or:

Explain the security mechanism mentioned in the document.
💡 Example Usage
General Question
User:
What is artificial intelligence?


Assistant:
Artificial intelligence is...
Document Question
User:
What does the document say about network security?


Assistant:
According to the document...
Calculation
User:
Calculate 125 * 48


Assistant:
The result is 6000
Conversation
User:
My name is Saran.


Assistant:
Nice to meet you, Saran.


User:
What is my name?


Assistant:
Your name is Saran.
🔄 How RAG Works

The document-question answering process follows this flow:

Document
   ↓
Document Loader
   ↓
Text Processing
   ↓
Embeddings
   ↓
Vector Index
   ↓
User Question
   ↓
Similarity Search
   ↓
Relevant Document Chunks
   ↓
LLM
   ↓
Final Answer

This approach helps the application ground its answers in the information contained within the provided documents.

🔒 Security Considerations

This project is intended primarily as a learning and portfolio project.

Some important considerations for production use include:

Never expose API keys in source code.
Never commit secrets.toml to GitHub.
Use proper authentication if the application is publicly accessible.
Validate and sanitize uploaded files.
Restrict the types and sizes of uploaded documents.
Replace unsafe evaluation methods in the calculator with a proper expression parser.
Implement logging and monitoring for a production deployment.
Consider persistent vector storage for larger document collections.
🚀 Future Improvements

Some planned improvements include:

 Drag-and-drop document uploading
 Support for multiple document formats
 Persistent vector database
 User authentication
 Multiple chat sessions
 Streaming AI responses
 Source citations for RAG answers
 Document management interface
 Conversation export
 Production cloud deployment
 Improved calculator security
 Support for multiple AI models
🎯 Project Objective

The main objective of this project is to understand how modern AI applications combine LLMs, RAG, vector search, tools, and conversation memory into a single application.

Rather than building a basic chatbot, this project demonstrates how an AI assistant can interact with external knowledge and tools while maintaining conversational context.

👨‍💻 Author

Saran Raj B R

Cybersecurity Engineering Student
Interested in Cybersecurity, AI/ML, and Software Development.

⭐ Acknowledgements

This project was built using:

LlamaIndex
Streamlit
Ollama
Python
PyPDF