Agentic RAG Knowledge Assistant

An intelligent, retrieval-augmented generation (RAG) knowledge assistant designed to process, query, and retrieve precise context from custom documentation. Powered by **LangChain**, **FAISS**, and planned for advanced state management via **LangGraph**.

## Features
* **Semantic Document Retrieval:** Uses FAISS for high-speed vector similarity search over embedded knowledge bases.
* **Context-Aware Responses:** Integrates retrieved chunks dynamically into LLM prompts to minimize hallucinations.
* **Modular Pipeline:** Easily extendable for advanced multi-step reasoning workflows.

## Prerequisites & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Asad-Hussain/agentic-rag-knowledge-assistant.git](https://github.com/Asad-Hussain/agentic-rag-knowledge-assistant.git)
   cd agentic-rag-knowledge-assistant
Install dependencies:

Bash
pip install langchain faiss-cpu sentence-transformers python-dotenv
Configure your environment variables:

Code snippet
OPENAI_API_KEY=your_openai_api_key_here
Usage
Execute the Jupyter notebook or run the core Python script to build the vector store and start querying your data:

Bash
python app.py
