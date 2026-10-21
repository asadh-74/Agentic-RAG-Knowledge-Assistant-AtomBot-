# AtomBot

AtomBot is now upgraded to a LangGraph-based agent workflow with retrieval-augmented generation (RAG).

## What changed

- PDF ingestion with `pypdf`
- Text chunking with `langchain-text-splitters`
- Embeddings with `langchain-huggingface`
- FAISS-based document retrieval
- LLM orchestration via `langchain-groq`
- Agent orchestration with `langgraph`
- Streamlit chat interface

## Architecture

The app now uses a simple LangGraph workflow:

1. Receive the user question
2. Retrieve relevant chunks from the vector store
3. Build a grounded context prompt
4. Send the prompt + conversation history to Groq
5. Return the answer and supporting sources

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Groq key:

```bash
cp .env.example .env
```

4. Add your settings, for example:

```env
GROQ_API_KEY=your_key_here
LLM_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Create the vector store

```bash
python create_vectorstore.py --pdf data/feature_engineering.pdf --output faiss_index
```

## Run the app

```bash
streamlit run app.py
```

## Project structure

```text
.
├── app.py
├── create_vectorstore.py
├── rag.py
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
├── faiss_index/
└── README.md
```

This is now a proper LangGraph agent flow rather than a simple linear RAG script.
