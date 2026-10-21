# AtomBot

AtomBot now uses a multi-agent LangGraph workflow with tool calling for two sources of knowledge:

- local PDF retrieval from the uploaded knowledge base
- web search for current or external information

## Features

- PDF ingestion with `pypdf`
- chunking with `langchain-text-splitters`
- semantic retrieval with FAISS
- embedding generation with `langchain-huggingface`
- LLM orchestration with `langchain-groq`
- multi-agent routing via `langgraph`
- tool calling with local document search and web search
- Streamlit chat UI

## Architecture

The app now routes each question through a planner:

- question is document-focused -> use local document search
- question needs current or external context -> use web search
- both results are passed to the final answer agent

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

3. Create a `.env` file:

```bash
cp .env.example .env
```

4. Add your Groq key and model configuration:

```env
GROQ_API_KEY=your_key_here
LLM_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## Build the vector store

```bash
python create_vectorstore.py --pdf data/feature_engineering.pdf --output faiss_index
```

## Run the app

```bash
streamlit run app.py
```

This version adds real tool-calling behavior for both document retrieval and web search.
