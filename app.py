from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

load_dotenv()

DATA_DIR = Path("data")
VECTORSTORE_PATH = Path("faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")


class AgentState(TypedDict):
    question: str
    history: str
    local_context: str
    web_context: str
    answer: str


def ensure_pdf_exists(pdf_path: str | Path) -> Path:
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_file}. Add your PDF under the data/ folder first."
        )
    return pdf_file


def build_vectorstore(pdf_path: str | Path, persist_path: str | Path = VECTORSTORE_PATH) -> FAISS:
    pdf_file = ensure_pdf_exists(pdf_path)
    loader = PyPDFLoader(str(pdf_file))
    documents = loader.load()

    if not documents:
        raise ValueError(f"No pages were extracted from {pdf_file}.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(persist_path))
    return vectorstore


def load_vectorstore() -> FAISS:
    if not VECTORSTORE_PATH.exists():
        raise FileNotFoundError(
            "No FAISS vector store found. Run create_vectorstore.py before using the app."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


@tool
def local_document_search(query: str) -> str:
    """Search the uploaded PDF knowledge base for local grounding."""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information found in the local document."

    return "\n\n".join(doc.page_content for doc in documents)


@tool
def web_search(query: str) -> str:
    """Search the open web for current or external information."""
    search_tool = DuckDuckGoSearchResults()
    results = search_tool.invoke(query)

    if not results:
        return "No web results were found."

    formatted = []
    for item in results:
        if isinstance(item, dict):
            title = item.get("title", "Untitled")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            formatted.append(f"{title}: {snippet} ({link})")
        else:
            formatted.append(str(item))

    return "\n\n".join(formatted)


def decide_tool(state: AgentState) -> str:
    question = state["question"].lower()
    external_keywords = [
        "latest",
        "recent",
        "news",
        "today",
        "current",
        "now",
        "live",
        "upcoming",
        "what happened",
        "world",
        "market",
        "breaking",
        "2025",
        "2026",
    ]

    if any(keyword in question for keyword in external_keywords):
        return "web_search"
    return "document_search"


def search_local(state: AgentState) -> AgentState:
    local_result = local_document_search.invoke({"query": state["question"]})
    return {
        **state,
        "local_context": local_result,
        "web_context": "",
    }


def search_web(state: AgentState) -> AgentState:
    web_result = web_search.invoke({"query": state["question"]})
    return {
        **state,
        "local_context": state.get("local_context", ""),
        "web_context": web_result,
    }


def answer_question(state: AgentState) -> AgentState:
    llm = ChatGroq(model=LLM_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_template(
        """
        You are AtomBot, a research assistant.

        Use the local document context and web search context when helpful.
        If the answer is not present in either source, say:
        "I could not find this information in the document or on the web."

        Do not invent information.

        Conversation history:
        {history}

        Local document context:
        {local_context}

        Web search context:
        {web_context}

        Current question:
        {question}

        Answer:
        """
    )

    combined_context = "\n\n".join(
        part for part in [state.get("local_context", ""), state.get("web_context", "")] if part.strip()
    )

    response = llm.invoke(
        prompt.format(
            history=state.get("history", ""),
            local_context=state.get("local_context", ""),
            web_context=state.get("web_context", ""),
            question=state["question"],
        )
    )

    return {
        **state,
        "answer": response.content,
        "local_context": state.get("local_context", ""),
        "web_context": state.get("web_context", ""),
    }


def create_multi_agent_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", lambda state: state)
    workflow.add_node("document_search", search_local)
    workflow.add_node("web_search", search_web)
    workflow.add_node("answer", answer_question)

    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        decide_tool,
        {
            "document_search": "document_search",
            "web_search": "web_search",
        },
    )
    workflow.add_edge("document_search", "answer")
    workflow.add_edge("web_search", "answer")
    workflow.add_edge("answer", END)

    return workflow.compile()


if __name__ == "__main__":
    pdf_file = DATA_DIR / "feature_engineering.pdf"
    if pdf_file.exists():
        build_vectorstore(pdf_file)
        print(f"Vector store built from {pdf_file}.")
    else:
        print("No PDF found in data/. Add a PDF and run create_vectorstore.py.")
