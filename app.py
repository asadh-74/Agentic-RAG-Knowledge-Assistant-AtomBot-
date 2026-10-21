from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired

load_dotenv()

DATA_DIR = Path("data")
VECTORSTORE_PATH = Path("faiss_index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")


class AgentState(TypedDict):
    question: str
    history: str
    context: str
    documents: list
    answer: NotRequired[str]


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


def create_langgraph_agent():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGroq(model=LLM_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """
        You are AtomBot, an expert research assistant.

        Use the retrieved document context and conversation history to answer the user's question.
        If the answer isn't in the document, say:
        "I could not find this information in the document."

        Do not invent information.

        Conversation history:
        {history}

        Document context:
        {context}

        Current question:
        {question}

        Answer:
        """
    )

    def retrieve(state: AgentState):
        documents = retriever.invoke(state["question"])
        context = "\n\n".join(doc.page_content for doc in documents)
        return {
            "documents": documents,
            "context": context,
        }

    def generate(state: AgentState):
        response = llm.invoke(
            prompt.format(
                history=state.get("history", ""),
                context=state["context"],
                question=state["question"],
            )
        )
        return {"answer": response.content}

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


if __name__ == "__main__":
    pdf_file = DATA_DIR / "feature_engineering.pdf"
    if pdf_file.exists():
        build_vectorstore(pdf_file)
        print(f"Vector store built from {pdf_file}.")
    else:
        print("No PDF found in data/. Add a PDF and run create_vectorstore.py.")
