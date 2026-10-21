import streamlit as st

from rag import create_rag_chain

st.set_page_config(page_title="AtomBot", page_icon="🤖", layout="centered")

st.title("AtomBot")
st.caption("Agentic RAG Knowledge Assistant")


@st.cache_resource
def load_rag():
    return create_rag_chain()


ask = load_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask a question about the document...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    history = ""
    for message in st.session_state.messages[:-1]:
        history += f'{message["role"].capitalize()}: {message["content"]}\n'

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, documents = ask(question, history)
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.stop()

        st.write(answer)

        with st.expander("View Sources"):
            for i, document in enumerate(documents, start=1):
                page = document.metadata.get("page", "Unknown")
                if isinstance(page, int):
                    page += 1
                st.markdown(f"**Source {i} | Page {page}**")
                st.write(document.page_content)

    st.session_state.messages.append({"role": "assistant", "content": answer})
