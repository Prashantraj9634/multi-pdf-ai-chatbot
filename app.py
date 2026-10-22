"""
app.py - Streamlit user interface for the Multi-PDF AI Chatbot.

Run with:  streamlit run app.py
All RAG logic lives in rag.py; this file only handles the screen.
"""

import streamlit as st

import rag

st.set_page_config(page_title="Multi-PDF AI Chatbot", page_icon="📚", layout="wide")



# Session state = values that survive Streamlit's re-runs

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []      # chat history: role, content, sources
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processed_summary" not in st.session_state:
        st.session_state.processed_summary = []  # per-file results from indexing
    if "status" not in st.session_state:
        st.session_state.status = "No documents processed yet."


def save_uploaded_files(uploaded_files):
    """Write the uploaded files into data/PDFs/ and return their paths."""
    rag.PDF_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for uploaded in uploaded_files:
        destination = rag.PDF_DIR / uploaded.name
        destination.write_bytes(uploaded.getbuffer())
        paths.append(destination)
    return paths


def format_source_lines(sources):
    """[("AI.pdf", 4)] -> ["AI.pdf — Page 4"]"""
    return [f"{filename} — Page {page}" for filename, page in sources]



# Sidebar: upload + process

def render_sidebar():
    with st.sidebar:
        st.header("📄 Your documents")
        uploaded_files = st.file_uploader(
            "Upload one or more PDFs", type="pdf", accept_multiple_files=True
        )
        st.write(f"**Uploaded documents:** {len(uploaded_files)}")

        if st.button("Process PDFs", type="primary", use_container_width=True):
            if not uploaded_files:
                st.session_state.status = "⚠️ Please upload at least one PDF first."
            else:
                process_uploaded_files(uploaded_files)

        st.divider()
        st.subheader("Status")
        st.info(st.session_state.status)

        for item in st.session_state.processed_summary:
            if "error" in item:
                st.warning(f"{item['file']}: {item['error']}")
            else:
                st.caption(f"✅ {item['file']} — {item['pages']} pages, {item['chunks']} chunks")

        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def process_uploaded_files(uploaded_files):
    try:
        with st.spinner("Reading PDFs, creating embeddings and building the index..."):
            paths = save_uploaded_files(uploaded_files)
            store, summary = rag.process_pdfs(paths)
        st.session_state.vector_store = store
        st.session_state.processed_summary = summary
        st.session_state.messages = []  # new documents -> fresh conversation
        good = [s for s in summary if "error" not in s]
        st.session_state.status = f"✅ Ready! Indexed {len(good)} document(s). Ask a question."
    except rag.RAGError as error:
        st.session_state.vector_store = None
        st.session_state.processed_summary = []
        st.session_state.status = f"❌ {error}"
    except Exception as error:  # anything unexpected
        st.session_state.vector_store = None
        st.session_state.status = f"❌ Unexpected error while processing: {error}"



# Main area: chat

def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.markdown("**Sources:**")
            for line in format_source_lines(message["sources"]):
                st.markdown(f"- {line}")


def handle_question(question):
    """Run the RAG pipeline for one question and add the result to the chat."""
    # Show the user's message and remember it
    st.session_state.messages.append({"role": "user", "content": question})
    render_message(st.session_state.messages[-1])

    # Question asked before documents were processed
    if st.session_state.vector_store is None:
        reply = "⚠️ Please upload your PDFs and click **Process PDFs** before asking questions."
        st.session_state.messages.append({"role": "assistant", "content": reply})
        render_message(st.session_state.messages[-1])
        return

    # History sent to the model = everything BEFORE this question
    history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                result = rag.answer_question(st.session_state.vector_store, question, history)
                reply = {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
                debug = result
            except rag.RAGError as error:
                reply = {"role": "assistant", "content": f"⚠️ {error}"}
                debug = None
            except Exception as error:
                reply = {"role": "assistant", "content": f"⚠️ Something went wrong: {error}"}
                debug = None

        st.markdown(reply["content"])
        if reply.get("sources"):
            st.markdown("**Sources:**")
            for line in format_source_lines(reply["sources"]):
                st.markdown(f"- {line}")

        # Transparency panel: great for demos and interviews
        if debug:
            with st.expander("How this answer was found"):
                st.write(f"**Search query used:** {debug['search_query']}")
                for doc, distance in debug["retrieved"]:
                    st.caption(
                        f"{doc.metadata['source']} — Page {doc.metadata['page']} "
                        f"(distance {distance:.3f}, lower = more similar)"
                    )
                    st.text(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))

    st.session_state.messages.append(reply)


def main():
    init_session_state()
    render_sidebar()

    st.title("📚 Multi-PDF AI Chatbot")
    st.caption("Upload PDFs, click **Process PDFs**, then ask questions about them.")

    # Replay the conversation so far
    for message in st.session_state.messages:
        render_message(message)

    question = st.chat_input("Ask a question about your documents...")
    if question:
        handle_question(question)


main()
