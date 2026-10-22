"""
rag.py - the RAG (Retrieval-Augmented Generation) engine.

Pipeline:
    PDF -> load -> split into chunks -> embed (local) -> store in ChromaDB
    Question -> (rewrite using chat history) -> embed -> similarity search
             -> retrieved context -> Hugging Face LLM -> answer + sources
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load variables from the .env file (HF_TOKEN, optional HF_MODEL)
load_dotenv()


# Settings (all in one place so they are easy to explain and change)

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "data" / "PDFs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "pdf_chunks"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # runs locally, free
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

# LLM served through Hugging Face Inference Providers. Change it in .env if you like.
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
MAX_HISTORY_MESSAGES = 6  # last 3 question/answer pairs are remembered

NOT_FOUND_MESSAGE = "I could not find this information in the uploaded documents."



# Custom errors -> the UI catches these and shows a friendly message

class RAGError(Exception):
    """Base class for all expected, user-facing errors."""


class EmptyPDFError(RAGError):
    """The PDF has no extractable text (for example a scanned image PDF)."""


class VectorStoreNotReadyError(RAGError):
    """ChromaDB has no documents yet (PDFs were not processed)."""


class MissingTokenError(RAGError):
    """HF_TOKEN is not set."""


class LLMError(RAGError):
    """The Hugging Face API call failed or returned something unusable."""



# Phase 3 - PDF loading

def load_pdf(pdf_path):
    """Read one PDF and return a list of Documents (one per page with text).

    Metadata is cleaned so it is easy to display:
        source -> file name only (not the full path)
        page   -> human page number starting at 1 (PyPDF starts at 0)
    """
    pdf_path = Path(pdf_path)
    pages = PyPDFLoader(str(pdf_path)).load()

    documents = []
    for page in pages:
        if not page.page_content.strip():
            continue  # skip blank pages
        page.metadata["source"] = pdf_path.name
        page.metadata["page"] = int(page.metadata.get("page", 0)) + 1
        documents.append(page)

    if not documents:
        raise EmptyPDFError(
            f"'{pdf_path.name}' has no readable text. It may be empty or a scanned image PDF."
        )
    return documents



# Phase 4 - Chunking

def split_into_chunks(documents):
    """Split page-sized Documents into overlapping chunks.

    Each chunk keeps its page's metadata (source filename + page number).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


# Phase 5 - Embeddings (local, free)

@lru_cache(maxsize=1)
def get_embeddings():
    """Load the Sentence Transformers model once and reuse it."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)



# Phase 6 - ChromaDB

def build_vector_store(chunks):
    """Create a fresh ChromaDB collection from the chunks and save it to disk.

    The previous collection is deleted first so old PDFs do not mix with new ones.
    """
    if not chunks:
        raise EmptyPDFError("No text chunks were created, so there is nothing to index.")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Remove the old collection (not the folder: on Windows the DB file stays locked,
    # so deleting the folder would fail with a PermissionError).
    try:
        Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        ).delete_collection()
    except Exception:
        pass  # nothing to delete the first time

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )


def load_vector_store():
    """Open the saved ChromaDB. Raises VectorStoreNotReadyError if it is empty."""
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        raise VectorStoreNotReadyError("No documents have been processed yet.")

    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    if store._collection.count() == 0:
        raise VectorStoreNotReadyError("The vector database is empty. Process your PDFs first.")
    return store


def process_pdfs(pdf_paths):
    """Full indexing pipeline for several PDFs.

    Returns (vector_store, summary) where summary is a list of dicts like
    {"file": "AI.pdf", "pages": 12, "chunks": 30} or {"file": ..., "error": "..."}.
    A bad PDF is reported but does not stop the others.
    """
    if not pdf_paths:
        raise RAGError("Please upload at least one PDF first.")

    all_chunks = []
    summary = []
    for path in pdf_paths:
        name = Path(path).name
        try:
            documents = load_pdf(path)
            chunks = split_into_chunks(documents)
            all_chunks.extend(chunks)
            summary.append({"file": name, "pages": len(documents), "chunks": len(chunks)})
        except EmptyPDFError as error:
            summary.append({"file": name, "error": str(error)})
        except Exception as error:  # corrupted / encrypted PDF etc.
            summary.append({"file": name, "error": f"Could not read this PDF ({error})."})

    if not all_chunks:
        raise EmptyPDFError("None of the uploaded PDFs contained readable text.")

    return build_vector_store(all_chunks), summary


# ----------------------------------------------------------------------------
# Phase 7 - Retrieval
# ----------------------------------------------------------------------------
def retrieve_chunks(vector_store, question, top_k=TOP_K):
    """Embed the question and return the top_k most similar chunks.

    Returns a list of (Document, distance). A SMALLER distance means MORE similar.
    """
    return vector_store.similarity_search_with_score(question, k=top_k)


def format_sources(retrieved):
    """Turn retrieved chunks into a sorted, de-duplicated [(filename, page), ...]."""
    unique = {(doc.metadata["source"], doc.metadata["page"]) for doc, _ in retrieved}
    return sorted(unique)

# Phase 8 - Hugging Face LLM

def get_llm_client():
    """Create the Hugging Face Inference Providers client from HF_TOKEN."""
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        raise MissingTokenError(
            "HF_TOKEN is missing. Copy .env.example to .env and add your Hugging Face token."
        )
    return InferenceClient(api_key=token)


def call_llm(messages, max_tokens=700, temperature=0.1):
    """Send chat messages to the Hugging Face model and return the text reply."""
    client = get_llm_client()
    model = os.getenv("HF_MODEL", DEFAULT_LLM_MODEL)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as error:
        text = str(error)
        if "401" in text or "403" in text:
            raise LLMError("Hugging Face rejected the token. Check HF_TOKEN in your .env file.")
        if "402" in text or "429" in text:
            raise LLMError("Hugging Face usage limit reached. Wait a bit or try another model.")
        raise LLMError(f"Could not reach the Hugging Face model '{model}': {text}")

    # Defensive check for an unexpected response shape
    try:
        answer = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        raise LLMError("Hugging Face returned an unexpected response format.")
    if not answer or not answer.strip():
        raise LLMError("Hugging Face returned an empty answer. Please try again.")
    return answer.strip()


# Phase 9 - Conversational memory

def history_to_text(history):
    """Format the last few messages as plain text for prompts."""
    recent = history[-MAX_HISTORY_MESSAGES:]
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in recent)


def rewrite_followup_question(question, history):
    """Turn a follow-up like 'Give me an example' into a standalone question.

    Retrieval only looks at the text it is given, so 'Give me an example' alone would
    search for nothing useful. We ask the LLM to rewrite it using the chat history,
    e.g. 'Give an example of supervised learning'.
    """
    if not history:
        return question  # first question: nothing to resolve

    prompt = (
        "Rewrite the user's latest question as a single standalone question, using the "
        "conversation to replace words like 'it', 'that' or 'this'. Do NOT answer it. "
        "If it is already standalone, return it unchanged. Return ONLY the question.\n\n"
        f"Conversation:\n{history_to_text(history)}\n\n"
        f"Latest question: {question}\n\nStandalone question:"
    )
    try:
        rewritten = call_llm([{"role": "user", "content": prompt}], max_tokens=100, temperature=0)
    except LLMError:
        return question  # if rewriting fails, fall back to the original question
    return rewritten.strip().strip('"') or question



# Phase 8 + 9 + 11 - Putting it together: answer a question with sources

SYSTEM_PROMPT = f"""You are a helpful assistant that answers questions using ONLY the provided context from PDF documents.
Rules:
- Use only the context. Do not use outside knowledge and do not guess.
- If the context does not contain the answer, reply exactly: "{NOT_FOUND_MESSAGE}"
- Keep the answer clear and concise.
- Use the conversation history only to understand follow-up questions."""


def build_context(retrieved):
    """Join the retrieved chunks into one context block, labelled with their source."""
    parts = []
    for doc, _ in retrieved:
        label = f"[{doc.metadata['source']} - Page {doc.metadata['page']}]"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def answer_question(vector_store, question, history=None):
    """Answer a question from the PDFs.

    Returns a dict:
        answer        - the LLM's answer text
        sources       - [(filename, page), ...] used for the answer ([] if not found)
        search_query  - the (possibly rewritten) question used for retrieval
        retrieved     - [(Document, distance), ...] for transparency in the UI
    """
    history = history or []
    if vector_store is None:
        raise VectorStoreNotReadyError("Please upload and process your PDFs first.")

    # 1) Make follow-ups standalone so retrieval works
    search_query = rewrite_followup_question(question, history)

    # 2) Similarity search in ChromaDB
    retrieved = retrieve_chunks(vector_store, search_query)
    if not retrieved:
        return {"answer": NOT_FOUND_MESSAGE, "sources": [], "search_query": search_query, "retrieved": []}

    # 3) Ask the LLM, giving it the context + short history + the question
    user_prompt = (
        f"Context:\n{build_context(retrieved)}\n\n"
        f"Conversation so far:\n{history_to_text(history) or '(none)'}\n\n"
        f"Question: {search_query}\n\nAnswer:"
    )
    answer = call_llm(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )

    # 4) Only show sources if the model actually found an answer
    sources = [] if NOT_FOUND_MESSAGE.lower() in answer.lower() else format_sources(retrieved)
    return {"answer": answer, "sources": sources, "search_query": search_query, "retrieved": retrieved}
