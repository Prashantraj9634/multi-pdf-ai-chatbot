# 📚 Multi-PDF AI Chatbot

Upload several PDFs and chat with them. Answers come from your documents only, and every answer shows its **sources (PDF name + page number)**.

## Project Overview

This is a **RAG (Retrieval-Augmented Generation)** application. Instead of asking an LLM to answer from its memory (where it can make things up), we first **search your PDFs for the relevant passages** and give only those passages to the LLM. The LLM then writes an answer from them.

## Features

- Upload multiple PDFs at once
- Free: local embeddings + Hugging Face Inference Providers (no OpenAI)
- ChromaDB vector database stored on disk
- Source citations (file + page) on every answer
- Follow-up questions ("Give me an example") work
- Says "I could not find this information…" when the answer is not in the PDFs
- Friendly errors: no PDFs, empty PDF, missing token, API failure, question before processing

## Architecture

```
PDF → PDF Loader → Text Chunks → Embeddings → ChromaDB
                                                 │
Question → (rewrite using history) → Embedding → Similarity Search
                                                 │
                                       Retrieved Context
                                                 │
                                       Hugging Face LLM
                                                 │
                                       Answer + Sources
```

| File | Job |
|---|---|
| `rag.py` | All RAG logic (load, chunk, embed, store, retrieve, generate, memory) |
| `app.py` | Streamlit screen only (sidebar + chat) |
| `check_setup.py` | One-time check of the embedding model and your HF token |

## Tech Stack

Python · LangChain · ChromaDB · Sentence Transformers (`all-MiniLM-L6-v2`) · Hugging Face Inference Providers · Streamlit · PyPDF · python-dotenv

## How RAG Works in This Project

1. **Indexing** (when you click *Process PDFs*): read PDF → split into chunks → turn each chunk into a vector → save in ChromaDB with `source` and `page`.
2. **Answering** (when you ask): turn the question into a vector → find the 4 closest chunks → put them in a prompt → LLM answers using only that context → show the sources.

## Installation (Windows)

Use Python 3.10–3.13. Open **PowerShell** in the project folder:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

(If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.)

On Windows, `pip` installs the small CPU version of PyTorch by default, so the install is fairly quick. macOS/Linux: use `source venv/bin/activate` instead of `venv\Scripts\activate`.

## Hugging Face Token Setup

1. Create a free account at https://huggingface.co and open **Settings → Access Tokens**.
2. Create a token. Use a **fine-grained** token with the permission **"Make calls to Inference Providers"** (a plain Read token may be rejected for this).
3. Copy the example file and edit it:

```powershell
copy .env.example .env
notepad .env
```

Set `HF_TOKEN=hf_xxxxxxxx` (your real token, no quotes). Never share or commit `.env`.

Optional: set `HF_MODEL=` to another chat model if the default is slow or out of free credits.

## Test Everything First (do this before the interview!)

```powershell
python check_setup.py
```

You should see an embedding size of 384 and `model replied: API OK`. The first run downloads the embedding model (~90 MB); after that it works offline.

## How to Run

```powershell
streamlit run app.py
```

It opens at http://localhost:8501.

## How to Upload PDFs

1. In the sidebar, click **Browse files** and select one or more PDFs.
2. Click **Process PDFs** and wait for "✅ Ready!".
3. Ask questions in the chat box. Click **Clear chat** to start a fresh conversation.

Note: scanned (image-only) PDFs have no text and are reported as "no readable text".

## How Retrieval Works

`similarity_search_with_score(question, k=4)` embeds the question with the same model used for the chunks and returns the 4 chunks whose vectors are closest. The "How this answer was found" panel in the app shows them with their distance (**lower = more similar**).

## How ChromaDB Is Used

ChromaDB stores, for every chunk: the text, its 384-number vector and metadata (`source`, `page`). It is saved in the `chroma_db/` folder, so no separate database server is needed. Clicking *Process PDFs* again replaces the old collection so old documents do not mix with new ones.

## How Embeddings Work

An embedding model turns text into a list of numbers (384 for MiniLM) so that texts with similar **meaning** get similar numbers. "Car" and "automobile" land close together even though the words differ. This runs on your own CPU, so it is free.

## How Conversational Memory Works

The app keeps the chat in `st.session_state.messages`. Retrieval only sees the text it is given, so "Give me an example" alone would search for nothing useful. Before searching, the LLM rewrites the follow-up into a standalone question using the last 6 messages ("Give an example of supervised learning"). That rewritten question is used for the search. The recent history is also included in the final prompt.

## Example Questions

- What is supervised learning?
- Give me an example. *(follow-up)*
- Summarize the main idea of the second document.
- What does the document say about data privacy?
- Who won the football world cup? *(should answer "not found")*

## Troubleshooting

| Problem | Fix |
|---|---|
| `HF_TOKEN is missing` | Create `.env` from `.env.example` and add your token; restart Streamlit |
| `Hugging Face rejected the token` | Token needs "Make calls to Inference Providers" permission; or it was copied wrongly |
| `usage limit reached` | Free credits used up; wait, or set another `HF_MODEL` |
| First processing is slow | The embedding model is downloading once (~90 MB) |
| `no readable text` | The PDF is scanned images; use a text-based PDF |
| Answer says "not found" for something in the PDF | Rephrase using words from the document |
| `streamlit` not recognized | Activate the venv first (`venv\Scripts\activate`) |

## Interview Explanation

**30-second pitch:** "It's a RAG chatbot over multiple PDFs. PDFs are split into overlapping chunks, embedded locally with Sentence Transformers, and stored in ChromaDB with page metadata. For each question I run a similarity search, pass the top 4 chunks to a Hugging Face LLM with a strict prompt, and show the answer with file and page citations. Follow-ups work because I rewrite them into standalone questions using chat history."

### 15 Likely Interview Questions and Answers

**1. What is RAG and why use it?**
Retrieval-Augmented Generation: retrieve relevant text first, then let the LLM answer from it. It lets an LLM answer about private or new documents without retraining, and reduces hallucination.

**2. Walk me through your pipeline.**
Upload → PyPDFLoader → RecursiveCharacterTextSplitter (1000/200) → MiniLM embeddings → ChromaDB. At question time: embed question → top-4 similarity search → prompt with context → HF LLM → answer + sources.

**3. Why chunk the documents?**
LLMs have limited context, and embedding a whole document blurs many topics into one vector. Small chunks give precise retrieval and cheaper prompts.

**4. Why chunk size 1000 and overlap 200?**
1000 characters is roughly a paragraph or two: enough context, still focused. Overlap of 200 keeps a sentence that straddles a boundary intact in at least one chunk. These are common starting values; I'd tune them by testing answer quality.

**5. Why RecursiveCharacterTextSplitter?**
It tries to split at paragraphs, then lines, then sentences/words, so chunks break at natural boundaries rather than mid-sentence.

**6. What are embeddings?**
Numeric vectors that represent meaning. Similar meaning → nearby vectors. MiniLM gives 384 dimensions.

**7. Why `all-MiniLM-L6-v2`?**
Small (~90 MB), fast on CPU, free, good quality for semantic search, and runs locally so documents are not sent to a third party for embedding.

**8. How does similarity search work?**
The question is embedded with the same model, then ChromaDB finds the stored vectors closest to it (by distance; smaller means more similar) and returns the top k=4.

**9. Why ChromaDB?**
Simple, open source, runs inside the Python process, saves to a local folder, and integrates directly with LangChain. For an MVP, no server is needed. For large production scale I'd consider a managed or server-based vector DB (e.g. pgvector, Pinecone).

**10. How do you stop hallucinations?**
The prompt says to use only the context, not guess, and to reply with a fixed "not found" message otherwise. Temperature is low (0.1). Retrieval limits what the model sees. Sources are shown so answers can be verified. This reduces hallucination but cannot eliminate it.

**11. How does conversational memory work?**
Chat history is stored in Streamlit session state. Follow-up questions are rewritten by the LLM into standalone questions before retrieval, and the last few messages are added to the prompt. It's a simple window of 6 messages, not a complex agent.

**12. How do you produce source citations?**
Each chunk keeps `source` (file name) and `page` in its metadata. After retrieval I collect the unique (file, page) pairs and display them. If the answer is "not found", no sources are shown.

**13. Why Hugging Face instead of OpenAI?**
Cost: it has a free tier, and I wanted no paid API usage. Embeddings run fully locally. Open models also avoid vendor lock-in; because the LLM call is in one function, swapping providers is easy.

**14. What are the limitations and how would you improve it?**
Scanned PDFs need OCR; tables and images are not understood; the chunk size is not tuned; there is no reranking. Improvements: OCR, a reranker, hybrid (keyword + vector) search, streaming answers, evaluation of answer quality, and an agentic layer with tools.

**15. How would you scale it?**
Persist per-user collections, process PDFs in a background job, cache embeddings, use a hosted vector DB, add a FastAPI backend, and add authentication and monitoring.

### Why these choices (short answers)

- **Chunking:** splits long text into small overlapping pieces so retrieval is precise and the overlap avoids cutting ideas in half.
- **Embeddings:** meaning as numbers; same model for chunks and questions.
- **Similarity search:** nearest vectors to the question vector = most relevant passages.
- **Memory:** store history, rewrite follow-ups into standalone questions.
- **Hallucination prevention:** strict grounded prompt, low temperature, "not found" fallback, visible sources.
- **Hugging Face over OpenAI:** free, no paid API, local embeddings, easy to swap.

## Project Structure

```
multi-pdf-ai-chatbot/
├── app.py            # Streamlit UI
├── rag.py            # RAG engine
├── check_setup.py    # one-time test of embeddings + HF token
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/PDFs/        # uploaded PDFs are saved here
└── chroma_db/        # ChromaDB files are saved here
```
