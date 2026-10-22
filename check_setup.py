"""
check_setup.py - run this ONCE on your machine before the demo:

    python check_setup.py

It tests the two things that need the internet:
  1. the local embedding model (downloads ~90 MB the first time, then works offline)
  2. the real Hugging Face LLM call using your HF_TOKEN
"""

import rag

print("1) Loading embedding model (first run downloads it, please wait)...")
vector = rag.get_embeddings().embed_query("hello world")
print(f"   OK - embedding has {len(vector)} numbers (expected 384)\n")

print("2) Calling the Hugging Face LLM...")
try:
    reply = rag.call_llm([{"role": "user", "content": "Reply with exactly: API OK"}], max_tokens=200)
    print(f"   OK - model replied: {reply}")
except rag.RAGError as error:
    print(f"   FAILED - {error}")
