import sys
import time
import chromadb
from ingest import embed, client

GEN_MODEL = "gemini-3.8-flash"   # change if you get a 404 model error


def generate_with_retry(prompt, retries=4):
    """Call Gemini, retrying when the server is busy (503) or rate limited (429)."""
    for attempt in range(retries):
        try:
            return client.models.generate_content(model=GEN_MODEL, contents=prompt).text
        except Exception as e:
            msg = str(e)
            retryable = "503" in msg or "429" in msg
            if attempt == retries - 1 or not retryable:
                raise
            wait = 5 * 2 ** attempt   # 5s, 10s, 20s
            print(f"Server busy, retrying in {wait}s...")
            time.sleep(wait)


def answer(question, k=4):
    col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")
    res = col.query(query_embeddings=[embed([question])[0]], n_results=k)
    docs, metas = res["documents"][0], res["metadatas"][0]

    context = "\n\n".join(
        f"[{m['source']} p.{m['page']}]\n{d}" for d, m in zip(docs, metas)
    )
    prompt = f"""Answer using only the context below. If the answer is not
in the context, say you don't know. Cite sources like [file p.N].

Context:
{context}

Question: {question}"""

    text = generate_with_retry(prompt)
    return text, metas, docs


if __name__ == "__main__":
    question = " ".join(sys.argv[1:])
    if not question:
        print('Usage: python src/answer.py "your question"')
        sys.exit(1)

    text, metas, docs = answer(question)
    print(text)
    print("\n--- Sources retrieved ---")
    for m, d in zip(metas, docs):
        snippet = d[:100].replace("\n", " ")
        print(f"{m['source']} (page {m['page']}): {snippet}...")