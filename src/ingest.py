import os, glob
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
EMBED_MODEL = "gemini-embedding-001"

def chunk_text(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def embed(texts):
    res = client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return [e.values for e in res.embeddings]

def main():
    col = chromadb.PersistentClient(path="chroma_db").get_or_create_collection("docs")
    for path in glob.glob("data/*.pdf"):
        name = os.path.basename(path)
        for page_num, page in enumerate(PdfReader(path).pages, start=1):
            chunks = chunk_text(page.extract_text() or "")
            if not chunks:
                continue
            col.upsert(
                ids=[f"{name}-p{page_num}-c{i}" for i in range(len(chunks))],
                documents=chunks,
                embeddings=embed(chunks),
                metadatas=[{"source": name, "page": page_num}] * len(chunks),
            )
        print("Indexed", name)

if __name__ == "__main__":
    main()