import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY. Put it in .env and restart the terminal session.")
client = OpenAI(api_key=api_key)

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"


@dataclass
class Chunk:
    id: str
    text: str
    meta: Dict[str, Any]


def load_pdf_pages(path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        pages.append((i + 1, page.extract_text() or ""))
    return pages


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = (text or "").replace("\n", " ").strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_chunks_from_pdf(pdf_path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page_num, page_text in load_pdf_pages(pdf_path):
        page_chunks = chunk_text(page_text)
        for j, c in enumerate(page_chunks):
            chunks.append(
                Chunk(
                    id=f"{os.path.basename(pdf_path)}:p{page_num}:c{j}",
                    text=c,
                    meta={"source": pdf_path, "page": page_num, "chunk_index": j},
                )
            )
    return chunks


def build_chunks_from_transcript(txt_path: str) -> List[Chunk]:
    text = load_text_file(txt_path)
    raw_chunks = chunk_text(text)
    return [
        Chunk(
            id=f"{os.path.basename(txt_path)}:c{i}",
            text=c,
            meta={"source": txt_path, "chunk_index": i},
        )
        for i, c in enumerate(raw_chunks)
    ]


def get_embeddings_batch(texts: List[str], model: str = EMBED_MODEL, batch_size: int = 100) -> List[List[float]]:
    all_embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        batch_embeddings = [item.embedding for item in resp.data]
        all_embeddings.extend(batch_embeddings)
        print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks")
    return all_embeddings


def embed_query(query: str, model: str = EMBED_MODEL) -> List[float]:
    resp = client.embeddings.create(model=model, input=query)
    return resp.data[0].embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def retrieve_top_k(query_embedding: List[float], chunk_embeddings: List[List[float]], k: int = 5) -> List[Tuple[int, float]]:
    q = np.array(query_embedding, dtype=np.float32)
    sims: List[Tuple[int, float]] = []
    for idx, emb in enumerate(chunk_embeddings):
        e = np.array(emb, dtype=np.float32)
        sims.append((idx, cosine_similarity(q, e)))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]


def format_context(selected_chunks: List[Chunk]) -> str:
    return "\n\n---\n\n".join([f"[{ch.id}]\n{ch.text}" for ch in selected_chunks])


def rag_answer(question: str, chunks: List[Chunk], chunk_embeddings: List[List[float]], top_k: int = 5) -> Dict[str, Any]:
    q_emb = embed_query(question)
    top = retrieve_top_k(q_emb, chunk_embeddings, k=top_k)

    selected = [chunks[idx] for idx, _score in top]
    context = format_context(selected)

    system = (
        "You are a helpful assistant answering strictly from the provided context.\n"
        "Rules:\n"
        "1) Use ONLY the context. If the context doesn't contain the answer, say you don't know.\n"
        "2) Add citations using the bracketed chunk IDs, like [file.pdf:p2:c0].\n"
        "3) Be concise and accurate.\n"
    )

    user = f"Question:\n{question}\n\nContext:\n{context}"

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

    return {
        "answer": resp.choices[0].message.content,
        "retrieved_chunks": [ch.id for ch in selected],
        "scores": top,
    }


def auto_detect_docs() -> Tuple[str | None, str | None]:
    pdf = None
    txt = None
    if not os.path.isdir("docs"):
        return None, None
    for fn in os.listdir("docs"):
        p = os.path.join("docs", fn)
        if fn.lower().endswith(".pdf") and pdf is None:
            pdf = p
        if fn.lower().endswith(".txt") and txt is None:
            txt = p
    return pdf, txt


def main():
    pdf_path, transcript_path = auto_detect_docs()
    if not pdf_path and not transcript_path:
        raise FileNotFoundError("No docs found. Put a .pdf and/or .txt into ./docs/")

    chunks: List[Chunk] = []
    if pdf_path:
        print(f"Loading PDF: {pdf_path}")
        chunks += build_chunks_from_pdf(pdf_path)
    if transcript_path:
        print(f"Loading transcript: {transcript_path}")
        chunks += build_chunks_from_transcript(transcript_path)

    chunks = [c for c in chunks if c.text.strip()]
    if not chunks:
        raise RuntimeError("Extracted zero text. Your PDF may be scanned/image-based (needs OCR).")

    print(f"Total chunks: {len(chunks)}")

    texts = [c.text for c in chunks]
    chunk_embeddings = get_embeddings_batch(texts)

    while True:
        q = input("\nAsk a question (or type 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        result = rag_answer(q, chunks, chunk_embeddings, top_k=5)
        print("\nANSWER:\n" + (result["answer"] or ""))
        print("\nRetrieved chunk IDs:")
        for (idx, score), cid in zip(result["scores"], result["retrieved_chunks"]):
            print(f" - {cid} | score={score:.4f}")


if __name__ == "__main__":
    main()
