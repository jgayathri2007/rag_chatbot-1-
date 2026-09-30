import os
import json
import re
from pathlib import Path

import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
INDEX_FILE = DATA_DIR / "index.json"

# Small and practical embedding model for a starter project.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Ollama runs the LLM locally. Install Ollama and pull a model first.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


class RAGEngine:
    def __init__(self):
        DATA_DIR.mkdir(exist_ok=True)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.records = []
        self.embeddings = np.empty((0, 384), dtype=np.float32)
        self._load()

    def _load(self):
        if not INDEX_FILE.exists():
            return
        try:
            payload = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
            self.records = payload.get("records", [])
            vectors = payload.get("embeddings", [])
            if vectors:
                self.embeddings = np.array(vectors, dtype=np.float32)
        except Exception:
            self.records = []
            self.embeddings = np.empty((0, 384), dtype=np.float32)

    def _save(self):
        INDEX_FILE.write_text(
            json.dumps(
                {"records": self.records, "embeddings": self.embeddings.tolist()},
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

    @staticmethod
    def _clean(text):
        return re.sub(r"\s+", " ", text or "").strip()

    def _chunk(self, text, chunk_size=700, overlap=120):
        text = self._clean(text)
        if not text:
            return []

        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def add_pdf(self, path, filename):
        reader = PdfReader(path)
        new_records = []

        for page_no, page in enumerate(reader.pages, start=1):
            text = self._clean(page.extract_text())
            for chunk in self._chunk(text):
                new_records.append({
                    "file": filename,
                    "page": page_no,
                    "text": chunk
                })

        if not new_records:
            return 0

        texts = [r["text"] for r in new_records]
        vectors = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)

        # Remove an earlier copy of the same filename.
        keep = [i for i, r in enumerate(self.records) if r["file"] != filename]
        if keep:
            self.records = [self.records[i] for i in keep]
            self.embeddings = self.embeddings[keep]
        else:
            self.records = []
            self.embeddings = np.empty((0, vectors.shape[1]), dtype=np.float32)

        self.records.extend(new_records)
        self.embeddings = np.vstack([self.embeddings, vectors])
        self._save()
        return len(new_records)

    def clear(self):
        self.records = []
        self.embeddings = np.empty((0, 384), dtype=np.float32)
        if INDEX_FILE.exists():
            INDEX_FILE.unlink()

    def retrieve(self, question, top_k=4):
        if not self.records:
            return []

        q = self.embedder.encode(
            [question],
            normalize_embeddings=True,
            show_progress_bar=False
        )[0].astype(np.float32)

        scores = self.embeddings @ q
        indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                **self.records[i],
                "score": float(scores[i])
            }
            for i in indices
            if float(scores[i]) >= 0.20
        ]

    def _generate(self, prompt):
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    def ask(self, question):
        results = self.retrieve(question)

        if not results:
            return (
                "I couldn't find relevant information in the uploaded documents. "
                "Please upload the correct document or ask a question related to it."
            ), []

        context_parts = []
        for i, r in enumerate(results, start=1):
            context_parts.append(
                f"[Source {i} | {r['file']} | page {r['page']}]\n{r['text']}"
            )
        context = "\n\n".join(context_parts)

        prompt = f"""You are a helpful document question-answering assistant.

Answer the user's question using ONLY the provided document context.
If the context does not contain the answer, say that the answer is not available
in the uploaded documents. Do not invent facts.

Context:
{context}

User question:
{question}

Answer clearly and concisely. When useful, mention the source document/page.
"""

        try:
            answer = self._generate(prompt)
        except requests.exceptions.ConnectionError:
            answer = (
                "The document search worked, but the local LLM is not running. "
                "Please start Ollama and make sure the configured model is available."
            )
        except Exception as exc:
            answer = f"LLM error: {exc}"

        sources = [
            {"file": r["file"], "page": r["page"]}
            for r in results
        ]
        return answer, sources
