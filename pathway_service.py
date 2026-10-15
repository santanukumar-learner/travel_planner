
import os
from typing import Any, Dict, List

from retrieval_documents import build_documents_from_files
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ----------------------------
# Config
# ----------------------------
DATA_FILES = [
    os.getenv("ATTRACTIONS_JSON", "rag.json"),
    os.getenv("RESTAURANTS_JSON", "restaurants_rourkela.json"),
]
HOST = os.getenv("PATHWAY_HOST", "0.0.0.0")
PORT = int(os.getenv("PATHWAY_PORT", "8765"))

# The embedding model is loaded at startup and may download weights on its first run.
EMBEDDER = os.getenv("PATHWAY_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")


def build_documents() -> List[Dict[str, Any]]:
    existing = []
    for path in DATA_FILES:
        if os.path.exists(path):
            existing.append(path)
        else:
            print(f"[Pathway] WARNING: file not found: {path}")
    docs = build_documents_from_files(existing)
    print(f"[Pathway] Loaded {len(docs)} documents from {DATA_FILES}")
    return docs


# ----------------------------
# In-memory SentenceTransformer embeddings with NumPy similarity
# ----------------------------
# The historical service filename is retained; no Pathway streaming graph is used.
from sentence_transformers import SentenceTransformer
import numpy as np

_DOCS: List[Dict[str, Any]] = []
_EMB = None
_MODEL = None

def init_index():
    global _DOCS, _EMB, _MODEL
    _DOCS = build_documents()
    _MODEL = SentenceTransformer(EMBEDDER)
    vectors = _MODEL.encode([d["text"] for d in _DOCS], normalize_embeddings=True)
    _EMB = np.asarray(vectors, dtype=np.float32)
    print("[Pathway] Index ready. Serving /search ...")


def search(query: str, k: int = 5):
    qv = _MODEL.encode([query], normalize_embeddings=True).astype(np.float32)[0]
    scores = _EMB @ qv
    idx = np.argsort(-scores)[:k].tolist()
    out = []
    for i in idx:
        out.append({
            "text": _DOCS[i]["text"],
            "metadata": _DOCS[i]["metadata"],
            "score": float(scores[i]),
        })
    return out


# ----------------------------
# HTTP API
# ----------------------------
app = FastAPI()

class SearchReq(BaseModel):
    query: str
    k: int = 5

@app.on_event("startup")
def _startup():
    init_index()

@app.get("/health")
def health():
    return {"status": "ok", "docs": len(_DOCS)}

@app.post("/search")
def do_search(req: SearchReq):
    return {"results": search(req.query, req.k)}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
