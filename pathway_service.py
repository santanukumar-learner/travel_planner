
import os, json
from typing import Any, Dict, List

import pathway as pw
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

DATA_FILES = [
    os.getenv("ATTRACTIONS_JSON", "rag.json"),
    os.getenv("RESTAURANTS_JSON", "restaurants_rourkela.json"),
]
HOST = os.getenv("PATHWAY_HOST", "0.0.0.0")
PORT = int(os.getenv("PATHWAY_PORT", "8765"))

# IMPORTANT: for now we skip remote model downloads and use a simple placeholder embedder strategy.
# Once you share your current embedding model from rag.py, we’ll plug the same model here.
# If you already have embeddings in your JSON, we can use them directly.
EMBEDDER = os.getenv("PATHWAY_EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (str, int, float, bool)):
        return str(x)
    return json.dumps(x, ensure_ascii=False)


def _record_to_text(rec: Dict[str, Any], source: str) -> str:
    title = rec.get("title") or rec.get("name") or rec.get("place") or ""
    category = rec.get("category") or rec.get("type") or rec.get("cuisine") or ""
    desc = rec.get("description") or rec.get("about") or rec.get("summary") or ""
    address = rec.get("address") or rec.get("location") or ""
    tags = rec.get("tags") or rec.get("highlights") or ""

    core = [
        f"Source: {source}",
        f"Title: {_safe_str(title)}",
        f"Category: {_safe_str(category)}",
        f"Description: {_safe_str(desc)}",
        f"Address: {_safe_str(address)}",
        f"Tags: {_safe_str(tags)}",
        f"Raw: {json.dumps(rec, ensure_ascii=False)}",
    ]
    return "\n".join([c for c in core if c.strip()])


def _load_json_file(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for key in ("items", "data", "places", "restaurants", "attractions"):
            if key in data and isinstance(data[key], list):
                return [x for x in data[key] if isinstance(x, dict)]
        return [data]
    return []


def build_documents() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for fp in DATA_FILES:
        if not os.path.exists(fp):
            print(f"[Pathway] WARNING: file not found: {fp}")
            continue

        records = _load_json_file(fp)
        source = os.path.basename(fp)

        for i, rec in enumerate(records):
            text = _record_to_text(rec, source=source)
            meta = {
                "source": source,
                "row": i,
                "title": rec.get("title") or rec.get("name") or rec.get("place"),
                "category": rec.get("category") or rec.get("type") or rec.get("cuisine"),
            }
            docs.append({"text": text, "metadata": meta})
    print(f"[Pathway] Loaded {len(docs)} documents from {DATA_FILES}")
    return docs


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
