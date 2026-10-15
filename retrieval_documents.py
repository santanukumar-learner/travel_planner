"""Shared corpus serialization for the search API and evaluation runners."""

import json
from pathlib import Path


def _safe_str(value):
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def record_to_text(record, source):
    # Preserve the search service's document representation.
    title = record.get("title") or record.get("name") or record.get("place") or ""
    category = record.get("category") or record.get("type") or record.get("cuisine") or ""
    description = record.get("description") or record.get("about") or record.get("summary") or ""
    address = record.get("address") or record.get("location") or ""
    tags = record.get("tags") or record.get("highlights") or ""
    return "\n".join([
        f"Source: {source}", f"Title: {_safe_str(title)}",
        f"Category: {_safe_str(category)}", f"Description: {_safe_str(description)}",
        f"Address: {_safe_str(address)}", f"Tags: {_safe_str(tags)}",
        f"Raw: {json.dumps(record, ensure_ascii=False)}",
    ])


def load_records(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [record for record in data if isinstance(record, dict)]
    if isinstance(data, dict):
        for key in ("items", "data", "places", "restaurants", "attractions"):
            if isinstance(data.get(key), list):
                return [record for record in data[key] if isinstance(record, dict)]
        return [data]
    raise ValueError(f"Unsupported JSON corpus: {Path(path).name}")


def build_documents_from_files(paths):
    documents = []
    seen = set()
    for path in paths:
        source = Path(path).name
        for row, record in enumerate(load_records(path)):
            record_id = record.get("id") or record.get("place_id") or f"row-{row}"
            doc_id = f"{source}:{record_id}"
            if doc_id in seen:
                raise ValueError(f"Duplicate corpus document ID: {doc_id}")
            seen.add(doc_id)
            documents.append({
                "doc_id": doc_id,
                "text": record_to_text(record, source),
                "record": record,
                "metadata": {
                    "doc_id": doc_id, "source": source, "row": row,
                    "title": record.get("title") or record.get("name") or record.get("place"),
                    "category": record.get("category") or record.get("type") or record.get("cuisine"),
                },
            })
    return documents
