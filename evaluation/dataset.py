"""Validate schema, stable IDs, evidence, and corpus hashes before scoring."""

import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from retrieval_documents import build_documents_from_files


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evaluation/datasets/rourkela_v1.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_sha256(path):
    # Ignore file whitespace/CRLF changes, but retain record and property order:
    # property order affects the Raw representation sent to the semantic embedder.
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_dataset(path=DEFAULT_DATASET, root=ROOT):
    path, root = Path(path), Path(root)
    dataset = json.loads(path.read_text(encoding="utf-8"))
    schema = json.loads((ROOT / "evaluation/schemas/dataset.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(dataset)
    files = []
    for source in dataset["corpus"]:
        corpus_path = (root / source["path"]).resolve()
        if not corpus_path.is_relative_to(root.resolve()):
            raise ValueError("Corpus files must be inside the project")
        if json_sha256(corpus_path) != source["sha256"]:
            raise ValueError(f"Corpus changed: {source['path']}; review labels and version the benchmark")
        files.append(corpus_path)
    documents = build_documents_from_files(files)
    by_id = {doc["doc_id"]: doc for doc in documents}
    seen_ids, seen_questions = set(), set()
    for case in dataset["cases"]:
        normalized = " ".join(case["question"].casefold().split())
        if case["id"] in seen_ids or normalized in seen_questions:
            raise ValueError(f"Duplicate case ID or question: {case['id']}")
        seen_ids.add(case["id"])
        seen_questions.add(normalized)
        if case["answerable"] != bool(case["relevant_doc_ids"]):
            raise ValueError(f"Answerability/labels disagree: {case['id']}")
        evidence_ids = {item["doc_id"] for item in case["evidence"]}
        if evidence_ids != set(case["relevant_doc_ids"]):
            raise ValueError(f"Each relevant document needs evidence: {case['id']}")
        for item in case["evidence"]:
            if item["doc_id"] not in by_id:
                raise ValueError(f"Unknown relevant document: {case['id']}")
            record_text = json.dumps(by_id[item["doc_id"]]["record"], ensure_ascii=False)
            if item["quote"] not in record_text:
                raise ValueError(f"Evidence no longer matches the corpus: {case['id']}")
    return dataset, documents


def validate_prediction(prediction, case, known_ids):
    schema = json.loads((ROOT / "evaluation/schemas/prediction.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(prediction)
    if prediction["id"] != case["id"]:
        raise ValueError("Prediction ID does not match case")
    if set(prediction["retrieved_doc_ids"]) - set(known_ids):
        raise ValueError(f"Prediction contains unknown document IDs: {case['id']}")
