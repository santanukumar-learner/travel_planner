import argparse
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import shutil
from urllib.error import HTTPError

import pytest
from jsonschema import Draft202012Validator, ValidationError

from evaluation import run as runner
from evaluation.backends import BM25Retriever, GeminiClient, HTTPRetriever, JUDGE_SCHEMA
from evaluation.dataset import ROOT, load_dataset, json_sha256, validate_prediction
from evaluation.metrics import ABSTENTION, corpus_bleu, retrieval_scores
from retrieval_documents import build_documents_from_files


@pytest.fixture
def corpus():
    return load_dataset()


@pytest.fixture
def small_dataset(tmp_path, corpus):
    dataset, _ = corpus
    data = deepcopy(dataset)
    data["dataset_id"] = "rourkela-test-fixture"
    data["cases"] = [data["cases"][0], data["cases"][90]]
    path = tmp_path / "small.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def arguments(tmp_path, dataset, **overrides):
    values = dict(dataset=dataset, output=tmp_path / "results", k=5, retriever="bm25",
                  retriever_url="http://127.0.0.1:8766", generator="extractive",
                  model="test-model", judge_model=None, predictions=None, baseline=None, max_drop=0.02)
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("ranked,relevant,k,hit,rr,recall", [
    (["a", "b", "c"], ["b"], 3, 1, 0.5, 1),
    (["a", "b", "c"], ["c"], 2, 0, 0, 0),
    (["a", "b", "c"], ["a", "c"], 2, 1, 1, 0.5),
    ([], ["a"], 5, 0, 0, 0),
])
def test_retrieval_metrics_respect_cutoff_and_multiple_relevant_docs(ranked, relevant, k, hit, rr, recall):
    score = retrieval_scores(ranked, relevant, k)
    assert score == {"hit": hit, "reciprocal_rank": rr, "recall": recall}


def test_unsupported_query_does_not_count_as_a_retrieval_miss():
    assert retrieval_scores(["a"], [], 5) == {"hit": None, "reciprocal_rank": None, "recall": None}


def test_duplicate_ranks_cannot_inflate_retrieval_scores():
    with pytest.raises(ValueError, match="Duplicate"):
        retrieval_scores(["a", "a", "b"], ["b"], 3)


def test_bleu_is_normalized_corpus_score_with_reference_streams():
    refs = [["the park has a toy train", "there is a toy train in the park"], ["visit the temple in the evening"]]
    score = corpus_bleu([row[0] for row in refs], refs)
    assert score["score"] == pytest.approx(1.0)
    assert "tok:13a" in score["signature"]
    assert corpus_bleu(["", ""], refs)["score"] == 0


def test_dataset_has_100_authored_cases_and_no_fake_users(corpus):
    dataset, documents = corpus
    assert dataset["origin"] == "curated"
    assert len(dataset["cases"]) == 100
    assert sum(case["answerable"] for case in dataset["cases"]) == 90
    assert len(documents) == 70
    assert len({doc["doc_id"] for doc in documents}) == len(documents)


def test_corpus_changes_require_label_review(tmp_path, corpus):
    dataset, _ = corpus
    for source in dataset["corpus"]:
        shutil.copyfile(ROOT / source["path"], tmp_path / source["path"])
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    changed = json.loads((tmp_path / "rag.json").read_text(encoding="utf-8"))
    changed[0]["content"] += " Changed fact."
    (tmp_path / "rag.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="Corpus changed"):
        load_dataset(path, root=tmp_path)


def test_json_formatting_and_windows_line_endings_do_not_break_reproduction(tmp_path, corpus):
    dataset, _ = corpus
    for source in dataset["corpus"]:
        rows = json.loads((ROOT / source["path"]).read_text(encoding="utf-8"))
        output = tmp_path / source["path"]
        output.write_bytes(json.dumps(rows, indent=1, ensure_ascii=False).replace("\n", "\r\n").encode("utf-8"))
        assert json_sha256(output) == source["sha256"]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    assert load_dataset(path, root=tmp_path)[0]["dataset_version"] == "1.0.0"


@pytest.mark.parametrize("corruption", ["duplicate_question", "unknown_document", "bad_evidence"])
def test_bad_annotations_fail_before_scoring(tmp_path, corpus, corruption):
    data = deepcopy(corpus[0])
    if corruption == "duplicate_question":
        data["cases"][1]["question"] = data["cases"][0]["question"].upper()
    elif corruption == "unknown_document":
        data["cases"][0]["relevant_doc_ids"] = ["rag.json:not-present"]
        data["cases"][0]["evidence"][0]["doc_id"] = "rag.json:not-present"
    else:
        data["cases"][0]["evidence"][0]["quote"] = "An invented fact not in the corpus."
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path)


def test_replay_requires_all_cases_and_rejects_unknown_documents(tmp_path, corpus):
    dataset, documents = corpus
    case = dataset["cases"][0]
    path = tmp_path / "predictions.jsonl"
    prediction = {"id": case["id"], "retrieved_doc_ids": case["relevant_doc_ids"], "answer": "answer"}
    path.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="complete benchmark"):
        runner.read_predictions(path, dataset["cases"], {doc["doc_id"] for doc in documents})
    prediction["retrieved_doc_ids"] = ["unknown"]
    with pytest.raises(ValueError, match="unknown document"):
        validate_prediction(prediction, case, {doc["doc_id"] for doc in documents})


def test_http_retriever_maps_legacy_metadata_and_checks_corpus(monkeypatch, corpus):
    _, documents = corpus
    doc = documents[0]
    result = {"text": doc["text"], "metadata": {"source": doc["metadata"]["source"], "row": 0}, "score": 0.9}
    captured = []

    def response(req, timeout):
        captured.append(json.loads(req.data))
        return BytesIO(json.dumps({"results": [result]}).encode())

    monkeypatch.setattr("evaluation.backends.request.urlopen", response)
    retriever = HTTPRetriever("http://127.0.0.1:8766", documents)
    assert retriever.search("test question", 5)[0]["doc_id"] == doc["doc_id"]
    assert captured == [{"query": "test question", "k": 5}]
    result["text"] = "stale index"
    with pytest.raises(ValueError, match="differs"):
        retriever.search("test", 5)


def test_schema_validates_actual_judge_dimensions():
    validator = Draft202012Validator(JUDGE_SCHEMA)
    valid = {"groundedness": 5, "relevance": 4, "reference_coverage": 3, "reason": "Missing a supporting fact."}
    validator.validate(valid)
    for bad in [dict(valid, groundedness=6), dict(valid, relevance=2.5), {key: value for key, value in valid.items() if key != "reason"}]:
        with pytest.raises(ValidationError):
            validator.validate(bad)


def test_gemini_client_uses_header_auth_and_keeps_provider_model_version(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-invalid-key")
    payloads = []

    def response(req, timeout):
        assert req.get_header("X-goog-api-key") == "test-only-invalid-key"
        assert "key=" not in req.full_url
        payloads.append(json.loads(req.data))
        return BytesIO(json.dumps({"candidates": [{"content": {"parts": [{"text": '{"groundedness":5}'}]}}],
                                  "usageMetadata": {"promptTokenCount": 12}, "modelVersion": "test-model-version"}).encode())

    monkeypatch.setattr("evaluation.backends.request.urlopen", response)
    answer, usage = GeminiClient("test-model").generate("prompt", JUDGE_SCHEMA)
    assert answer == '{"groundedness":5}'
    assert usage["promptTokenCount"] == 12
    assert usage["modelVersion"] == "test-model-version"
    assert payloads[0]["generationConfig"]["responseJsonSchema"] == JUDGE_SCHEMA


def test_provider_errors_do_not_log_secrets(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-only-invalid-key")

    def denied(req, timeout):
        raise HTTPError(req.full_url, 403, "provider body contains private detail", None, None)

    monkeypatch.setattr("evaluation.backends.request.urlopen", denied)
    with pytest.raises(RuntimeError, match="Gemini HTTP 403") as exc:
        GeminiClient("test-model").generate("prompt")
    assert "private detail" not in str(exc.value)
    assert "test-only-invalid-key" not in str(exc.value)


def test_observed_live_queue_failure_is_reported(corpus):
    fixture = json.loads((ROOT / "evaluation/fixtures/unsupported_queue_failure.json").read_text(encoding="utf-8"))
    data = deepcopy(corpus[0])
    data["cases"] = [next(case for case in data["cases"] if case["id"] == fixture["prediction"]["id"])]
    config = {"generator": "replay", "k": 5, "judge_model": None}
    summary = runner.summarize(data, [fixture["prediction"]], config)
    assert summary["metrics"]["hit_rate_at_k"] is None
    assert summary["metrics"]["unsupported_abstention_accuracy"] == 0
    assert len(summary["unsupported_answer_failures"]) == 1
    corrected = dict(fixture["prediction"], answer=ABSTENTION)
    assert runner.summarize(data, [corrected], config)["metrics"]["unsupported_abstention_accuracy"] == 1


def test_offline_cli_writes_complete_traces_and_can_replay(tmp_path, small_dataset):
    args = arguments(tmp_path, small_dataset)
    assert runner.run(args) == 0
    summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    assert summary["coverage"] == {"cases_scored": 2, "answers_present": 2, "errors": 0, "judged_cases": 0}
    assert summary["llm_judge"]["status"] == "not_run"
    traces = [json.loads(line) for line in (args.output / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(traces) == 2
    assert "retrieval" in traces[0]["latency_ms"]
    assert "metrics" in traces[0]
    replay = arguments(tmp_path, small_dataset, output=tmp_path / "replay",
                       predictions=args.output / "predictions.jsonl", baseline=args.output / "summary.json")
    assert runner.run(replay) == 0
    replay_summary = json.loads((replay.output / "summary.json").read_text(encoding="utf-8"))
    assert replay_summary["metrics"] == summary["metrics"]
    assert replay_summary["regression"]["passed"]
    with pytest.raises(ValueError, match="not empty"):
        runner.run(args)


def test_generation_failures_keep_full_denominator_and_abort_on_auth_failure(tmp_path, small_dataset, monkeypatch):
    calls = []

    class DeniedModel:
        def __init__(self, model):
            pass

        def generate(self, prompt, schema=None):
            calls.append(prompt)
            raise RuntimeError("Gemini HTTP 403; check access")

    monkeypatch.setattr(runner, "GeminiClient", DeniedModel)
    args = arguments(tmp_path, small_dataset, generator="gemini")
    assert runner.run(args) == 1
    summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    assert len(calls) == 1
    assert summary["coverage"]["cases_scored"] == 2
    assert summary["coverage"]["errors"] == 2
    assert summary["coverage"]["answers_present"] == 0
    assert summary["metrics"]["corpus_bleu"]["score"] == 0


def test_invalid_judge_rating_is_never_reported_as_a_measurement(tmp_path, small_dataset, monkeypatch):
    class BadJudge:
        def __init__(self, model):
            pass

        def generate(self, prompt, schema=None):
            return json.dumps({"groundedness": 99, "relevance": 5, "reference_coverage": 5, "reason": "Invalid."}), {}

    monkeypatch.setattr(runner, "GeminiClient", BadJudge)
    args = arguments(tmp_path, small_dataset, judge_model="mock-invalid-judge")
    assert runner.run(args) == 1
    summary = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    assert summary["llm_judge"]["status"] == "failed"
    assert summary["coverage"]["judged_cases"] == 0


def test_traces_survive_an_interrupted_run(tmp_path, small_dataset, monkeypatch):
    original = runner.extractive_answer
    calls = 0

    def interrupted(query, documents):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return original(query, documents)

    monkeypatch.setattr(runner, "extractive_answer", interrupted)
    args = arguments(tmp_path, small_dataset)
    with pytest.raises(KeyboardInterrupt):
        runner.run(args)
    saved = (args.output / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(saved) == 1
    assert json.loads(saved[0])["id"] == "rkl-001"
    assert not (args.output / "summary.json").exists()


def test_regression_gate_fails_a_drop_and_rejects_a_changed_dataset(tmp_path, small_dataset):
    args = arguments(tmp_path, small_dataset)
    assert runner.run(args) == 0
    baseline = json.loads((args.output / "summary.json").read_text(encoding="utf-8"))
    candidate = deepcopy(baseline)
    candidate["metrics"]["mrr_at_k"] -= 0.1
    assert runner.check_regression(candidate, baseline, 0.02)["regressed_metrics"] == ["mrr_at_k"]
    candidate["configuration"]["dataset_sha256"] = "different-labels"
    with pytest.raises(ValueError, match="different labels"):
        runner.check_regression(candidate, baseline, 0.02)
