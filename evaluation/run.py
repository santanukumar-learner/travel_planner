"""Run with python -m evaluation.run; exits nonzero on errors or regression."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter
from urllib.parse import urlsplit

from dotenv import load_dotenv
from jsonschema import Draft202012Validator, ValidationError

from evaluation.backends import (BM25Retriever, HTTPRetriever, GeminiClient, QA_PROMPT,
                                 JUDGE_PROMPT, JUDGE_SCHEMA, extractive_answer)
from evaluation.dataset import ROOT, DEFAULT_DATASET, load_dataset, sha256, json_sha256, validate_prediction
from evaluation.metrics import ABSTENTION, corpus_bleu, mean, retrieval_scores


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_predictions(path, cases, known_ids):
    predictions = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        prediction = json.loads(line)
        if prediction["id"] in predictions:
            raise ValueError(f"Duplicate prediction ID: {prediction['id']}")
        predictions[prediction["id"]] = prediction
    if set(predictions) != {case["id"] for case in cases}:
        raise ValueError("Prediction IDs must match the complete benchmark; no dropped cases")
    for case in cases:
        validate_prediction(predictions[case["id"]], case, known_ids)
    return predictions


def summarize(dataset, predictions, config):
    cases = dataset["cases"]
    answerable = [case for case in cases if case["answerable"]]
    negative = [case for case in cases if not case["answerable"]]
    rows = {pred["id"]: pred for pred in predictions}
    scores = {case["id"]: retrieval_scores(rows[case["id"]]["retrieved_doc_ids"],
                                         case["relevant_doc_ids"], config["k"]) for case in cases}
    generation_requested = config["generator"] != "none"
    bleu = corpus_bleu(
        [rows[case["id"]]["answer"] or "" for case in answerable],
        [case["reference_answers"] for case in answerable],
    ) if generation_requested else {"score": None, "signature": None, "scale": "0-1"}
    metrics = {
        "hit_rate_at_k": mean([scores[case["id"]]["hit"] for case in answerable]),
        "mrr_at_k": mean([scores[case["id"]]["reciprocal_rank"] for case in answerable]),
        "recall_at_k": mean([scores[case["id"]]["recall"] for case in answerable]),
        "corpus_bleu": bleu,
        "unsupported_abstention_accuracy": mean([
            float((rows[case["id"]]["answer"] or "").strip() == ABSTENTION) for case in negative
        ]) if generation_requested else None,
    }
    slices = {}
    for category in sorted({case["category"] for case in cases}):
        members = [case for case in cases if case["category"] == category and case["answerable"]]
        slices[category] = {"answerable_cases": len(members),
                            "hit_rate_at_k": mean([scores[case["id"]]["hit"] for case in members]),
                            "mrr_at_k": mean([scores[case["id"]]["reciprocal_rank"] for case in members])}
    judges = [pred["judge"] for pred in predictions if pred.get("judge")]
    times = [pred.get("latency_ms", {}).get("total", 0) for pred in predictions]
    errors = sum(bool(pred.get("error")) for pred in predictions)
    return {
        "schema_version": 1, "status": "complete" if not errors else "completed_with_errors",
        "dataset": {"id": dataset["dataset_id"], "version": dataset["dataset_version"],
                    "origin": dataset["origin"], "cases": len(cases), "answerable_cases": len(answerable),
                    "unsupported_cases": len(negative), "corpus": dataset["corpus"]},
        "configuration": config, "metrics": metrics, "slices": slices,
        "coverage": {"cases_scored": len(cases), "answers_present": sum(pred["answer"] is not None for pred in predictions),
                     "errors": errors, "judged_cases": len(judges)},
        "llm_judge": {
            "status": ("measured" if len(judges) == len(cases) else "partial") if judges else
                      "not_run" if not config.get("judge_model") else "failed",
            "model": config.get("judge_model"),
            "groundedness": mean([row["groundedness"] for row in judges]),
            "relevance": mean([row["relevance"] for row in judges]),
            "reference_coverage": mean([row["reference_coverage"] for row in judges]),
            "scale": "1-5", "coverage": len(judges) / len(cases),
        },
        "latency_ms": {"mean": mean(times), "median": statistics.median(times) if times else None},
        "failures": [
            {"id": case["id"], "question": case["question"],
             "first_relevant_rank": (1 / scores[case["id"]]["reciprocal_rank"]
                                     if scores[case["id"]]["reciprocal_rank"] else None),
             "error": rows[case["id"]].get("error")}
            for case in cases if (case["answerable"] and scores[case["id"]]["hit"] == 0)
            or rows[case["id"]].get("error")
        ],
        "unsupported_answer_failures": [
            {"id": case["id"], "question": case["question"], "answer": rows[case["id"]]["answer"]}
            for case in negative if generation_requested and
            (rows[case["id"]]["answer"] or "").strip() != ABSTENTION
        ],
    }


def check_regression(summary, baseline, max_drop):
    if max_drop < 0:
        raise ValueError("max_drop must be nonnegative")
    for key in ("id", "version", "corpus"):
        if summary["dataset"][key] != baseline["dataset"][key]:
            raise ValueError("Cannot compare runs on different dataset/corpus versions")
    for key in ("dataset_sha256", "k", "hash_format"):
        if summary["configuration"][key] != baseline["configuration"][key]:
            raise ValueError("Cannot compare runs with different labels or rank cutoffs")
    if summary["metrics"]["corpus_bleu"]["signature"] != baseline["metrics"]["corpus_bleu"]["signature"]:
        raise ValueError("Cannot compare different BLEU configurations")
    drops = []
    for key in ("hit_rate_at_k", "mrr_at_k", "recall_at_k", "unsupported_abstention_accuracy"):
        current, old = summary["metrics"][key], baseline["metrics"][key]
        if old is not None and (current is None or current < old - max_drop - 1e-12):
            drops.append(key)
    current, old = summary["metrics"]["corpus_bleu"]["score"], baseline["metrics"]["corpus_bleu"]["score"]
    if old is not None and (current is None or current < old - max_drop - 1e-12):
        drops.append("corpus_bleu")
    if summary["coverage"]["errors"] > baseline["coverage"]["errors"]:
        drops.append("errors")
    return {"passed": not drops, "max_absolute_drop": max_drop, "regressed_metrics": drops}


def value(number):
    return "not measured" if number is None else f"{number:.4f}"


def report_markdown(summary):
    data, config, metrics = summary["dataset"], summary["configuration"], summary["metrics"]
    lines = [
        "# RAG evaluation results", "",
        f"Measured run: {summary['run']['created_at_utc']}.",
        f"Dataset: **{data['version']}**, **{data['cases']} curated cases**, "
        f"including {data['answerable_cases']} answerable and {data['unsupported_cases']} unsupported questions.",
        "**These are authored benchmark cases, not 100 users and not a reproduction of the resume study.**", "",
        f"Retriever: **{config['retriever']}**. Answer generator: **{config['generator']}**. Cutoff: **{config['k']}**.",
        "The BM25/extractive baseline is separate from the deployed semantic/Gemini RAG and from the route-planning UI.", "",
        "| Metric | Measured result | Denominator / scale |", "| --- | ---: | --- |",
        f"| Hit@{config['k']} | {value(metrics['hit_rate_at_k'])} | {data['answerable_cases']} answerable cases |",
        f"| MRR@{config['k']} | {value(metrics['mrr_at_k'])} | {data['answerable_cases']} answerable cases |",
        f"| Recall@{config['k']} | {value(metrics['recall_at_k'])} | mean fraction of relevant documents found |",
        f"| Corpus BLEU-4 | {value(metrics['corpus_bleu']['score'])} | 0-1; answerable cases; failed answers count as empty |",
        f"| Unsupported-query abstention | {value(metrics['unsupported_abstention_accuracy'])} | {data['unsupported_cases']} unsupported cases |",
        f"| LLM judge | {summary['llm_judge']['status']} | {summary['coverage']['judged_cases']}/{data['cases']} cases |",
        "", f"BLEU signature: {metrics['corpus_bleu']['signature']}.",
        f"Errors: **{summary['coverage']['errors']}**. Answers present: **{summary['coverage']['answers_present']}**.",
        "", "## Retrieval slices", "", "| Category | Answerable cases | Hit@k | MRR@k |", "| --- | ---: | ---: | ---: |",
    ]
    for category, row in summary["slices"].items():
        lines.append(f"| {category} | {row['answerable_cases']} | {value(row['hit_rate_at_k'])} | {value(row['mrr_at_k'])} |")
    lines += ["", "## Observed retrieval misses / errors", ""]
    lines += [f"- {row['id']}: {row['question']}" for row in summary["failures"]] or ["None in this run."]
    lines += ["", "## Unsupported questions answered instead of abstained", ""]
    lines += [f"- {row['id']}: {row['question']} Answer: {row['answer']}"
              for row in summary["unsupported_answer_failures"]] or ["None in this run."]
    lines += [
        "", "## Interpretation", "",
        "- This small dataset reuses 10 attraction records and selected restaurant records; many questions name the place explicitly.",
        "- It is a regression/development benchmark, not a representative or independent test set or a user satisfaction study.",
        "- Relevance labels and references are curated from the included JSON, whose facts were not independently verified or refreshed.",
        "- BLEU measures reference-word overlap, not factuality, safety, or itinerary quality.",
        "- High retrieval scores do not mean the answer generator handles unsupported questions correctly.",
        "- LLM scores are only measured when an actual judge returns validated ratings; otherwise they stay unmeasured.",
        "- Token usage is recorded when Gemini supplies it; monetary cost is not estimated.",
        "", "See [methodology](../../README.md) and [per-case traces](traces.jsonl) for evidence.",
    ]
    return "\n".join(lines) + "\n"


def run(args):
    load_dotenv(ROOT / ".env")
    dataset, documents = load_dataset(args.dataset)
    by_id = {doc["doc_id"]: doc for doc in documents}
    if args.k < 1:
        raise ValueError("k must be positive")
    if args.output.resolve().exists() and any(args.output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new path to preserve previous runs")
    if args.retriever == "http":
        parsed = urlsplit(args.retriever_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Retriever URL must be http(s), without credentials, a query, or a fragment")
    config = {"retriever": args.retriever, "generator": args.generator, "k": args.k,
              "retriever_url": args.retriever_url if args.retriever == "http" else None,
              "model": args.model if args.generator == "gemini" else None,
              "judge_model": args.judge_model, "temperature": 0,
              "dataset_sha256": json_sha256(args.dataset),
              "hash_format": "compact-json-preserve-order-utf8-v1",
              "qa_prompt_sha256": digest(QA_PROMPT), "judge_prompt_sha256": digest(JUDGE_PROMPT)}
    replay = read_predictions(args.predictions, dataset["cases"], by_id) if args.predictions else None
    if replay is not None:
        config["retriever"] = "replay"
        config["generator"] = "replay" if args.generator != "none" else "none"
        config["predictions_sha256"] = sha256(args.predictions)
    retriever = None if replay is not None else (
        BM25Retriever(documents) if args.retriever == "bm25" else HTTPRetriever(args.retriever_url, documents))
    generator = GeminiClient(args.model) if args.generator == "gemini" and replay is None else None
    judge = GeminiClient(args.judge_model) if args.judge_model else None
    traces = []
    args.output.mkdir(parents=True, exist_ok=True)
    cases_by_id = {case["id"]: case for case in dataset["cases"]}

    def persist_trace(trace):
        row = {**trace, "metrics": retrieval_scores(trace["retrieved_doc_ids"],
               cases_by_id[trace["id"]]["relevant_doc_ids"], args.k)}
        with (args.output / "traces.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for case in dataset["cases"]:
        started = perf_counter()
        retrieval_valid = False
        trace = {"id": case["id"], "retrieved_doc_ids": [], "answer": None,
                 "error": None, "latency_ms": {}, "judge": None, "usage": {}, "steps": []}
        try:
            step = perf_counter()
            if replay is not None:
                trace.update(replay[case["id"]])
                retrieved = [by_id[doc_id] for doc_id in trace["retrieved_doc_ids"][:args.k]]
            else:
                retrieved = retriever.search(case["question"], args.k)
                trace["retrieved_doc_ids"] = [doc["doc_id"] for doc in retrieved]
            validate_prediction({key: trace[key] for key in ("id", "retrieved_doc_ids", "answer", "error")}, case, by_id)
            retrieval_valid = True
            trace["latency_ms"]["retrieval"] = (perf_counter() - step) * 1000
            trace["steps"].append({"name": "retrieval", "ranked_documents": [
                {"doc_id": doc["doc_id"], "score": doc.get("score")} for doc in retrieved]})
            context = [{"doc_id": doc["doc_id"], "text": doc["text"]} for doc in retrieved]
            step = perf_counter()
            if replay is None:
                if args.generator == "extractive":
                    trace["answer"] = extractive_answer(case["question"], retrieved)
                elif generator:
                    prompt = QA_PROMPT.format(payload=json.dumps({"question": case["question"], "records": context}, ensure_ascii=False))
                    trace["answer"], trace["usage"]["generation"] = generator.generate(prompt)
                    trace["steps"].append({"name": "generation", "model": args.model, "prompt_sha256": digest(prompt)})
            trace["latency_ms"]["generation"] = (perf_counter() - step) * 1000
            if judge and trace["answer"] is not None and not trace["error"]:
                step = perf_counter()
                prompt = JUDGE_PROMPT.format(payload=json.dumps({
                    "question": case["question"], "answerable": case["answerable"],
                    "answer": trace["answer"], "reference_answers": case["reference_answers"], "records": context,
                }, ensure_ascii=False))
                raw, trace["usage"]["judge"] = judge.generate(prompt, JUDGE_SCHEMA)
                rating = json.loads(raw)
                Draft202012Validator(JUDGE_SCHEMA).validate(rating)
                trace["judge"] = rating
                trace["latency_ms"]["judge"] = (perf_counter() - step) * 1000
                trace["steps"].append({"name": "judge", "model": args.judge_model, "prompt_sha256": digest(prompt)})
        except Exception as exc:
            if not retrieval_valid:
                trace["retrieved_doc_ids"] = []
            trace["error"] = str(exc) if isinstance(exc, ValueError) or (
                isinstance(exc, RuntimeError) and str(exc).startswith("Gemini")) else type(exc).__name__
        trace["latency_ms"]["total"] = (perf_counter() - started) * 1000
        traces.append(trace)
        persist_trace(trace)
        if trace["error"] and trace["error"].startswith(("Gemini HTTP 401", "Gemini HTTP 403", "Gemini HTTP 429")):
            for remaining in dataset["cases"][len(traces):]:
                traces.append({"id": remaining["id"], "retrieved_doc_ids": [], "answer": None,
                               "error": "Skipped after provider access/quota failure", "judge": None, "latency_ms": {}})
                persist_trace(traces[-1])
            break
    summary = summarize(dataset, traces, config)
    summary["run"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("sacrebleu", "jsonschema", "python-dotenv")},
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
                          for path in sorted(list((ROOT / "evaluation").glob("*.py")) +
                                             list((ROOT / "evaluation/schemas").glob("*.json")))} |
                         {name: sha256(ROOT / name) for name in
                          ("retrieval_documents.py", "pathway_service.py", "requirements-eval.txt")},
    }
    if args.baseline:
        summary["regression"] = check_regression(summary, json.loads(args.baseline.read_text(encoding="utf-8")), args.max_drop)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(report_markdown(summary), encoding="utf-8")
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps({key: trace[key] for key in ("id", "retrieved_doc_ids", "answer", "error")}, ensure_ascii=False) + "\n")
    print(f"Hit@{args.k}={value(summary['metrics']['hit_rate_at_k'])}; "
          f"MRR@{args.k}={value(summary['metrics']['mrr_at_k'])}; BLEU={value(summary['metrics']['corpus_bleu']['score'])}; "
          f"errors={summary['coverage']['errors']}; judge={summary['llm_judge']['status']}")
    return 1 if summary["coverage"]["errors"] or not summary.get("regression", {"passed": True})["passed"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--retriever", choices=["bm25", "http"], default="bm25")
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8766")
    parser.add_argument("--generator", choices=["extractive", "gemini", "none"], default="extractive")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--judge-model", help="Optional actual Gemini judge; omitted means not_run")
    parser.add_argument("--predictions", type=Path, help="Replay complete saved predictions JSONL")
    parser.add_argument("--baseline", type=Path, help="Fail on metric regression against a comparable run")
    parser.add_argument("--max-drop", type=float, default=0.02)
    try:
        return run(parser.parse_args())
    except (ValueError, OSError, ValidationError) as exc:
        parser.exit(2, f"Evaluation configuration error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
