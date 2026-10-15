# RAG evaluation

This directory contains a runnable benchmark, saved measurements, per-query traces, and regression tests. It evaluates retrieval and answer text separately from the Streamlit route-planning UI.

## Saved results and provenance

See the [measured baseline report](results/bm25-extractive-v1/report.md), [machine-readable summary](results/bm25-extractive-v1/summary.json), and [all 100 traces](results/bm25-extractive-v1/traces.jsonl).

The baseline uses **BM25 retrieval and extractive answers**, not Gemini, FAISS, or the SentenceTransformer retriever. The dataset contains **100 authored cases, not 100 users**:

| Slice | Cases |
| --- | ---: |
| Named attraction/restaurant questions | 50 |
| Attraction paraphrases | 30 |
| Questions requiring two documents | 10 |
| Unsupported questions requiring abstention | 10 |

There are 90 answerable cases. All 70 source records are searchable: 10 attractions and 60 restaurant records. Questions reference 10 attractions and 10 selected restaurants. Several cases reuse the same records and many name their target explicitly, so near-perfect retrieval on this benchmark does not establish broad RAG quality.

Questions, references, relevance labels, and evidence are in [rourkela_v1.json](datasets/rourkela_v1.json). References were curated from the included JSON; source facts, ratings, hours, distances, and fees were not independently verified or refreshed. Use this as a development/regression benchmark, not as an independent held-out study.

## Relationship to the resume

The supplied resume reports a historical evaluation involving **100 users**, with **hit rate 0.80, MRR 0.70, BLEU 0.60**, and LLM-based review. The underlying user queries, labels, model outputs, rank cutoff, BLEU configuration, and reviewer records were not supplied.

Those figures are **historical self-reported claims, not reproduced results**. They are not inserted into the generated summary or used as pass thresholds. The curated benchmark is a new evaluation with a different population, configuration, and methodology. Do not present its numbers as validation of that user study.

LLM judging has runnable code but was **not run** for the saved baseline. The configured Gemini credential returned HTTP 403 during a model-list access check; see [access-check evidence](results/gemini_access_check.json). No LLM judge score is fabricated or substituted with a heuristic.

## Reproduce the offline baseline

From the project root:

~~~powershell
python -m venv .venv-eval
.\.venv-eval\Scripts\Activate.ps1
python -m pip install -r requirements-eval.txt
python -m pytest -q
python -m evaluation.run --output evaluation/results/my-baseline-run
~~~

The default run needs no API keys, model downloads, database, Docker, or running search server. Use a fresh output directory; existing results are preserved.

Each run writes:

- `summary.json`: scores, denominators, coverage, per-slice metrics, source hashes, dependency versions, and configuration.
- `report.md`: a readable table with limitations and observed failures.
- `traces.jsonl`: ranked document IDs/scores, answers, errors, step timings, available Gemini token usage, and validated judge ratings. Completed cases are flushed even if a later case is interrupted.
- `predictions.jsonl`: a minimal replay format for deterministic rescoring.

Latency is local wall-clock time per case, excluding initial corpus/index construction. Replay timings measure rescoring overhead, not original model inference. Gemini model-version metadata is kept when supplied; monetary costs are not estimated.

## Metric definitions

| Metric | Definition |
| --- | --- |
| Hit@k | Fraction of answerable queries with at least one relevant document in the first k results |
| MRR@k | Mean of 1/rank of the first relevant document, or zero if none occurs within k |
| Recall@k | Mean fraction of each answerable query's relevant documents retrieved within k |
| Corpus BLEU-4 | SacreBLEU over the 90 answerable reference/answer pairs; divided by 100 to report on a 0-1 scale |
| Unsupported-query abstention | Fraction of the 10 unsupported cases answered with exactly `INSUFFICIENT_CONTEXT` |
| LLM judge | Separate 1-5 ratings for groundedness, question relevance, and reference coverage |

BLEU uses tokenizer `13a`, lowercase text, exponential smoothing, and effective order. The exact SacreBLEU signature is saved in every summary. It is corpus BLEU, not an average of sentence scores. Failed answers contribute empty strings; failed cases are not silently dropped. Retrieval metrics exclude unsupported questions because they have no relevant documents; abstention evaluates those separately.

BLEU is a word-overlap signal and can penalize valid paraphrases. It does not establish factuality, safety, travel feasibility, or user satisfaction. LLM ratings also require review: temperature zero is not a guarantee of determinism, model aliases can change, and a judge can share biases with the answer model.

The initial baseline answered live queue-time questions with static temple facts. That observed failure is preserved in [a regression fixture](fixtures/unsupported_queue_failure.json) and checked by the tests. It is a curated-benchmark failure, not a claimed production incident.

## Evaluate the semantic retriever and Gemini

The existing search API now shares [document IDs and serialization](../retrieval_documents.py) with the evaluator. It still uses SentenceTransformer embeddings and NumPy similarity. The unused Pathway import was removed; this service does not implement a Pathway streaming graph.

Start it in another terminal, using port 8766 so it can coexist with the Doctor Dashboard prescription service on 8765:

~~~powershell
python -m pip install fastapi uvicorn sentence-transformers numpy
$env:PATHWAY_HOST = "127.0.0.1"
$env:PATHWAY_PORT = "8766"
python pathway_service.py
~~~

The first startup can download the embedding model. For the legacy planner in `rag.py`, set `PATHWAY_BASE_URL=http://127.0.0.1:8766`; the default remains port 8765 for compatibility.

Configure a working `GEMINI_API_KEY` in the ignored root `.env`, then use model IDs available to your account:

~~~powershell
python -m evaluation.run --retriever http --retriever-url http://127.0.0.1:8766 --generator gemini --model YOUR_AVAILABLE_MODEL --judge-model YOUR_JUDGE_MODEL --output evaluation/results/live-rag-run
~~~

This is a separate live **semantic retrieval + Gemini QA** configuration. Its concise, evidence-only QA prompt is versioned in [backends.py](backends.py); it is intentionally identified separately from the legacy itinerary prompt in `rag.py`. Neither live configuration has measured results in the checked-in offline baseline. Use `--generator none` for retrieval-only scoring, or omit `--judge-model` to avoid judge calls.

The generator sees only the question and retrieved records. Gold answers are supplied only to scoring and the judge. The judge returns JSON that is validated against a 1-5 rubric; invalid or failed judgments are errors, not scores. Access/quota failures stop further API calls and preserve the full benchmark denominator. Judge coverage and failures remain explicit.

## Score saved outputs from another RAG pipeline

Capture the original ranked document IDs and answer together; do not retrieve again later and assume those were the documents used by the original answer.

Each JSONL row must have:

~~~json
{"id":"rkl-001","retrieved_doc_ids":["rag.json:religious_1"],"answer":"The Hanuman statue at Hanuman Vatika is 75 feet tall.","error":null}
~~~

Provide exactly one row per benchmark case, including failures. Unknown IDs, duplicates, stale corpus content, and missing cases fail validation.

~~~powershell
python -m evaluation.run --predictions path/to/saved_predictions.jsonl --output evaluation/results/replayed-run
~~~

Replay calculates the same metrics without making generation requests. Add `--judge-model YOUR_JUDGE_MODEL` only if you want actual new judge calls.

## Catch regressions

~~~powershell
python -m evaluation.run --baseline evaluation/results/bm25-extractive-v1/summary.json --max-drop 0.02 --output evaluation/results/regression-check
~~~

The gate rejects drops greater than the configured absolute tolerance in Hit@k, MRR@k, Recall@k, BLEU, or unsupported-query abstention, and any increase in error count. The default 0.02 tolerance is an explicit engineering choice, not a statistically calibrated threshold. Exit status is nonzero for run errors or regressions.

Comparison requires the same dataset hash, corpus version, rank cutoff, and BLEU signature. Dataset JSON Schema plus corpus SHA-256 hashes prevent silently evaluating stale labels. JSON hashes use compact UTF-8 serialization while preserving record/property order, so Windows/Linux line endings and indentation do not break reproduction; changed facts or record order do. Source-code hashes record raw file bytes separately. To change source data or annotations, review evidence, increment the benchmark version, and establish a new baseline rather than editing an old report.

Offline regression checks cover the BM25/extractive configuration and evaluator. Prompt, embedding, or Gemini changes require a matching live run or recorded outputs; an offline baseline alone cannot detect their effects.

## Code map and sources

- [metrics.py](metrics.py): cutoff-aware retrieval metrics and corpus BLEU.
- [dataset.py](dataset.py), [schemas](schemas/dataset.schema.json): validation, corpus hashes, evidence checks, prediction contracts.
- [backends.py](backends.py): BM25/extractive baseline, actual HTTP search adapter, Gemini generation and structured judge rubric.
- [run.py](run.py): execution, streaming traces, reports, coverage, replay, regression gating.
- [tests](tests/test_evaluation.py): ranks, score scales, missing/stale data, provider errors, schema failures, regression detection, interrupted-run persistence.
- [SacreBLEU documentation](https://github.com/mjpost/sacrebleu): metric API and reproducibility signatures.
- [Gemini GenerateContent API](https://ai.google.dev/api/generate-content): request schema, structured JSON generation, response model version, and token usage.
