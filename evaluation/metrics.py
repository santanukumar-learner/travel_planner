"""Metric definitions are explicit about rank cutoffs and BLEU's scale."""

from sacrebleu.metrics import BLEU


ABSTENTION = "INSUFFICIENT_CONTEXT"


def retrieval_scores(ranked_ids, relevant_ids, k):
    if k < 1:
        raise ValueError("k must be positive")
    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("Duplicate retrieved document IDs would distort ranks")
    if not relevant_ids:
        return {"hit": None, "reciprocal_rank": None, "recall": None}
    relevant = set(relevant_ids)
    ranked = ranked_ids[:k]
    first = next((rank for rank, doc_id in enumerate(ranked, 1) if doc_id in relevant), None)
    return {
        "hit": float(first is not None),
        "reciprocal_rank": 1.0 / first if first else 0.0,
        "recall": len(set(ranked) & relevant) / len(relevant),
    }


def corpus_bleu(answers, references):
    if len(answers) != len(references):
        raise ValueError("One reference list is required per answer")
    if not answers:
        return {"score": None, "signature": None, "scale": "0-1"}
    if any(not refs for refs in references):
        raise ValueError("Every answer needs at least one reference")
    metric = BLEU(tokenize="13a", lowercase=True, smooth_method="exp", effective_order=True)
    # SacreBLEU expects reference streams, not per-case lists. None pads variable counts.
    streams = [[refs[index] if index < len(refs) else None for refs in references]
               for index in range(max(map(len, references)))]
    score = metric.corpus_score(answers, streams)
    return {"score": score.score / 100, "signature": str(metric.get_signature()), "scale": "0-1"}


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None
