# RAG evaluation results

Measured run: 2026-10-15T06:48:25.719964+00:00.
Dataset: **1.0.0**, **100 curated cases**, including 90 answerable and 10 unsupported questions.
**These are authored benchmark cases, not 100 users and not a reproduction of the resume study.**

Retriever: **bm25**. Answer generator: **extractive**. Cutoff: **5**.
The BM25/extractive baseline is separate from the deployed semantic/Gemini RAG and from the route-planning UI.

| Metric | Measured result | Denominator / scale |
| --- | ---: | --- |
| Hit@5 | 1.0000 | 90 answerable cases |
| MRR@5 | 0.9944 | 90 answerable cases |
| Recall@5 | 1.0000 | mean fraction of relevant documents found |
| Corpus BLEU-4 | 0.3688 | 0-1; answerable cases; failed answers count as empty |
| Unsupported-query abstention | 0.0000 | 10 unsupported cases |
| LLM judge | not_run | 0/100 cases |

BLEU signature: nrefs:1|case:lc|eff:yes|tok:13a|smooth:exp|version:2.6.0.
Errors: **0**. Answers present: **100**.

## Retrieval slices

| Category | Answerable cases | Hit@k | MRR@k |
| --- | ---: | ---: | ---: |
| multi_document | 10 | 1.0000 | 1.0000 |
| named_entity | 50 | 1.0000 | 1.0000 |
| paraphrase | 30 | 1.0000 | 0.9833 |
| unsupported | 0 | not measured | not measured |

## Observed retrieval misses / errors

None in this run.

## Unsupported questions answered instead of abstained

- rkl-091: What is today's live queue time at Hanuman Vatika? Answer: Hanuman Vatika is a popular temple complex in Rourkela featuring one of the tallest statues of Hanuman at 75 feet.
- rkl-092: What is the current weather forecast for Pitamahal Dam tomorrow? Answer: Pitamahal Dam, about 22 km from Rourkela, is a picturesque reservoir surrounded by hills and forests.
- rkl-093: What is the latest 2026 crime rate at Koel Riverbank? Answer: Koel Riverbank is a peaceful riverside spot near Rourkela, suitable for relaxation, picnics, and evening walks.
- rkl-094: How much is the current boating ticket at Mandira Dam? Answer: Mandira Dam is a scenic reservoir near Kansbahal’s foothills, constructed to supply water to Rourkela Steel Plant.
- rkl-095: What is the phone number of Hotel Shubham? Answer: Hotel Shubham is at Railway Station, Shubham Lane, Rourkela.
- rkl-096: What time does Pizza Den close tonight? Answer: Pizza Den is at Madhusudan Marg, Station Road, Rourkela.
- rkl-097: Which rooms are available at Hotel Radhika Regency right now? Answer: Hotel Radhika Regency is at near Vijaya Bank, Bisra Road, Udit Nagar, Rourkela.
- rkl-098: What is the live train fare from Delhi to Rourkela? Answer: Indira Gandhi Park is one of the largest public parks in Rourkela.
- rkl-099: What is the Wi-Fi password at Curry Pot? Answer: Curry Pot is at Infront of Hotel Mayfair, H 1, Panposh Road, Civil Township, Rourkela.
- rkl-100: Which attractions in Delhi are described in this Rourkela knowledge base? Answer: Indian fast food is at Laxmi Market, near Community Centre, Sector 17, Rourkela 3, Sector-17, Rourkela.

## Interpretation

- This small dataset reuses 10 attraction records and selected restaurant records; many questions name the place explicitly.
- It is a regression/development benchmark, not a representative or independent test set or a user satisfaction study.
- Relevance labels and references are curated from the included JSON, whose facts were not independently verified or refreshed.
- BLEU measures reference-word overlap, not factuality, safety, or itinerary quality.
- High retrieval scores do not mean the answer generator handles unsupported questions correctly.
- LLM scores are only measured when an actual judge returns validated ratings; otherwise they stay unmeasured.
- Token usage is recorded when Gemini supplies it; monetary cost is not estimated.

See [methodology](../../README.md) and [per-case traces](traces.jsonl) for evidence.
