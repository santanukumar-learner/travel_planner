# Rourkela Travel Planner

Run the local Streamlit planner from this folder:

```powershell
python -m pip install -r requirements-streamlit.txt
python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501). Select a date, starting place, radius, hours, and optional stops. The app creates a route, shows the stops on a map, and can export the itinerary as JSON. No coordinates need to be entered.

If Windows shows the Microsoft Store "Python was not found" message in an older VS Code terminal, open a new terminal or run `& "$env:LOCALAPPDATA\Python\bin\python.exe" -m streamlit run streamlit_app.py`. You can also use `powershell -ExecutionPolicy Bypass -File .\run_streamlit.ps1` from this folder.

This local version uses `rag.json` and needs no API keys, Docker, or Pathway. Travel duration is an estimate based on 20 km/h with one hour at each stop. Places without coordinates cannot be routed. The older Gemini, weather, and crowd prediction modules remain separate and are not used by this Streamlit app.

## RAG evaluation

The repository now includes [evaluation code and methodology](evaluation/README.md), a [versioned 100-case curated benchmark](evaluation/datasets/rourkela_v1.json), and [measured results with per-query traces](evaluation/results/bm25-extractive-v1/report.md).

**Measured configuration: BM25 retrieval + extractive answers, k=5.** This offline baseline is separate from the semantic/Gemini RAG and from the key-free Streamlit route planner.

| Metric | Measured baseline result |
| --- | ---: |
| Hit@5 | 1.0000 |
| MRR@5 | 0.9944 |
| Recall@5 | 1.0000 |
| Corpus BLEU-4, 0-1 scale | 0.3688 |
| Unsupported-query abstention | 0.0000 |
| LLM judging | Not run; configured Gemini credential returned HTTP 403 |

Retrieval/BLEU use 90 answerable questions; abstention uses 10 unsupported questions. These are **100 authored cases, not 100 users**. Many cases name the relevant place, and references reuse a small corpus, so these scores are development measurements rather than evidence of broad user-facing quality. The baseline fails to abstain on unsupported questions; that observed failure is captured in a regression fixture.

Reproduce the evaluation without API keys:

~~~powershell
python -m venv .venv-eval
.\.venv-eval\Scripts\python.exe -m pip install -r requirements-eval.txt
.\.venv-eval\Scripts\python.exe -m pytest -q
.\.venv-eval\Scripts\python.exe -m evaluation.run --output evaluation/results/local-run
~~~

The evaluator also supports the actual HTTP semantic retriever, Gemini answers, schema-validated LLM judging, saved-output replay, and regression gates. Live LLM results are not claimed for this offline run. See the [evaluation guide](evaluation/README.md) for commands, metric definitions, corpus hashes, limitations, and the distinction from historical resume figures.

## Render

`render.yaml` defines a free Python web service. Put this folder's app files in the root of a Git repository, connect the repository to Render, and create a Blueprint from it. Render will install `requirements-streamlit.txt`, start Streamlit on its assigned port, and check `/_stcore/health`. The old `Requirements.txt` and Docker files are not used by this service.
