# pathway_retriever_client.py
import requests

class PathwayRetrieverClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8765"):
        self.base_url = base_url.rstrip("/")

    def search(self, query: str, k: int = 5):
        r = requests.post(
            f"{self.base_url}/search",
            json={"query": query, "k": k},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["results"]  # list of {text, metadata, score}
