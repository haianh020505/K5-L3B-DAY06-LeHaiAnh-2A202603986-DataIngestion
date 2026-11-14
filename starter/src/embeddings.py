"""Real embedding provider: OpenRouter's OpenAI-compatible /embeddings endpoint."""
import json
import os
import time
import urllib.error
import urllib.request
from typing import Protocol

OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class OpenRouterEmbedder:
    """Documents and queries are embedded identically (the model is symmetric).

    Any provider error raises; there is no fallback to mock vectors.
    """

    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing: set it in starter/.env")
        self.model_name = model_name
        self.batch_size = batch_size
        self.usage = {"requests": 0, "prompt_tokens": 0, "cost_usd_reported": 0.0}
        self.dimensions = len(self._post(["dimension probe"])[0])

    def _post(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model_name, "input": texts}).encode()
        req = urllib.request.Request(OPENROUTER_URL, data=body, headers={
            "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    payload = json.load(resp)
                break
            except urllib.error.HTTPError as exc:
                # retry only rate limits / transient server errors
                if exc.code not in (429, 500, 502, 503) or attempt == 3:
                    raise RuntimeError(f"OpenRouter HTTP {exc.code}: {exc.read()[:300]!r}") from exc
                time.sleep(2 ** attempt)
        if "data" not in payload:
            raise RuntimeError(f"OpenRouter returned no embeddings: {str(payload)[:300]}")
        usage = payload.get("usage") or {}
        self.usage["requests"] += 1
        self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        self.usage["cost_usd_reported"] += usage.get("cost", 0) or 0
        vectors = [d["embedding"] for d in sorted(payload["data"], key=lambda d: d["index"])]
        if len(vectors) != len(texts):
            raise RuntimeError(f"expected {len(texts)} vectors, got {len(vectors)}")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._post(texts[i:i + self.batch_size]))
        if any(len(v) != self.dimensions for v in out):
            raise RuntimeError("inconsistent embedding dimensions")
        return out

    def embed_query(self, question: str) -> list[float]:
        return self._post([question])[0]
