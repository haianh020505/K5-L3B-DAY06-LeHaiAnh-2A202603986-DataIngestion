"""Evidence-based Hit@1 / Hit@3 with per-query traces. Manual labels are NOT inferred here."""
import re
from typing import Any


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def gold_list(question: dict[str, Any]) -> list[dict[str, Any]]:
    """Primary gold plus any documented alternative gold evidence."""
    primary = {"source_id": question["gold_source_id"], "page": question["gold_page"],
               "evidence_quote": question["evidence_quote"]}
    return [primary, *question.get("alt_gold", [])]


def is_hit(hit: dict[str, Any], golds: list[dict[str, Any]]) -> bool:
    meta = hit["metadata"]
    return any(meta["document_id"] == g["source_id"] and meta["page"] == g["page"]
               and norm(g["evidence_quote"]) in norm(hit["text"]) for g in golds)


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    golds = gold_list(question)
    flags = [is_hit(h, golds) for h in hits]
    page_match = [any(h["metadata"]["page"] == g["page"] for g in golds) for h in hits]
    return {
        "hit@1": bool(flags[:1] and flags[0]),
        "hit@3": any(flags[:3]),
        "hit_ranks": [i + 1 for i, f in enumerate(flags) if f],
        # gold page retrieved but quote not fully inside one chunk => evidence split / truncated
        "gold_page_without_full_evidence": any(p and not f for p, f in zip(page_match, flags)),
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for t in traces:
        cell = out.setdefault(t["strategy"], {}).setdefault(t["split"], {"n": 0, "hit@1": 0, "hit@3": 0})
        cell["n"] += 1
        cell["hit@1"] += t["eval"]["hit@1"]
        cell["hit@3"] += t["eval"]["hit@3"]
    return out
