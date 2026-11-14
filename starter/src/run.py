"""Day 06 pipeline: ingest -> chunk (A/B) -> embed -> Chroma -> retrieve A/B/C -> traces + metrics.

Run from starter/:  python -m src.run
"""
import csv
import json
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

from .chunking import by_paragraph, fixed_size
from .embeddings import OpenRouterEmbedder
from .evaluation import evaluate_query, norm, summarize
from .ingestion import load_pdf, normalize_text
from .store import ChromaStore

STARTER = Path(__file__).resolve().parents[1]
ROOT = STARTER.parent
RESULTS = STARTER / "results"
STRATEGIES = {
    "A": {"chunking": "fixed_size", "chunk_size": 500, "overlap": 50, "index": "A", "use_filter": False,
          "hypothesis": "Baseline: fixed windows may cut evidence sentences and mix topics."},
    "B": {"chunking": "by_paragraph", "chunk_size": 800, "overlap": 0, "index": "B", "use_filter": False,
          "hypothesis": "Paragraph/heading chunks keep evidence sentences whole -> higher Evidence Hit@3 and coherence than A."},
    "C": {"chunking": "by_paragraph", "chunk_size": 800, "overlap": 0, "index": "B", "use_filter": True,
          "hypothesis": "Same index as B plus chapter filter (only when the question supplies one) -> fewer cross-chapter confusions."},
}


def section_layout(reader: PdfReader, pages: list[int]) -> dict[int, dict]:
    """Chapter and section starts from the PDF outline (bookmarks), keyed by physical page."""
    sections = []  # (start_page, chapter, title)
    for i, item in enumerate(reader.outline):
        if isinstance(item, list) or not item.title.startswith("Chapter "):
            continue
        chapter = int(item.title.split()[1])
        sections.append((reader.get_destination_page_number(item) + 1, chapter, f"{chapter}.0 Introduction"))
        children = reader.outline[i + 1] if i + 1 < len(reader.outline) and isinstance(reader.outline[i + 1], list) else []
        for child in children:
            if child.title[:1].isdigit():
                sections.append((reader.get_destination_page_number(child) + 1, chapter, normalize_text(child.title)))
    sections.sort()
    layout = {}
    for p in pages:
        before = [s for s in sections if s[0] <= p]
        intro = next((s[2] for s in before if s[0] == p and s[2].endswith("Introduction")), None)
        layout[p] = {"chapter": before[-1][1],
                     "section_at_start": intro or next(s[2] for s in reversed(before) if s[0] < p),
                     "starting_here": [s[2] for s in sections if s[0] == p and not s[2].endswith("Introduction")]}
    return layout


def add_section_markers(page, info: dict) -> None:
    # use the LAST occurrence: the chapter-outline list on the first page repeats the titles
    # pad with newlines so a heading on the page's first/last line is found too (bug found in metadata audit)
    padded, markers = "\n" + page.text + "\n", []
    for title in info["starting_here"]:
        idx = padded.rfind("\n" + title + "\n")
        if idx >= 0:
            markers.append((idx, title))  # idx in padded == line start in page.text
    page.metadata.update(chapter=info["chapter"], section_at_start=info["section_at_start"],
                         section_markers=sorted(markers))


def validate_benchmark(bench: dict, pages_by_no: dict) -> list[str]:
    qs, problems = bench["questions"], []
    dev = [q for q in qs if q["split"] == "dev"]
    test = [q for q in qs if q["split"] == "test"]
    filt = [q for q in qs if q.get("filter")]
    if len(qs) < 20 or len(dev) < 15 or len(test) < 5:
        problems.append(f"need >=20 questions with >=15 dev / >=5 test, got {len(dev)}/{len(test)}")
    if len(filt) < 3 or not any(q["split"] == "test" for q in filt):
        problems.append("need >=3 filter questions with >=1 in test")
    for q in qs:
        for g in [{"page": q["gold_page"], "evidence_quote": q["evidence_quote"]}, *q.get("alt_gold", [])]:
            page = pages_by_no.get(g["page"])
            if page is None or norm(g["evidence_quote"]) not in norm(page.text):
                problems.append(f"{q['id']}: evidence not found on page {g['page']}")
    return problems


def length_stats(chunks) -> dict:
    lens = [len(c.text) for c in chunks]
    return {"chunks": len(lens), "total_chars": sum(lens), "min": min(lens), "median": statistics.median(lens),
            "mean": round(statistics.mean(lens), 1), "max": max(lens)}


def covering_chunk(chunks, page: int, offset: int):
    return next((c for c in chunks if c.page == page and
                 c.metadata["char_start"] <= offset < c.metadata["char_start"] + len(c.text)), None)


def write_audit(path: Path, rows: list[dict]) -> None:
    """Never overwrite an existing audit file: it may hold manual labels."""
    write_csv(path.with_suffix(".new.csv") if path.exists() else path, rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    load_dotenv(STARTER / ".env")
    RESULTS.mkdir(exist_ok=True)
    src = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))["sources"][0]
    pdf_path = ROOT / src["local_path"]
    pages_wanted = [p for ch in src["chapters"].values() for p in range(ch["pages"][0], ch["pages"][1] + 1)]

    # 1. ingest + validate corpus and benchmark
    t0 = time.perf_counter()
    pages, errors = load_pdf(pdf_path, src["source_id"], src["local_path"], pages_wanted)
    layout = section_layout(PdfReader(pdf_path), pages_wanted)
    for page in pages:
        add_section_markers(page, layout[page.page])
    ingest_s = time.perf_counter() - t0
    if len(pages) < 30:
        raise SystemExit(f"only {len(pages)} content pages; need >= 30")
    write_csv(ROOT / "data/manifest_pages.csv", [
        {"source_id": p.document_id, "page": p.page, "chapter": p.metadata["chapter"],
         "section_at_start": p.metadata["section_at_start"], "raw_chars": p.metadata["raw_chars"],
         "normalized_chars": len(p.text)} for p in pages])
    bench = json.loads((ROOT / "benchmark/questions.json").read_text(encoding="utf-8"))
    problems = validate_benchmark(bench, {p.page: p for p in pages})
    if problems:
        raise SystemExit("benchmark invalid:\n" + "\n".join(problems))

    # 2. chunk, embed, store (C reuses B's index; it differs only by the query-time filter)
    model = os.getenv("EMBEDDING_MODEL") or "openai/text-embedding-3-small"
    embedder = OpenRouterEmbedder(model)
    store = ChromaStore(STARTER / "chroma_db", embedder.dimensions)
    chunk_sets = {"A": [c for p in pages for c in fixed_size(p, 500, 50)],
                  "B": [c for p in pages for c in by_paragraph(p, 800)]}
    index_log = {}
    for name, chunks in chunk_sets.items():
        t0, tok0 = time.perf_counter(), embedder.usage["prompt_tokens"]
        vectors = embedder.embed_documents([c.text for c in chunks])
        first = store.upsert(name, chunks, vectors)
        elapsed = time.perf_counter() - t0
        again = store.upsert(name, chunks, vectors)  # re-ingest same version: count must not grow
        if again["count_after"] != len(chunks):
            raise SystemExit(f"re-ingest changed count for {name}: {again}")
        index_log[name] = {**length_stats(chunks), "embed_tokens": embedder.usage["prompt_tokens"] - tok0,
                           "index_seconds": round(elapsed, 2), "first_upsert": first, "reingest": again}

    # 3. retrieve top-3 for every question and strategy
    traces = []
    for q in bench["questions"]:
        qvec = embedder.embed_query(q["question"])
        for name, cfg in STRATEGIES.items():
            filters = q.get("filter") if cfg["use_filter"] else None
            hits = store.search(cfg["index"], qvec, 3, filters)
            traces.append({"strategy": name, "id": q["id"], "split": q["split"], "type": q["type"],
                           "question": q["question"], "filter_applied": filters,
                           "gold_page": q["gold_page"], "evidence_quote": q["evidence_quote"],
                           "hits": hits, "eval": evaluate_query(q, hits)})
    with (RESULTS / "traces.jsonl").open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    summary = summarize(traces)

    # 4. run log, experiment log, manual-audit templates (labels left blank for the human grader)
    run_log = {"model": model, "provider": "OpenRouter", "dimensions": embedder.dimensions,
               "vector_store": "chromadb " + __import__("chromadb").__version__ + " PersistentClient, cosine, HNSW (approximate)",
               "pages_loaded": len(pages), "page_errors": errors, "ingest_seconds": round(ingest_s, 2),
               "index": index_log, "embedding_usage": embedder.usage, "summary": summary,
               "run_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (RESULTS / "run_log.json").write_text(json.dumps(run_log, indent=2, ensure_ascii=False), encoding="utf-8")

    log_rows = []
    for name, cfg in STRATEGIES.items():
        s = summary[name]
        log_rows.append({"strategy": name, "chunking": cfg["chunking"], "chunk_size": cfg["chunk_size"],
                         "overlap": cfg["overlap"], "metadata": "document_id,source,page,chapter,section,char_start",
                         "filter": "chapter (only if question has filter)" if cfg["use_filter"] else "none",
                         "vectorizer": f"{model} via OpenRouter ({embedder.dimensions}d)",
                         "hypothesis": cfg["hypothesis"],
                         "dev_hit1": f"{s['dev']['hit@1']}/{s['dev']['n']}", "dev_hit3": f"{s['dev']['hit@3']}/{s['dev']['n']}",
                         "test_hit1": f"{s['test']['hit@1']}/{s['test']['n']}", "test_hit3": f"{s['test']['hit@3']}/{s['test']['n']}",
                         "notes": f"{index_log[cfg['index']]['chunks']} chunks in index {cfg['index']}"})
    write_csv(ROOT / "artifacts/experiment_log.csv", log_rows)

    audit = []
    for t in traces:
        if t["split"] == "test":
            for h in t["hits"]:
                m = h["metadata"]
                audit.append({"strategy": t["strategy"], "id": t["id"], "filter": json.dumps(t["filter_applied"]),
                              "rank": h["rank"], "chunk_id": h["chunk_id"], "page": m["page"], "chapter": m["chapter"],
                              "section": m["section"], "auto_evidence_hit": h["rank"] in t["eval"]["hit_ranks"],
                              "text": h["text"], "metadata_correct_0_1": "", "filter_respected_0_1": "", "note": ""})
    write_audit(RESULTS / "audit_metadata_test.csv", audit)
    write_audit(RESULTS / "audit_grounding_test.csv", [
        {"strategy": t["strategy"], "id": t["id"], "question": t["question"],
         "top3_chunk_ids": " | ".join(h["chunk_id"] for h in t["hits"]),
         "auto_evidence_hit@3": t["eval"]["hit@3"], "grounding_0_1_2": "", "reason": ""}
        for t in traces if t["split"] == "test"])

    # coherence sample: middle of every 7th content page, fixed before any labels exist
    positions = [(p.page, len(p.text) // 2) for p in pages[::7][:10]]
    coh = []
    for name in ("A", "B"):
        for page_no, off in positions:
            c = covering_chunk(chunk_sets[name], page_no, off)
            coh.append({"strategy": name + (" (=C)" if name == "B" else ""), "page": page_no, "offset": off,
                        "chunk_id": c.chunk_id, "text": c.text, "coherence_0_1_2": "", "reason": ""})
    write_audit(RESULTS / "audit_coherence.csv", coh)

    print(json.dumps({k: v for k, v in run_log.items() if k in ("model", "dimensions", "pages_loaded", "summary")}, indent=2))
    print({k: (v["chunks"], v["index_seconds"]) for k, v in index_log.items()}, embedder.usage)


if __name__ == "__main__":
    main()
