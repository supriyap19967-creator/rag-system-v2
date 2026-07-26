import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from pinecone import Pinecone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.embeddings import BGE_EMBEDDING_DIMENSIONS


VISUAL_FILTER = {
    "$and": [
        {"source_type": {"$eq": "pdf"}},
        {"content_type": {"$eq": "visual"}},
    ]
}


def _probe_vector() -> List[float]:
    vector = [0.0] * BGE_EMBEDDING_DIMENSIONS
    vector[0] = 1.0
    return vector


def _metadata(match: object) -> Dict[str, object]:
    if isinstance(match, dict):
        return dict(match.get("metadata") or {})
    return dict(getattr(match, "metadata", {}) or {})


def _match_id(match: object) -> str:
    if isinstance(match, dict):
        return str(match.get("id") or "")
    return str(getattr(match, "id", "") or "")


def _figure_id(metadata: Dict[str, object]) -> str:
    explicit = str(metadata.get("figure_id") or "").strip()
    if explicit:
        return explicit
    text = " ".join(
        str(metadata.get(key) or "")
        for key in ("caption", "original_text", "visual_data", "nearby_text")
    )
    match = re.search(r"\b(Fig\.?|Figure|Table|Chart|Panel)\s+(\d+(?:\.\d+)?[A-Za-z]?)", text, re.IGNORECASE)
    if match:
        kind, number = match.groups()
        kind = "Figure" if kind.lower().startswith("fig") else kind.title()
        return f"{kind} {number}"
    visual_type = str(metadata.get("visual_type") or "visual").title()
    page = metadata.get("source_page") or metadata.get("page") or "unknown"
    return f"{visual_type} page {page}"


def _section_from_figure_id(figure_id: str) -> str:
    match = re.search(r"\b(?:Figure|Table|Chart|Panel)\s+(\d+)(?:\.\d+)?", figure_id or "", re.IGNORECASE)
    if match:
        return f"Chapter {match.group(1)}"
    return "Visual context"


def repair(index, namespace: str, limit: int) -> Dict[str, int]:
    response = index.query(
        namespace=namespace,
        vector=_probe_vector(),
        top_k=limit,
        include_metadata=True,
        filter=VISUAL_FILTER,
    )
    matches = list(getattr(response, "matches", []) or [])
    updated = 0
    skipped = 0
    for match in matches:
        metadata = _metadata(match)
        vector_id = _match_id(match)
        if not vector_id:
            skipped += 1
            continue
        figure_id = _figure_id(metadata)
        section = str(metadata.get("section") or metadata.get("section_header") or "").strip()
        patch = {}
        if not metadata.get("figure_id") and figure_id:
            patch["figure_id"] = figure_id
        if not section:
            patch["section"] = _section_from_figure_id(figure_id)
            patch["section_header"] = patch["section"]
        if not patch:
            skipped += 1
            continue
        index.update(id=vector_id, namespace=namespace, set_metadata=patch)
        updated += 1
    return {"matched": len(matches), "updated": updated, "skipped": skipped}


def main() -> int:
    load_dotenv()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Repair visual metadata in Pinecone without re-embedding.")
    parser.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", "bge_small_v1"))
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()

    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    if not api_key or not index_name:
        raise SystemExit("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME.")

    client = Pinecone(api_key=api_key)
    index = client.Index(index_name)
    result = repair(index, args.namespace, args.limit)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
