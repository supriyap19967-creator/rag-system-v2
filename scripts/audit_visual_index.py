import argparse
import json
import os
import sys
from collections import Counter
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


def _response_value(response: object, key: str, default: object = None) -> object:
    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)


def _matches(response: object) -> List[object]:
    if isinstance(response, dict):
        return list(response.get("matches") or [])
    return list(getattr(response, "matches", []) or [])


def _metadata(match: object) -> Dict[str, object]:
    if isinstance(match, dict):
        return dict(match.get("metadata") or {})
    return dict(getattr(match, "metadata", {}) or {})


def _figure_id(metadata: Dict[str, object]) -> str:
    import re

    explicit = str(metadata.get("figure_id") or "").strip()
    if explicit:
        return explicit
    caption = str(metadata.get("caption") or metadata.get("original_text") or metadata.get("text") or "")
    match = re.search(r"\b(Figure|Fig\.?|Table|Chart|Panel)\s+(\d+(?:\.\d+)?[A-Za-z]?)", caption, flags=re.IGNORECASE)
    if match:
        kind, number = match.groups()
        kind = "Figure" if kind.lower().startswith("fig") else kind.title()
        return f"{kind} {number}"
    visual_type = str(metadata.get("visual_type") or "visual").strip().lower() or "visual"
    page = metadata.get("source_page") or metadata.get("page") or "unknown"
    return f"{visual_type.title()} page {page}"


def _probe_vector() -> List[float]:
    vector = [0.0] * BGE_EMBEDDING_DIMENSIONS
    vector[0] = 1.0
    return vector


def audit(index, namespace: str, limit: int) -> Dict[str, object]:
    stats = index.describe_index_stats()
    response = index.query(
        namespace=namespace,
        vector=_probe_vector(),
        top_k=limit,
        include_metadata=True,
        filter=VISUAL_FILTER,
    )
    matches = _matches(response)
    visual_types = Counter()
    caption_statuses = Counter()
    unique_pages = set()
    unique_keys = set()
    unique_figure_ids = set()
    unique_sections = set()
    missing_caption = 0
    missing_image_path = 0
    missing_figure_id = 0
    missing_section = 0
    image_path_exists = 0
    examples = []

    for match in matches:
        metadata = _metadata(match)
        visual_type = str(metadata.get("visual_type") or "<missing>")
        visual_types[visual_type] += 1
        caption_statuses[str(metadata.get("vision_captioning_status") or metadata.get("caption_source") or "<missing>")] += 1
        source_pdf = os.path.basename(str(metadata.get("source_pdf") or metadata.get("source_files") or metadata.get("source") or ""))
        page = metadata.get("source_page") or metadata.get("page") or ""
        figure_id = _figure_id(metadata)
        section = str(metadata.get("section") or metadata.get("section_header") or "").strip()
        unique_keys.add((source_pdf, page, figure_id))
        if page:
            unique_pages.add(str(page))
        if figure_id:
            unique_figure_ids.add(figure_id)
        if section:
            unique_sections.add(section)
        caption = str(metadata.get("caption") or "").strip()
        image_path = str(metadata.get("image_local_path") or metadata.get("image_path") or "").strip()
        if not caption:
            missing_caption += 1
        if not str(metadata.get("figure_id") or "").strip():
            missing_figure_id += 1
        if not section:
            missing_section += 1
        if not image_path:
            missing_image_path += 1
        elif Path(image_path).exists():
            image_path_exists += 1
        if len(examples) < 10:
            examples.append(
                {
                    "figure_id": figure_id,
                    "source_pdf": source_pdf,
                    "page": page,
                    "visual_type": visual_type,
                    "section": section,
                    "caption": caption[:260],
                    "image_path": image_path,
                    "image_path_exists": bool(image_path and Path(image_path).exists()),
                }
            )

    status = "ok"
    notes = []
    if len(unique_keys) < 10:
        status = "incomplete"
        notes.append("Visual extraction/indexing coverage is incomplete.")
    if not matches:
        status = "empty"
        notes.append("No indexed visual chunks were found.")

    return {
        "namespace": namespace,
        "total_vector_count": _response_value(stats, "total_vector_count", None),
        "visual_chunks_sampled": len(matches),
        "query_limit": limit,
        "unique_visual_count": len(unique_keys),
        "unique_pages_covered": len(unique_pages),
        "unique_figure_table_ids_covered": len(unique_figure_ids),
        "unique_sections_covered": len(unique_sections),
        "sample_pages": sorted(unique_pages, key=lambda value: int(value) if value.isdigit() else value)[:30],
        "sample_figure_table_ids": sorted(unique_figure_ids)[:30],
        "visual_types_count": dict(sorted(visual_types.items())),
        "vision_captioning_status_count": dict(sorted(caption_statuses.items())),
        "missing_caption_count": missing_caption,
        "missing_figure_id_count": missing_figure_id,
        "missing_section_count": missing_section,
        "missing_image_path_count": missing_image_path,
        "image_path_exists_count": image_path_exists,
        "examples": examples,
        "coverage_status": status,
        "notes": notes,
    }


def main() -> int:
    load_dotenv()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Audit indexed PDF visual coverage in Pinecone.")
    parser.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", "bge_small_v1"))
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()

    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    if not api_key or not index_name:
        raise SystemExit("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME.")

    client = Pinecone(api_key=api_key)
    index = client.Index(index_name)
    summary = audit(index, args.namespace, args.limit)
    summary["index"] = index_name
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["coverage_status"] != "ok":
        print("VISUAL_INDEX_STATUS: Visual extraction/indexing coverage is incomplete.")
        return 2
    print("VISUAL_INDEX_STATUS: Visual index coverage looks usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
