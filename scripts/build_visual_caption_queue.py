import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from pinecone import Pinecone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.embeddings import BGE_EMBEDDING_DIMENSIONS


DEFAULT_QUEUE_PATH = ROOT / "Data" / "visual_caption_queue.json"
DEFAULT_VALIDATION_SETS = [
    ROOT / "scripts" / "visual_anchor_validation_set.json",
    ROOT / "scripts" / "visual_validation_set.json",
]
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


def _matches(response: object) -> List[object]:
    if isinstance(response, dict):
        return list(response.get("matches") or [])
    return list(getattr(response, "matches", []) or [])


def _metadata(match: object) -> Dict[str, object]:
    if isinstance(match, dict):
        return dict(match.get("metadata") or {})
    return dict(getattr(match, "metadata", {}) or {})


def _match_id(match: object) -> str:
    if isinstance(match, dict):
        return str(match.get("id") or "")
    return str(getattr(match, "id", "") or "")


def _load_validation_refs(paths: List[Path]) -> tuple[Set[str], Set[Tuple[str, str]]]:
    ids = set()
    pairs = set()
    for path in paths:
        if not path.exists():
            continue
        for case in json.loads(path.read_text(encoding="utf-8")):
            figure_id = str(case.get("expected_figure_id") or "").strip()
            page = str(case.get("expected_page") or "").strip()
            if figure_id:
                normalized_id = figure_id.lower()
                ids.add(normalized_id)
                if page:
                    pairs.add((normalized_id, page))
    return ids, pairs


def _caption_quality_score(caption: str, generated_description: str) -> int:
    text = f"{caption} {generated_description}".strip()
    score = 100
    if len(caption.strip()) < 25:
        score -= 25
    if re.search(r"\b(Figure|Table)\s+\d+(?:\.\d+)?:\s*\)", caption, flags=re.IGNORECASE):
        score -= 30
    if "Figure 8. 4" in text or re.search(r"\b(Figure|Table)\s+\d+\.\s+\d+\b", text):
        score -= 15
    if text.lower().count("figure") > 3 or text.lower().count("table") > 3:
        score -= 10
    if any(fragment in text for fragment in ("Â", "Ã", "â€“", "qualÂ")):
        score -= 15
    if str(generated_description or "").strip() == str(caption or "").strip():
        score -= 15
    return max(score, 0)


def _priority_for(
    metadata: Dict[str, object],
    validation_ids: Set[str],
    validation_pairs: Set[Tuple[str, str]],
) -> tuple[int, List[str], int]:
    figure_id = str(metadata.get("figure_id") or "").strip()
    page = str(metadata.get("source_page") or metadata.get("page") or "").strip()
    caption = str(metadata.get("caption") or "")
    generated_description = str(metadata.get("generated_description") or "")
    status = str(metadata.get("vision_captioning_status") or metadata.get("caption_source") or "").lower()
    quality = _caption_quality_score(caption, generated_description)
    score = 0
    reasons = []
    if figure_id.lower() and (figure_id.lower(), page) in validation_pairs:
        score += 100
        reasons.append("exact_validation_figure_page")
    elif figure_id.lower() in validation_ids:
        score += 30
        reasons.append("appears_in_validation_set_adjacent_or_duplicate")
    if status not in {"success", "gemini"}:
        score += 35
        reasons.append("missing_or_fallback_caption")
    if quality < 70:
        score += 100 - quality
        reasons.append(f"poor_caption_quality:{quality}")
    if any(term in f"{caption} {generated_description}".lower() for term in ("firms", "vehicle", "emissions", "quality infrastructure", "standards adoption")):
        score += 15
        reasons.append("high_business_demo_value")
    if str(metadata.get("image_local_path") or metadata.get("image_path") or "").strip() and not Path(str(metadata.get("image_local_path") or metadata.get("image_path"))).exists():
        score -= 20
        reasons.append("image_path_missing_locally")
    return score, reasons, quality


def build_queue(index, namespace: str, limit: int, validation_sets: List[Path]) -> Dict[str, object]:
    response = index.query(
        namespace=namespace,
        vector=_probe_vector(),
        top_k=limit,
        include_metadata=True,
        filter=VISUAL_FILTER,
    )
    validation_ids, validation_pairs = _load_validation_refs(validation_sets)
    items = []
    for match in _matches(response):
        metadata = _metadata(match)
        priority_score, priority_reason, quality_score = _priority_for(metadata, validation_ids, validation_pairs)
        image_path = str(metadata.get("image_local_path") or metadata.get("image_path") or "").strip()
        item = {
            "vector_id": _match_id(match),
            "source_pdf": os.path.basename(str(metadata.get("source_pdf") or metadata.get("source_files") or metadata.get("source") or "")),
            "page": metadata.get("source_page") or metadata.get("page") or "",
            "figure_id": metadata.get("figure_id") or "",
            "visual_type": metadata.get("visual_type") or "",
            "image_path": image_path,
            "image_path_exists": bool(image_path and Path(image_path).exists()),
            "current_caption": metadata.get("caption") or "",
            "caption_source": metadata.get("caption_source") or "",
            "vision_captioning_status": metadata.get("vision_captioning_status") or "",
            "caption_quality_score": quality_score,
            "priority_score": priority_score,
            "priority_reason": priority_reason,
            "status": "pending",
        }
        if priority_score > 0:
            items.append(item)
    items.sort(key=lambda item: item["priority_score"], reverse=True)
    return {
        "queue_version": 1,
        "namespace": namespace,
        "total_candidates": len(_matches(response)),
        "queued_count": len(items),
        "validation_figure_ids": sorted(validation_ids),
        "validation_figure_page_pairs": sorted([f"{figure_id}|{page}" for figure_id, page in validation_pairs]),
        "items": items,
    }


def main() -> int:
    load_dotenv()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Build a prioritized queue of visual chunks for Gemini recaptioning.")
    parser.add_argument("--output", type=Path, default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", "bge_small_v1"))
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--validation-set", type=Path, action="append", default=None)
    args = parser.parse_args()

    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    if not api_key or not index_name:
        raise SystemExit("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME.")
    validation_sets = args.validation_set or DEFAULT_VALIDATION_SETS
    index = Pinecone(api_key=api_key).Index(index_name)
    queue = build_queue(index, args.namespace, args.limit, validation_sets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"CAPTION QUEUE SAVED: {args.output}")
    print(f"Queued: {queue['queued_count']} of {queue['total_candidates']} visual chunks")
    print("Top 10:")
    for item in queue["items"][:10]:
        print(
            f"- score={item['priority_score']} {item['figure_id']} page={item['page']} "
            f"quality={item['caption_quality_score']} reasons={', '.join(item['priority_reason'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
