import argparse
import json
import os
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


def _caption_source(metadata: Dict[str, object]) -> str:
    explicit = str(metadata.get("vision_captioning_status") or metadata.get("caption_source") or "").strip()
    if explicit:
        return explicit
    generated = str(metadata.get("generated_description") or "").strip()
    if not generated:
        return "missing"
    # Older indexes did not store whether Gemini succeeded. Treat them as
    # unknown/fallback so caption quality is judged conservatively.
    return "unknown_or_fallback"


def spot_check(index, namespace: str, sample_size: int, fetch_limit: int) -> Dict[str, object]:
    response = index.query(
        namespace=namespace,
        vector=_probe_vector(),
        top_k=fetch_limit,
        include_metadata=True,
        filter=VISUAL_FILTER,
    )
    matches = list(getattr(response, "matches", []) or [])
    by_page: Dict[str, Dict[str, object]] = {}
    for match in matches:
        metadata = _metadata(match)
        page = str(metadata.get("source_page") or metadata.get("page") or "")
        if page and page not in by_page:
            by_page[page] = metadata
        if len(by_page) >= sample_size:
            break

    samples = []
    weak_count = 0
    fallback_like_count = 0
    for metadata in by_page.values():
        image_path = str(metadata.get("image_local_path") or metadata.get("image_path") or "").strip()
        caption = str(metadata.get("caption") or "").strip()
        generated = str(metadata.get("generated_description") or "").strip()
        nearby = str(metadata.get("nearby_text") or "").strip()
        source = _caption_source(metadata)
        if source != "success":
            fallback_like_count += 1
        if len(caption.split()) < 6 or len(generated.split()) < 8:
            weak_count += 1
        samples.append(
            {
                "source_pdf": os.path.basename(str(metadata.get("source_pdf") or metadata.get("source_files") or metadata.get("source") or "")),
                "page": metadata.get("source_page") or metadata.get("page") or "",
                "figure_id": metadata.get("figure_id") or "",
                "visual_type": metadata.get("visual_type") or "",
                "extracted_caption": caption,
                "vision_summary": generated,
                "caption_source_guess": source,
                "nearby_paragraph_snippet": nearby[:320],
                "image_path": image_path,
                "image_path_exists": bool(image_path and Path(image_path).exists()),
            }
        )

    return {
        "namespace": namespace,
        "sample_size": len(samples),
        "weak_caption_count": weak_count,
        "fallback_like_count": fallback_like_count,
        "semantic_visual_retrieval_may_be_weaker": fallback_like_count > len(samples) // 2,
        "samples": samples,
    }


def main() -> int:
    load_dotenv()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Spot-check indexed visual captions and image paths.")
    parser.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", "bge_small_v1"))
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--fetch-limit", type=int, default=10000)
    args = parser.parse_args()

    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    if not api_key or not index_name:
        raise SystemExit("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME.")

    client = Pinecone(api_key=api_key)
    index = client.Index(index_name)
    result = spot_check(index, args.namespace, args.sample_size, args.fetch_limit)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["semantic_visual_retrieval_may_be_weaker"]:
        print("CAPTION_QUALITY_STATUS: fallback-heavy captions; semantic visual retrieval may be weaker.")
    else:
        print("CAPTION_QUALITY_STATUS: caption samples look usable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
