import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set

from dotenv import load_dotenv
from pinecone import Pinecone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.embeddings import BGE_EMBEDDING_DIMENSIONS


DEFAULT_VALIDATION_SET = ROOT / "scripts" / "visual_anchor_validation_set.json"
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


def _validation_ids(path: Path) -> Set[str]:
    ids = set()
    if not path.exists():
        return ids
    for case in json.loads(path.read_text(encoding="utf-8")):
        figure_id = str(case.get("expected_figure_id") or "").strip()
        if figure_id:
            ids.add(figure_id.lower())
    return ids


def _is_gemini(metadata: Dict[str, object]) -> bool:
    status = str(metadata.get("vision_captioning_status") or "").lower()
    source = str(metadata.get("caption_source") or "").lower()
    return status == "success" or source == "gemini"


def main() -> int:
    load_dotenv()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Check Gemini caption coverage for visual validation subset.")
    parser.add_argument("--validation-set", type=Path, default=DEFAULT_VALIDATION_SET)
    parser.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", "bge_small_v1"))
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--go-threshold", type=float, default=0.60)
    parser.add_argument("--go-min-count", type=int, default=8)
    args = parser.parse_args()

    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    if not api_key or not index_name:
        raise SystemExit("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME.")

    validation_ids = _validation_ids(args.validation_set)
    index = Pinecone(api_key=api_key).Index(index_name)
    response = index.query(
        namespace=args.namespace,
        vector=_probe_vector(),
        top_k=args.limit,
        include_metadata=True,
        filter=VISUAL_FILTER,
    )
    validation_records = []
    for match in _matches(response):
        metadata = _metadata(match)
        figure_id = str(metadata.get("figure_id") or "").strip()
        if figure_id.lower() not in validation_ids:
            continue
        image_path = str(metadata.get("image_local_path") or metadata.get("image_path") or "").strip()
        validation_records.append(
            {
                "figure_id": figure_id,
                "page": metadata.get("source_page") or metadata.get("page") or "",
                "source_pdf": os.path.basename(str(metadata.get("source_pdf") or metadata.get("source_files") or metadata.get("source") or "")),
                "visual_type": metadata.get("visual_type") or "",
                "caption_source": metadata.get("caption_source") or "",
                "vision_captioning_status": metadata.get("vision_captioning_status") or "",
                "gemini_captioned": _is_gemini(metadata),
                "caption": str(metadata.get("caption") or "")[:220],
                "image_path": image_path,
                "image_path_exists": bool(image_path and Path(image_path).exists()),
            }
        )

    validation_records.sort(key=lambda item: (str(item["figure_id"]), int(item["page"]) if str(item["page"]).isdigit() else 0))
    gemini_count = sum(1 for item in validation_records if item["gemini_captioned"])
    total = len(validation_records)
    coverage = gemini_count / max(total, 1)
    remaining = [item for item in validation_records if not item["gemini_captioned"]]
    go = coverage >= args.go_threshold and gemini_count >= args.go_min_count
    summary = {
        "index": index_name,
        "namespace": args.namespace,
        "validation_set": str(args.validation_set),
        "validation_visuals_total": total,
        "gemini_captioned_validation_visuals": gemini_count,
        "coverage": round(coverage, 3),
        "go_threshold": args.go_threshold,
        "go_min_count": args.go_min_count,
        "remaining_uncaptioned_count": len(remaining),
        "remaining_uncaptioned_validation_visuals": remaining,
        "decision": "GO: validation caption coverage is sufficient for ranking tuning" if go else "NO-GO: continue targeted recaptioning",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
