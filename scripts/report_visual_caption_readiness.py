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
    parser = argparse.ArgumentParser(description="Report caption coverage and go/no-go recommendation for ranking tuning.")
    parser.add_argument("--validation-set", type=Path, default=DEFAULT_VALIDATION_SET)
    parser.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", "bge_small_v1"))
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--go-threshold", type=float, default=0.65)
    args = parser.parse_args()

    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    if not api_key or not index_name:
        raise SystemExit("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME.")
    index = Pinecone(api_key=api_key).Index(index_name)
    response = index.query(
        namespace=args.namespace,
        vector=_probe_vector(),
        top_k=args.limit,
        include_metadata=True,
        filter=VISUAL_FILTER,
    )
    validation_ids = _validation_ids(args.validation_set)
    all_metadata = [_metadata(match) for match in _matches(response)]
    total = len(all_metadata)
    gemini_total = sum(1 for metadata in all_metadata if _is_gemini(metadata))
    validation_records = [
        metadata
        for metadata in all_metadata
        if str(metadata.get("figure_id") or "").strip().lower() in validation_ids
    ]
    validation_total = len(validation_records)
    validation_gemini = sum(1 for metadata in validation_records if _is_gemini(metadata))
    overall_coverage = gemini_total / max(total, 1)
    validation_coverage = validation_gemini / max(validation_total, 1)
    go = validation_coverage >= args.go_threshold
    summary = {
        "index": index_name,
        "namespace": args.namespace,
        "overall_visual_caption_coverage": round(overall_coverage, 3),
        "overall_gemini_captioned": gemini_total,
        "overall_visual_chunks": total,
        "validation_caption_coverage": round(validation_coverage, 3),
        "validation_gemini_captioned": validation_gemini,
        "validation_visual_chunks": validation_total,
        "go_threshold": args.go_threshold,
        "recommendation": "GO: ranking threshold tuning is reasonable next." if go else "NO-GO: improve Gemini caption coverage before deeper ranking tuning.",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
