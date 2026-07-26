import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv
from langchain_core.documents import Document
from pinecone import Pinecone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.embeddings import get_bge_embeddings
from app.pdf_visual_extraction import _caption_image_with_gemini


DEFAULT_QUEUE_PATH = ROOT / "Data" / "visual_caption_queue.json"
DEFAULT_UPDATES_PATH = ROOT / "Data" / "visual_caption_updates.jsonl"
DEFAULT_BM25_PATH = ROOT / "Data" / "bm25_documents.json"
DEFAULT_VALIDATION_SET = ROOT / "scripts" / "visual_anchor_validation_set.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _response_value(response: object, key: str, default: object = None) -> object:
    if isinstance(response, dict):
        return response.get(key, default)
    return getattr(response, key, default)


def _fetch_metadata(index, namespace: str, vector_id: str) -> Dict[str, object]:
    response = index.fetch(ids=[vector_id], namespace=namespace)
    vectors = _response_value(response, "vectors", {}) or {}
    vector = vectors.get(vector_id) if isinstance(vectors, dict) else None
    if not vector:
        return {}
    metadata = _response_value(vector, "metadata", {}) or {}
    return dict(metadata)


def _combo_text(metadata: Dict[str, object], caption: str, vision_summary: str) -> str:
    previous_text = str(metadata.get("previous_text") or "").strip()
    next_text = str(metadata.get("next_text") or "").strip()
    visual_data = " ".join(part for part in (caption.strip(), vision_summary.strip()) if part)
    return f"[CONTEXT BEFORE]: {previous_text} | [VISUAL DATA]: {visual_data} | [CONTEXT AFTER]: {next_text}"


def _updated_metadata(metadata: Dict[str, object], vision_summary: str) -> tuple[str, Dict[str, object]]:
    caption = str(metadata.get("caption") or "").strip()
    page_content = _combo_text(metadata, caption, vision_summary)
    updated = dict(metadata)
    updated.update(
        {
            "original_text": page_content,
            "text": page_content,
            "visual_data": " ".join(part for part in (caption, vision_summary) if part),
            "generated_description": vision_summary,
            "vision_summary": vision_summary,
            "vision_captioning_status": "success",
            "caption_source": "gemini",
            "recaptioned_at": _now(),
        }
    )
    return page_content, updated


def _update_bm25_cache(path: Path, metadata: Dict[str, object], page_content: str) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    documents = payload.get("documents") if isinstance(payload, dict) else None
    if not isinstance(documents, list):
        return False
    image_path = str(metadata.get("image_local_path") or metadata.get("image_path") or "")
    figure_id = str(metadata.get("figure_id") or "")
    page = str(metadata.get("source_page") or metadata.get("page") or "")
    changed = False
    for item in documents:
        item_metadata = item.get("metadata") if isinstance(item, dict) else None
        if not isinstance(item_metadata, dict):
            continue
        same_image = image_path and image_path == str(item_metadata.get("image_local_path") or item_metadata.get("image_path") or "")
        same_visual = (
            figure_id
            and figure_id == str(item_metadata.get("figure_id") or "")
            and page == str(item_metadata.get("source_page") or item_metadata.get("page") or "")
        )
        if not (same_image or same_visual):
            continue
        item["page_content"] = page_content
        item_metadata.update(metadata)
        item_metadata["original_text"] = page_content
        changed = True
    if changed:
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return changed


def _validation_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    ids = set()
    for case in json.loads(path.read_text(encoding="utf-8")):
        figure_id = str(case.get("expected_figure_id") or "").strip()
        if figure_id:
            ids.add(figure_id.lower())
    return ids


def _eligible_items(
    queue: Dict[str, object],
    *,
    only_failed: bool,
    figure_id: str,
    validation_only: bool,
    validation_ids: Set[str],
) -> List[Dict[str, object]]:
    items = list(queue.get("items") or [])
    selected = []
    for item in items:
        status = str(item.get("status") or "pending")
        if only_failed and status != "failed":
            continue
        if not only_failed and status not in {"pending", "failed"}:
            continue
        if figure_id and str(item.get("figure_id") or "").lower() != figure_id.lower():
            continue
        if validation_only and str(item.get("figure_id") or "").lower() not in validation_ids:
            continue
        selected.append(item)
    selected.sort(key=lambda item: int(item.get("priority_score") or 0), reverse=True)
    return selected


def _write_update_record(path: Path, record: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    load_dotenv()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Recaption a small batch of queued visual chunks and upsert updates to Pinecone.")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-failed", action="store_true")
    parser.add_argument("--figure-id", default="")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--validation-set", type=Path, default=DEFAULT_VALIDATION_SET)
    parser.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", "bge_small_v1"))
    parser.add_argument("--updates-log", type=Path, default=DEFAULT_UPDATES_PATH)
    parser.add_argument("--bm25-cache", type=Path, default=DEFAULT_BM25_PATH)
    args = parser.parse_args()

    if not args.queue.exists():
        raise SystemExit(f"Queue not found: {args.queue}. Run scripts/build_visual_caption_queue.py first.")
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    validation_ids = _validation_ids(args.validation_set) if args.validation_only else set()
    if args.validation_only and not validation_ids:
        raise SystemExit(f"No validation figure/table IDs found in {args.validation_set}.")
    selected = _eligible_items(
        queue,
        only_failed=args.only_failed,
        figure_id=args.figure_id,
        validation_only=args.validation_only,
        validation_ids=validation_ids,
    )[: args.limit]
    print(
        f"Selected {len(selected)} queued visual(s). dry_run={args.dry_run} "
        f"validation_only={args.validation_only}"
    )
    for item in selected:
        print(
            f"- {item.get('figure_id')} page={item.get('page')} score={item.get('priority_score')} "
            f"path={item.get('image_path')}"
        )
    if args.dry_run or not selected:
        return 0

    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    if not api_key or not index_name:
        raise SystemExit("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME.")

    index = Pinecone(api_key=api_key).Index(index_name)
    embeddings = None
    updated_count = 0
    failed_count = 0
    for item in selected:
        vector_id = str(item.get("vector_id") or "").strip()
        image_path = str(item.get("image_path") or "").strip()
        record = {
            "timestamp_utc": _now(),
            "vector_id": vector_id,
            "figure_id": item.get("figure_id"),
            "page": item.get("page"),
            "image_path": image_path,
        }
        if not vector_id or not image_path or not Path(image_path).exists():
            item["status"] = "failed"
            item["last_error"] = "missing_vector_id_or_image_path"
            record.update({"status": "failed", "reason": item["last_error"]})
            _write_update_record(args.updates_log, record)
            failed_count += 1
            continue
        metadata = _fetch_metadata(index, args.namespace, vector_id)
        if not metadata:
            item["status"] = "failed"
            item["last_error"] = "vector_not_found"
            record.update({"status": "failed", "reason": "vector_not_found"})
            _write_update_record(args.updates_log, record)
            failed_count += 1
            continue
        vision_summary = _caption_image_with_gemini(image_path)
        if not vision_summary:
            item["status"] = "failed"
            item["last_error"] = "gemini_caption_failed_preserved_old_caption"
            item["last_attempt_at"] = _now()
            record.update({"status": "failed", "reason": item["last_error"]})
            _write_update_record(args.updates_log, record)
            failed_count += 1
            continue
        page_content, updated_metadata = _updated_metadata(metadata, vision_summary)
        if embeddings is None:
            embeddings = get_bge_embeddings()
        vector = embeddings.embed_documents([page_content])[0]
        index.upsert(
            vectors=[
                {
                    "id": vector_id,
                    "values": list(vector),
                    "metadata": updated_metadata,
                }
            ],
            namespace=args.namespace,
        )
        bm25_updated = _update_bm25_cache(args.bm25_cache, updated_metadata, page_content)
        item.update(
            {
                "status": "recaptioned",
                "last_success_at": _now(),
                "vision_captioning_status": "success",
                "caption_source": "gemini",
                "new_caption_preview": vision_summary[:240],
                "bm25_cache_updated": bm25_updated,
            }
        )
        record.update(
            {
                "status": "recaptioned",
                "caption_length": len(vision_summary),
                "bm25_cache_updated": bm25_updated,
            }
        )
        _write_update_record(args.updates_log, record)
        updated_count += 1

    args.queue.write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")
    print("RECAPTION SUMMARY")
    print(f"Updated: {updated_count}")
    print(f"Failed: {failed_count}")
    print(f"Queue updated: {args.queue}")
    print(f"Updates log: {args.updates_log}")
    if failed_count and not updated_count:
        print("RECOMMENDATION: if failures show Gemini 429/quota, wait for quota reset and rerun a small validation-only pass, or use a paid tier for one concentrated recaption cycle.")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
