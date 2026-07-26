import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from pinecone import Pinecone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_visual_index import audit
from scripts.evaluate_visual_retrieval import DEFAULT_VALIDATION_SET, evaluate, print_table


DEFAULT_OUTPUT_DIR = ROOT / "evaluation" / "visual_baselines"


def _coverage(summary: Dict[str, object]) -> Dict[str, object]:
    statuses = dict(summary.get("vision_captioning_status_count") or {})
    total = int(summary.get("visual_chunks_sampled") or 0)
    success = int(statuses.get("success") or statuses.get("gemini") or 0)
    fallback = total - success
    return {
        "total_visual_chunks": total,
        "gemini_captioned_count": success,
        "fallback_caption_count": fallback,
        "gemini_caption_coverage": round(success / max(total, 1), 3),
        "vision_captioning_status_count": statuses,
    }


def main() -> int:
    load_dotenv()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Freeze current visual retrieval metrics as a baseline JSON file.")
    parser.add_argument("--validation-set", type=Path, default=DEFAULT_VALIDATION_SET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", "bge_small_v1"))
    parser.add_argument("--audit-limit", type=int, default=10000)
    parser.add_argument("--fresh-prefix", default="visual-baseline")
    args = parser.parse_args()

    os.environ.setdefault("BYPASS_SEMANTIC_CACHE_FOR_VISUAL", "true")
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    if not api_key or not index_name:
        raise SystemExit("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME.")

    client = Pinecone(api_key=api_key)
    index = client.Index(index_name)
    audit_summary = audit(index, args.namespace, args.audit_limit)
    validation_cases = json.loads(args.validation_set.read_text(encoding="utf-8"))
    validation_summary = evaluate(validation_cases, fresh_prefix=args.fresh_prefix)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    coverage = _coverage(audit_summary)
    baseline = {
        "timestamp_utc": timestamp,
        "index": index_name,
        "namespace": args.namespace,
        "validation_set": str(args.validation_set),
        "index_version": {
            "total_vector_count": audit_summary.get("total_vector_count"),
            "visual_chunks": audit_summary.get("visual_chunks_sampled"),
            "unique_visual_count": audit_summary.get("unique_visual_count"),
            "unique_pages_covered": audit_summary.get("unique_pages_covered"),
            "unique_figure_table_ids_covered": audit_summary.get("unique_figure_table_ids_covered"),
        },
        "caption_coverage_estimate": coverage,
        "metrics": {
            "top1_accuracy": validation_summary.get("top1_accuracy"),
            "hit_rate": validation_summary.get("hit_rate"),
            "expected_hit_rate": validation_summary.get("expected_hit_rate"),
            "no_match_precision_proxy": round(
                1 - float(validation_summary.get("false_positive_no_match_rate") or 0),
                3,
            ),
            "false_positive_rate": validation_summary.get("false_positive_no_match_rate"),
            "repeat_rate": validation_summary.get("repeated_figure_rate"),
            "mismatch_rate": validation_summary.get("mismatch_rate"),
            "total_evaluated_queries": validation_summary.get("total"),
        },
        "validation": validation_summary,
        "audit": audit_summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"baseline_{timestamp}.json"
    output_path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")

    print_table(validation_summary)
    print("\nBASELINE SAVED")
    print(f"Path: {output_path}")
    print(
        "Caption coverage: "
        f"{coverage['gemini_captioned_count']}/{coverage['total_visual_chunks']} "
        f"({coverage['gemini_caption_coverage']:.1%}) Gemini-captioned"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
