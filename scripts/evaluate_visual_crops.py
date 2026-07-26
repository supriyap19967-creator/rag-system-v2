import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pdf_visual_extraction import MIN_CROP_QUALITY_SCORE, _quality_score_for_crop


DEFAULT_DEBUG_PATH = ROOT / "Data" / "visual_crop_debug.jsonl"


def _load_records(path: Path, limit: int) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(record)
            if limit and len(records) >= limit:
                break
    return records


def evaluate(debug_path: Path, limit: int = 20) -> Dict[str, object]:
    records = _load_records(debug_path, limit)
    failed_reasons: Counter[str] = Counter()
    examples: List[Dict[str, object]] = []
    usable = 0

    for record in records:
        image_path = Path(str(record.get("final_crop_path") or ""))
        visual_type = str(record.get("visual_type") or "figure").lower()
        if not image_path.is_absolute():
            image_path = ROOT / image_path

        if image_path.exists():
            quality = _quality_score_for_crop(image_path, visual_type)
        else:
            quality = {
                "score": 0.0,
                "usable": False,
                "reasons": ["missing_image_path"],
                "metrics": {},
            }

        score = float(quality.get("score") or 0.0)
        is_usable = bool(quality.get("usable"))
        if is_usable:
            usable += 1
        else:
            reasons = [str(reason) for reason in quality.get("reasons", [])] or ["unknown"]
            failed_reasons.update(reasons)
            if len(examples) < 8:
                examples.append(
                    {
                        "source_pdf": record.get("source_pdf"),
                        "page": record.get("page"),
                        "caption": str(record.get("caption") or "")[:180],
                        "visual_type": visual_type,
                        "image_path": str(image_path),
                        "score": score,
                        "reasons": reasons,
                    }
                )

    total = len(records)
    usable_rate = usable / total if total else 0.0
    return {
        "debug_path": str(debug_path),
        "total_tested": total,
        "usable_crops": usable,
        "usable_crop_rate": round(usable_rate, 3),
        "target_usable_crop_rate": 0.90,
        "minimum_quality_score": MIN_CROP_QUALITY_SCORE,
        "failed_quality_reasons": dict(failed_reasons.most_common()),
        "examples_needing_manual_review": examples,
        "status": "PASS" if usable_rate >= 0.90 and total > 0 else "NEEDS_REVIEW",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extracted PDF visual crop quality from debug JSONL records.")
    parser.add_argument("--debug-path", default=str(DEFAULT_DEBUG_PATH))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    result = evaluate(Path(args.debug_path), limit=args.limit)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
