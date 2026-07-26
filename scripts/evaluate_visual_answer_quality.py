import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_VALIDATION_SET = ROOT / "scripts" / "visual_validation_set.json"


def _load_cases(path: Path) -> List[Dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _answer_text(result: object) -> str:
    structured = getattr(result, "structured_answer", None)
    if hasattr(structured, "answer"):
        return str(structured.answer or "")
    answer = getattr(result, "answer", None)
    if hasattr(answer, "answer"):
        return str(answer.answer or "")
    return str(answer or "")


def _fact_count(answer: str) -> int:
    fact_section = re.search(
        r"Key extracted facts:\s*(.*?)(?:\n\s*Related paragraph insight:|\Z)",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not fact_section:
        return 0
    return len(re.findall(r"(?m)^\s*\*\s+\S", fact_section.group(1)))


def _title_repetition_only(answer: str) -> bool:
    facts = _fact_count(answer)
    if facts >= 2:
        return False
    return bool(re.search(r"What the visual shows:\s*(Figure|Table|Chart|Panel)\s+\d", answer, flags=re.IGNORECASE))


def _ocr_noise(answer: str) -> bool:
    return bool(re.search(r"(?:Â|Ã|â€“|[|_~]{2,}|\b\d+\s+\d+\s+\d+\s+\d+\b)", answer))


def _paragraph_alignment(answer: str) -> bool:
    match = re.search(
        r"Related paragraph insight:\s*(.*?)(?:\n\s*Combined interpretation:|\Z)",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return False
    insight = match.group(1).strip().lower()
    return bool(insight and "no strongly aligned" not in insight)


def main() -> int:
    load_dotenv()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Evaluate visual answer quality separately from retrieval quality.")
    parser.add_argument("--validation-set", type=Path, default=DEFAULT_VALIDATION_SET)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fresh-prefix", default="visual-answer-quality")
    args = parser.parse_args()

    os.environ.setdefault("BYPASS_SEMANTIC_CACHE_FOR_VISUAL", "true")
    from app.main import _execute_single_query

    cases = _load_cases(args.validation_set)
    if args.limit > 0:
        cases = cases[: args.limit]
    rows = []
    for index, case in enumerate(cases, start=1):
        query = str(case["query"])
        session_id = f"{args.fresh_prefix}-{index}-{uuid.uuid4().hex[:8]}"
        result = _execute_single_query(query, session_id, [])
        answer = _answer_text(result)
        row = {
            "query": query,
            "factfulness_depth_score": _fact_count(answer),
            "title_repetition_only": _title_repetition_only(answer),
            "ocr_noise_leaked": _ocr_noise(answer),
            "paragraph_alignment_used": _paragraph_alignment(answer),
            "answer_preview": answer[:300],
        }
        rows.append(row)

    total = len(rows)
    summary = {
        "total": total,
        "average_factfulness_depth_score": round(
            sum(row["factfulness_depth_score"] for row in rows) / max(total, 1),
            3,
        ),
        "title_repetition_rate": round(sum(1 for row in rows if row["title_repetition_only"]) / max(total, 1), 3),
        "ocr_noise_leakage_rate": round(sum(1 for row in rows if row["ocr_noise_leaked"]) / max(total, 1), 3),
        "paragraph_alignment_hit_rate": round(sum(1 for row in rows if row["paragraph_alignment_used"]) / max(total, 1), 3),
        "rows": rows,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
