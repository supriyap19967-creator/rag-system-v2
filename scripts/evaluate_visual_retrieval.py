import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_VALIDATION_SET = Path(__file__).with_name("visual_validation_set.json")


def _load_cases(path: Path) -> List[Dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _visuals_from_result(result: object) -> List[Tuple[str, object, str]]:
    from app.main import _visual_figure_id

    docs = list(getattr(result, "answer_docs", []) or [])
    visual_docs = [
        doc
        for doc in docs
        if str(doc.metadata.get("content_type", "")).lower() == "visual"
    ]
    results = []
    for doc in visual_docs:
        metadata = doc.metadata
        source = os.path.basename(str(metadata.get("source_pdf") or metadata.get("source_files") or metadata.get("source") or ""))
        page = metadata.get("source_page") or metadata.get("page") or ""
        results.append((_visual_figure_id(doc), page, source))
    return results


def _matches_expected(case: Dict[str, object], returned_figure: str, returned_page: object) -> bool:
    expected_behavior = str(case.get("expected_behavior") or "")
    expected_figure = str(case.get("expected_figure_id") or "").strip()
    expected_page = str(case.get("expected_page") or "").strip()
    if expected_behavior == "no_confident_match":
        return not returned_figure
    if expected_behavior == "figure_found" and not returned_figure:
        return False
    if expected_figure and expected_figure.lower() != returned_figure.lower():
        return False
    if expected_page and expected_page != str(returned_page):
        return False
    return True


def _expected_found_anywhere(case: Dict[str, object], returned_visuals: List[Tuple[str, object, str]]) -> bool:
    expected_behavior = str(case.get("expected_behavior") or "")
    expected_figure = str(case.get("expected_figure_id") or "").strip()
    expected_page = str(case.get("expected_page") or "").strip()
    if expected_behavior == "no_confident_match":
        return not returned_visuals
    if expected_behavior == "figure_found" and not expected_figure:
        return bool(returned_visuals)
    for figure, page, _source in returned_visuals:
        if expected_figure and expected_figure.lower() != str(figure).lower():
            continue
        if expected_page and expected_page != str(page):
            continue
        return True
    return False


def _top1_expected_id(case: Dict[str, object], returned_figure: str) -> bool:
    expected_behavior = str(case.get("expected_behavior") or "")
    expected_figure = str(case.get("expected_figure_id") or "").strip()
    if expected_behavior == "no_confident_match":
        return not returned_figure
    if expected_behavior == "figure_found" and not expected_figure:
        return bool(returned_figure)
    return bool(expected_figure and expected_figure.lower() == str(returned_figure).lower())


def _topic_similarity(left: str, right: str) -> float:
    stopwords = {
        "show",
        "chart",
        "figure",
        "figures",
        "table",
        "visual",
        "diagram",
        "about",
        "what",
        "does",
        "from",
        "world",
        "development",
        "report",
    }
    left_terms = {term for term in left.lower().replace("-", " ").split() if len(term) > 2 and term not in stopwords}
    right_terms = {term for term in right.lower().replace("-", " ").split() if len(term) > 2 and term not in stopwords}
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(len(left_terms | right_terms), 1)


def evaluate(cases: List[Dict[str, object]], *, fresh_prefix: str) -> Dict[str, object]:
    from app.main import _execute_single_query
    from app.retriever import RetrievalHints, get_relevant_documents

    rows = []
    mismatches = 0
    hits = 0
    no_matches = 0
    repeated = 0
    false_positive_no_match = 0
    expected_hits_anywhere = 0
    top1_correct = 0
    last_returned_key = None
    last_query = ""
    sticky_reuse_count = 0
    for index, case in enumerate(cases, start=1):
        query = str(case["query"])
        session_id = f"{fresh_prefix}-{index}-{uuid.uuid4().hex[:8]}"
        candidate_count = 0
        try:
            candidate_result = get_relevant_documents(
                query,
                top_k=10,
                hints=RetrievalHints(source_type="pdf", content_type="visual"),
            )
            candidate_count = int(candidate_result.semantic_match_count or len(candidate_result.documents))
        except Exception as exc:
            candidate_count = 0
        result = _execute_single_query(query, session_id, [])
        returned_visuals = _visuals_from_result(result)
        returned_figure, returned_page, returned_source = returned_visuals[0] if returned_visuals else ("", "", "")
        expected_figure = str(case.get("expected_figure_id") or "")
        expected_page = case.get("expected_page") or ""
        ok = _matches_expected(case, returned_figure, returned_page)
        hit_anywhere = _expected_found_anywhere(case, returned_visuals)
        top1_ok = _top1_expected_id(case, returned_figure)
        if not ok:
            mismatches += 1
        if str(case.get("expected_behavior") or "") == "no_confident_match" and returned_figure:
            false_positive_no_match += 1
        if hit_anywhere:
            expected_hits_anywhere += 1
        if top1_ok:
            top1_correct += 1
        if returned_figure:
            hits += 1
        else:
            no_matches += 1
        returned_key = (returned_source, returned_page, returned_figure)
        repeated_figure = bool(returned_figure and last_returned_key == returned_key)
        if repeated_figure:
            repeated += 1
        if repeated_figure and _topic_similarity(last_query, query) < 0.35:
            sticky_reuse_count += 1
        if returned_figure:
            last_returned_key = returned_key
            last_query = query
        rows.append(
            {
                "query": query,
                "expected_behavior": case.get("expected_behavior"),
                "expected_figure_id": expected_figure,
                "expected_page": expected_page,
                "returned_figure_id": returned_figure,
                "returned_page": returned_page,
                "returned_source": returned_source,
                "returned_visual_ids": [
                    {"figure_id": figure, "page": page, "source": source}
                    for figure, page, source in returned_visuals
                ],
                "pre_rerank_candidate_count": candidate_count,
                "visual_shown": bool(returned_figure),
                "repeated_figure": repeated_figure,
                "no_match": not bool(returned_figure),
                "hit_anywhere": hit_anywhere,
                "top1_correct": top1_ok,
                "match": ok,
                "notes": case.get("notes", ""),
            }
        )
    total = len(cases)
    return {
        "total": total,
        "mismatches": mismatches,
        "hits": hits,
        "no_matches": no_matches,
        "repeated_figures": repeated,
        "false_positive_no_match": false_positive_no_match,
        "expected_hits_anywhere": expected_hits_anywhere,
        "top1_correct": top1_correct,
        "false_positive_no_match_rate": round(false_positive_no_match / max(total, 1), 3),
        "hit_rate": round(hits / max(total, 1), 3),
        "expected_hit_rate": round(expected_hits_anywhere / max(total, 1), 3),
        "top1_accuracy": round(top1_correct / max(total, 1), 3),
        "repeated_figure_rate": round(repeated / max(total, 1), 3),
        "no_match_rate": round(no_matches / max(total, 1), 3),
        "mismatch_rate": round(mismatches / max(total, 1), 3),
        "sticky_reuse_count": sticky_reuse_count,
        "rows": rows,
    }


def print_table(summary: Dict[str, object]) -> None:
    print("VISUAL RETRIEVAL VALIDATION")
    print(f"Total: {summary['total']}  Mismatches: {summary['mismatches']}  Mismatch rate: {summary['mismatch_rate']}")
    print(
        f"Visual shown rate: {summary['hit_rate']}  Expected-hit rate: {summary['expected_hit_rate']}  "
        f"Top-1 accuracy: {summary['top1_accuracy']}"
    )
    print(
        f"Repeated-figure rate: {summary['repeated_figure_rate']}  No-match rate: {summary['no_match_rate']}"
    )
    print(
        f"No-match false positives: {summary['false_positive_no_match']}  "
        f"False-positive rate: {summary['false_positive_no_match_rate']}"
    )
    print(f"Adjacent sticky reuse count: {summary['sticky_reuse_count']}")
    for row in summary["rows"]:
        status = "OK" if row["match"] else "MISMATCH"
        print(
            f"[{status}] {row['query']} | returned={row['returned_figure_id'] or 'none'} "
            f"page={row['returned_page'] or 'none'} | expected={row['expected_figure_id'] or row['expected_behavior']} "
            f"page={row['expected_page'] or 'any'} | pre-rerank candidates={row['pre_rerank_candidate_count']}"
        )


def main() -> int:
    load_dotenv()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Evaluate visual retrieval against a small validation set.")
    parser.add_argument("--validation-set", type=Path, default=DEFAULT_VALIDATION_SET)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a compact table.")
    parser.add_argument("--list", action="store_true", help="Only list validation cases; do not query the app.")
    parser.add_argument("--fresh-prefix", default="visual-eval")
    args = parser.parse_args()

    os.environ.setdefault("BYPASS_SEMANTIC_CACHE_FOR_VISUAL", "true")
    cases = _load_cases(args.validation_set)
    if args.list:
        print(json.dumps({"case_count": len(cases), "cases": cases}, indent=2, ensure_ascii=False))
        return 0

    summary = evaluate(cases, fresh_prefix=args.fresh_prefix)
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print_table(summary)
    print("\nDebug workflow: use fresh sessions or bypass cache, compare returned (source,page,figure_id), inspect visual_candidate_top5 logs, then audit index coverage if many results are missing.")
    return 1 if summary["mismatches"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
