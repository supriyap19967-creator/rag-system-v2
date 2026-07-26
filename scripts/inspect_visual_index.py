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

from app.embeddings import BGE_EMBEDDING_DIMENSIONS, get_bge_embeddings


VISUAL_FILTER = {
    "$and": [
        {"source_type": {"$eq": "pdf"}},
        {"content_type": {"$eq": "visual"}},
    ]
}

KNOWN_HIT_QUERIES = [
    "show figure 4.2 from World Development Report 2025",
    "show chart firms lower-income countries world development report figure",
    "show table about firms adopting standards",
    "table about firms in lower income countries world development report",
]


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


def _score(match: object) -> float:
    if isinstance(match, dict):
        return float(match.get("score") or 0.0)
    return float(getattr(match, "score", 0.0) or 0.0)


def _namespace_vector_count(stats: object, namespace: str) -> object:
    namespaces = _response_value(stats, "namespaces", {}) or {}
    if hasattr(namespaces, "to_dict"):
        namespaces = namespaces.to_dict()
    if not isinstance(namespaces, dict):
        return None
    namespace_stats = namespaces.get(namespace)
    if namespace_stats is None:
        return None
    return _response_value(namespace_stats, "vector_count", None)


def _preview(metadata: Dict[str, object]) -> Dict[str, object]:
    image_path = str(metadata.get("image_local_path") or metadata.get("image_path") or "").strip()
    return {
        "source_type": metadata.get("source_type"),
        "content_type": metadata.get("content_type"),
        "visual_type": metadata.get("visual_type"),
        "source_pdf": metadata.get("source_pdf") or metadata.get("source_files") or metadata.get("source"),
        "page": metadata.get("source_page") or metadata.get("page"),
        "caption": str(metadata.get("caption") or "")[:300],
        "original_text_preview": str(metadata.get("original_text") or "")[:300],
        "image_path": image_path,
        "image_path_exists": bool(image_path and Path(image_path).exists()),
    }


def _probe_vector() -> List[float]:
    vector = [0.0] * BGE_EMBEDDING_DIMENSIONS
    vector[0] = 1.0
    return vector


def inspect_index(index, namespace: str, sample_limit: int, count_limit: int) -> Dict[str, object]:
    stats = index.describe_index_stats()
    response = index.query(
        namespace=namespace,
        vector=_probe_vector(),
        top_k=count_limit,
        include_metadata=True,
        filter=VISUAL_FILTER,
    )
    matches = _matches(response)
    visual_types: Dict[str, int] = {}
    schema_issues: Dict[str, int] = {}
    source_type_values: Dict[str, int] = {}
    content_type_values: Dict[str, int] = {}
    for match in matches:
        metadata = _metadata(match)
        source_type = str(metadata.get("source_type") or "<missing>")
        content_type = str(metadata.get("content_type") or "<missing>")
        visual_type = str(metadata.get("visual_type") or "<missing>")
        source_type_values[source_type] = source_type_values.get(source_type, 0) + 1
        content_type_values[content_type] = content_type_values.get(content_type, 0) + 1
        visual_types[visual_type] = visual_types.get(visual_type, 0) + 1
        if source_type != "pdf":
            schema_issues["source_type_not_pdf"] = schema_issues.get("source_type_not_pdf", 0) + 1
        if content_type != "visual":
            schema_issues["content_type_not_visual"] = schema_issues.get("content_type_not_visual", 0) + 1
        if visual_type == "<missing>":
            schema_issues["missing_visual_type"] = schema_issues.get("missing_visual_type", 0) + 1

    return {
        "namespace": namespace,
        "total_vector_count": _response_value(stats, "total_vector_count", None),
        "namespace_vector_count": _namespace_vector_count(stats, namespace),
        "visual_docs_count": len(matches),
        "visual_docs_count_limit": count_limit,
        "visual_count_truncated": len(matches) >= count_limit,
        "source_type_values": source_type_values,
        "content_type_values": content_type_values,
        "visual_type_values": visual_types,
        "schema_issues": schema_issues,
        "samples": [_preview(_metadata(match)) for match in matches[:sample_limit]],
    }


def run_known_hit_queries(index, namespace: str, queries: List[str], top_k: int) -> List[Dict[str, object]]:
    embedder = get_bge_embeddings()
    results: List[Dict[str, object]] = []
    for query in queries:
        vector = embedder.embed_query(query)
        response = index.query(
            namespace=namespace,
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            filter=VISUAL_FILTER,
        )
        matches = _matches(response)
        results.append(
            {
                "query": query,
                "match_count": len(matches),
                "top_matches": [
                    {
                        "score": _score(match),
                        **_preview(_metadata(match)),
                    }
                    for match in matches[:3]
                ],
            }
        )
    return results


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Inspect indexed PDF visual documents in Pinecone.")
    parser.add_argument("--namespace", default=os.getenv("PINECONE_NAMESPACE", "bge_small_v1"))
    parser.add_argument("--sample-limit", type=int, default=5)
    parser.add_argument("--count-limit", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-known-queries", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    if not api_key or not index_name:
        raise SystemExit("Missing PINECONE_API_KEY or PINECONE_INDEX_NAME.")

    client = Pinecone(api_key=api_key)
    index = client.Index(index_name)
    summary = inspect_index(index, args.namespace, args.sample_limit, args.count_limit)
    summary["index"] = index_name
    if not args.skip_known_queries:
        summary["known_hit_queries"] = run_known_hit_queries(index, args.namespace, KNOWN_HIT_QUERIES, args.top_k)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if int(summary.get("visual_docs_count") or 0) == 0:
        print("VISUAL_INDEX_STATUS: empty visual index; visual ingestion/reindexing is required.")
        return 2
    print("VISUAL_INDEX_STATUS: visual documents found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
