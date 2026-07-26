import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from app.embeddings import BGE_EMBEDDING_DIMENSIONS
from app.embedding_pipeline import PINECONE_METRIC


load_dotenv()


def _client() -> Pinecone:
    api_key = os.getenv("PINECONE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing PINECONE_API_KEY.")
    return Pinecone(api_key=api_key)


def _index_name() -> str:
    index_name = os.getenv("PINECONE_INDEX_NAME", "").strip()
    if not index_name:
        raise RuntimeError("Missing PINECONE_INDEX_NAME.")
    return index_name


def _namespace() -> str:
    return os.getenv("PINECONE_NAMESPACE", "bge_small_v1").strip() or "bge_small_v1"


def _list_index_names(pc: Pinecone) -> list[str]:
    listed = pc.list_indexes()
    if hasattr(listed, "names"):
        return list(listed.names())
    return [item["name"] for item in listed if isinstance(item, dict) and item.get("name")]


def stats() -> None:
    pc = _client()
    index = pc.Index(_index_name())
    result = index.describe_index_stats()
    print(result)


def clear_namespace() -> None:
    index = _client().Index(_index_name())
    namespace = _namespace()
    print(f"Deleting all vectors in namespace '{namespace}' from index '{_index_name()}'.")
    index.delete(delete_all=True, namespace=namespace)
    print("Namespace clear requested.")


def recreate_index() -> None:
    pc = _client()
    index_name = _index_name()
    cloud = os.getenv("PINECONE_CLOUD", "aws").strip() or "aws"
    region = os.getenv("PINECONE_REGION", "us-east-1").strip() or "us-east-1"

    if index_name in _list_index_names(pc):
        print(f"Deleting Pinecone index '{index_name}'.")
        pc.delete_index(index_name)
        while index_name in _list_index_names(pc):
            time.sleep(2)

    print(
        f"Creating Pinecone index '{index_name}' "
        f"with {BGE_EMBEDDING_DIMENSIONS} dimensions."
    )
    pc.create_index(
        name=index_name,
        dimension=BGE_EMBEDDING_DIMENSIONS,
        metric=PINECONE_METRIC,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    while index_name not in _list_index_names(pc):
        time.sleep(2)
    print("Index recreate complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe Pinecone maintenance helpers.")
    parser.add_argument(
        "action",
        choices=["stats", "clear-namespace", "recreate-index"],
    )
    args = parser.parse_args()

    if args.action == "stats":
        stats()
    elif args.action == "clear-namespace":
        clear_namespace()
    elif args.action == "recreate-index":
        recreate_index()


if __name__ == "__main__":
    main()
