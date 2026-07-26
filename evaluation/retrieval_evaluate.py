from evaluation.evaluation_questions import evaluation_data
from app.reranker import rerank

def recall_at_k(retrieved, relevant, k=5):
    retrieved_ids = [doc.metadata.get("doc_id") for doc in retrieved[:k]]
    hits = len(set(retrieved_ids) & set(relevant))
    return hits / len(relevant)

def evaluate(retriever, k=5):
    scores = []

    for item in evaluation_data:
        query = item["query"]
        relevant = item["relevant_docs"]

        retrieved_docs = retriever.invoke(query)

	# 🔥 Apply reranking
        retrieved = rerank(query, retrieved_docs)
        score = recall_at_k(retrieved, relevant, k)
        scores.append(score)

    return sum(scores) / len(scores)