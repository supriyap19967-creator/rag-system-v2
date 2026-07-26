from app.retriever import get_relevant_documents
from evaluation.evaluation_questions import evaluation_data

print("\n=== Retrieval Smoke Evaluation ===")
for item in evaluation_data:
    question = item["query"]
    print(f"\nQuestion: {question}")
    result = get_relevant_documents(question, top_k=5)
    print(f"Mode: {result.mode}")
    for document in result.documents:
        print(document.page_content[:200])
