from dotenv import load_dotenv
from app.llm import get_hybrid_llm
from app.retriever import get_relevant_documents
from app.schemas import IntentCategory

load_dotenv()

def test_intent_classification():
    llm = get_hybrid_llm()
    
    queries = [
        ("What was the GDP of India in 2020?", IntentCategory.NUMERICAL),
        ("What are the KYC regulations for banks?", IntentCategory.QUALITATIVE),
        ("How does the new carbon tax affect GDP growth?", IntentCategory.HYBRID)
    ]
    
    for query, _expected_intent in queries:
        result = llm.classify_intent(query)
        print(f"Query: {query}")
        print(f"Detected Intent: {result.intent}")
        print(f"Reasoning: {result.reasoning}")
        # We don't strictly assert because LLM can be variable, but we print for manual verification
        print("-" * 20)

def test_weighted_retrieval():
    # Numerical Query
    query_num = "What was the GDP of India in 2020?"
    print(f"\nTesting Numerical Retrieval for: {query_num}")
    result_num = get_relevant_documents(query_num, top_k=5)
    print(f"Mode: {result_num.mode}")
    for document in result_num.documents:
        print(document.metadata)
    
    # Qualitative Query
    query_qual = "What are the KYC regulations?"
    print(f"\nTesting Qualitative Retrieval for: {query_qual}")
    result_qual = get_relevant_documents(query_qual, top_k=5)
    print(f"Mode: {result_qual.mode}")
    for document in result_qual.documents:
        print(document.metadata)

if __name__ == "__main__":
    print("=== Testing Intent Classification ===")
    test_intent_classification()
    print("\n=== Testing Weighted Retrieval ===")
    test_weighted_retrieval()
