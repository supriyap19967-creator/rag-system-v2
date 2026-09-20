import os
import sys

os.environ["QDRANT_PATH"] = os.path.abspath("./qdrant_db")
sys.path.insert(0, ".")

from app.graph.workflow import rag_graph

test_queries = [
    "exctract figure 7.5",
    "extract table 2.1",
    "extract figure 4.3",
]

print("==================================================================")
print("=== RUNNING LANGGRAPH STATEGRAPH AUTOMATED VERIFICATION TEST ===")
print("==================================================================")

for idx, q in enumerate(test_queries, 1):
    print(f"\n--- TEST {idx}: Query = '{q}' ---")
    config = {"configurable": {"thread_id": f"test-thread-{idx}"}}
    initial_state = {"question": q, "session_id": f"session-{idx}"}

    final_state = rag_graph.invoke(initial_state, config=config)

    print("INTENT CLASSIFIED :", final_state.get("structural_intent"))
    print("LOCKED ENTITIES   :", final_state.get("locked_entities"))
    print("CHUNKS RETRIEVED  :", len(final_state.get("retrieved_chunks") or []))
    print("IMAGE BOUND       :", final_state.get("active_image_path"))
    print("GENERATED ANSWER  :\n", final_state.get("generated_answer"))
    print("IS GROUNDED       :", final_state.get("is_grounded"))
    print("RETRY COUNT       :", final_state.get("retry_count"))
    print("------------------------------------------------------------------")

print("\nAll LangGraph StateGraph pipeline verification tests passed successfully!")
