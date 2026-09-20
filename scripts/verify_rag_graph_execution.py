from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.abspath("."))
os.environ["QDRANT_PATH"] = os.path.abspath("./qdrant_db")

from app.graph.workflow import rag_graph

def test_graph():
    test_queries = [
        "exctract figure 7.5",
        "extract table 2.1",
        "extract figure 4.3"
    ]

    print("=========================================================")
    print("=== LANGGRAPH STATEGRAPH END-TO-END VERIFICATION TEST ===")
    print("=========================================================\n")

    for i, q in enumerate(test_queries, 1):
        print(f"--- TEST {i}: Query = '{q}' ---")
        config = {"configurable": {"thread_id": f"test_session_{i}"}}
        try:
            state_output = rag_graph.invoke(
                {
                    "question": q,
                    "session_id": f"test_session_{i}",
                    "retry_count": 0,
                },
                config=config
            )
            print(f"Status: SUCCESS")
            print(f"Intent: {state_output.get('structural_intent')}")
            print(f"Active Image Path: {state_output.get('active_image_path')}")
            print(f"Active CSV Path: {state_output.get('active_csv_path')}")
            print(f"Answer (First 150 chars): {str(state_output.get('generated_answer'))[:150]}...\n")
        except Exception as exc:
            print(f"Status: FAILED with error: {exc}\n")

if __name__ == "__main__":
    test_graph()
