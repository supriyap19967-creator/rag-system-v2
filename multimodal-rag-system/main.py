from __future__ import annotations

import sys
import requests
import pandas as pd
import streamlit as st
from pydantic_core import ValidationError

# Set backend endpoint URL
BACKEND_URL = "http://localhost:8000/agent_query"

def main():
    st.set_page_config(page_title="Multi-Modal Agentic RAG Control Panel", layout="wide")
    st.title("📊 Multi-Modal Agentic RAG Control Panel")
    st.subheader("Powered by PydanticAI & Gemini 2.0 (Decoupled Mode)")

    user_query = st.text_input(
        "Enter your query (e.g. 'Query Figure 1.2 coordinate details' or 'Calculate standard trends'):",
        placeholder="Type here..."
    )

    if st.button("Run Pipeline Inquiries") and user_query:
        with st.spinner("Delegating execution to FastAPI backend agent..."):
            try:
                # Call the backend FastAPI agent endpoint
                response = requests.post(
                    BACKEND_URL,
                    json={"query": user_query},
                    timeout=120
                )
                
                if response.status_code != 200:
                    st.error(f"Backend Agent Error (Status {response.status_code}): {response.text}")
                    return

                result_data = response.json()

                st.success("Pipeline executed successfully on backend!")
                
                # Show metadata summary card
                st.info(f"**Source Trail:** {result_data.get('source_routing_trail')}")

                st.write("### 📝 Grounded Reasoning")
                st.write(result_data.get("text_reasoning"))

                # Check for extracted coordinate details
                extracted_table = result_data.get("extracted_table")
                if extracted_table:
                    st.write("### 📈 Extracted Visual Tabular Matrix")
                    
                    # Convert list of rows into a Pandas DataFrame
                    df = pd.DataFrame(extracted_table)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("#### Data Grid View")
                        st.dataframe(df, use_container_width=True)
                        
                    with col2:
                        st.write("#### Visual Representation")
                        cols = list(df.columns)
                        if len(cols) >= 2:
                            st.write(f"Plotting values dynamically...")
                            st.line_chart(df)
                        else:
                            st.dataframe(df)
                else:
                    st.warning("No visual coordinate data returned for this query category.")

            except Exception as exc:
                st.error("Failed to connect or retrieve response from the backend agent server.")
                st.exception(exc)


if __name__ == "__main__":
    main()
