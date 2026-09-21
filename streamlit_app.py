from __future__ import annotations

import os
import sys
from pathlib import Path
import streamlit as st

# 1. MUST BE THE VERY FIRST STREAMLIT COMMAND EXECUTED (Second 0 UI render)
st.set_page_config(
    page_title="V2 Enterprise Multimodal RAG",
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# 2. Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 3. Render instant live loading UI while backend dependencies load
with st.spinner("⚡ Initializing Enterprise RAG Engine & AI Agent Framework..."):
    from streamlit_ui import StreamlitApp

# 4. Execute main application UI
if __name__ == "__main__" or True:
    StreamlitApp.main()
