from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure root directory is in Python path for all imports
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Launch main Streamlit interface
from streamlit_ui import StreamlitApp
