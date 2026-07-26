Set-Location "C:\Users\supri\recovered-rag-project"

& ".\venv\Scripts\python.exe" -m streamlit run ".\streamlit_ui\Streamlitapp.py" `
  --server.port 8501 `
  --server.address 0.0.0.0 `
  --server.headless true
