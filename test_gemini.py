import os
import pytest
from google import genai

def test_gemini_response():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "dummy_key_for_testing":
        pytest.skip("Skipping test: GOOGLE_API_KEY is not set.")

    # Initialize client
    client = genai.Client(api_key=api_key)

    # Use correct model
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Explain RAG in simple terms"
    )
    print(response.text)