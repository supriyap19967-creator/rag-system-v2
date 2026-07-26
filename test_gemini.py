from google import genai
import os

# Initialize client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Use correct model
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain RAG in simple terms"
)

print(response.text)