import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

def purge_index():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    namespace = os.getenv("PINECONE_NAMESPACE", "bge_small_v1")
    
    if not api_key or not index_name:
        print("Error: Missing PINECONE_API_KEY or PINECONE_INDEX_NAME in .env")
        return

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    print(f"Purging Pinecone Index: {index_name}, Namespace: {namespace}")
    try:
        index.delete(delete_all=True, namespace=namespace)
        print("Successfully deleted all vectors in the namespace.")
    except Exception as e:
        print(f"Error purging index: {e}")

if __name__ == "__main__":
    purge_index()
