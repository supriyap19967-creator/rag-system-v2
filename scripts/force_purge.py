import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

def force_purge():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key or not index_name:
        print("Error: Missing PINECONE_API_KEY or PINECONE_INDEX_NAME in .env")
        return

    pc = Pinecone(api_key=api_key)
    
    # List indexes to verify the name
    indexes = pc.list_indexes()
    index_names = [i.name for i in indexes]
    print(f"Available indexes: {index_names}")
    
    if index_name not in index_names:
        print(f"Error: Index '{index_name}' not found. Please check your PINECONE_INDEX_NAME.")
        return

    index = pc.Index(index_name)
    
    namespaces = ["default", "financial-rag", "v2_clean_data"]
    
    for ns in namespaces:
        print(f"Attempting to purge namespace: {ns}")
        try:
            index.delete(delete_all=True, namespace=ns)
            print(f"Successfully deleted all vectors in namespace: {ns}")
        except Exception as e:
            print(f"Namespace '{ns}' might already be empty or error occurred: {e}")

    print("\nForce purge complete.")

if __name__ == "__main__":
    force_purge()
