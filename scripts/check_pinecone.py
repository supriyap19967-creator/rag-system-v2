import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

def check_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key or not index_name:
        print("Error: Missing PINECONE_API_KEY or PINECONE_INDEX_NAME in .env")
        return

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    print(f"Checking Pinecone Index: {index_name}")
    
    # Query with a dummy vector to get top 50 results.
    # BAAI/bge-small-en-v1.5 uses 384 dimensions.
    results = index.query(
        vector=[0.0] * 384,
        top_k=50,
        include_metadata=True
    )
    
    csv_found = False
    gdp_found = False
    
    print("\n--- Top 50 Results Metadata ---")
    for i, match in enumerate(results.matches):
        metadata = match.metadata
        source = metadata.get("source") # build_vector_metadata uses 'source'
        text = metadata.get("original_text", "")
        
        is_csv = (source == "csv")
        has_gdp = "GDP" in text.upper()
        
        if is_csv: csv_found = True
        if has_gdp: gdp_found = True
        
        print(f"[{i:02d}] ID: {match.id} | Source: {source} | GDP in text: {has_gdp}")
        if is_csv or has_gdp:
            print(f"     Content snippet: {text[:100]}...")

    if not csv_found:
        print("\n" + "!"*40)
        print("WARNING: CSV DATA MISSING FROM INDEX")
        print("!"*40)
    else:
        print("\nCSV data was found in the index.")

    if not gdp_found:
        print("No results containing 'GDP' were found.")

if __name__ == "__main__":
    check_pinecone()
