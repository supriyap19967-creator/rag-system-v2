from qdrant_client import QdrantClient
from pathlib import Path

def main():
    c = QdrantClient(path=str(Path('qdrant_db')))
    cols = c.get_collections().collections
    for col in cols:
        info = c.get_collection(col.name)
        print('Collection:', col.name)
        print('Points:', info.points_count)
        
        # Dense vectors configuration
        dense = info.config.params.vectors
        dense_keys = list(dense.keys()) if hasattr(dense, 'keys') else dense
        print('Dense vectors config:', dense_keys)
        
        # Sparse vectors configuration
        sparse = info.config.params.sparse_vectors
        sparse_keys = list(sparse.keys()) if hasattr(sparse, 'keys') else sparse
        print('Sparse vectors config:', sparse_keys)

if __name__ == '__main__':
    main()
