from qdrant_client import QdrantClient as BaseQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from app.config import settings

class QdrantClientWrapper:
    def __init__(self):
        self.client = BaseQdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )

    def init_collection(self, vector_size: int, distance=Distance.COSINE):
        """
        Re-create or ensure collection exists with the appropriate vector size and distance metric.
        """
        collection_name = settings.QDRANT_COLLECTION_NAME
        
        # Check if collection already exists
        collections = self.client.get_collections()
        exists = any(col.name == collection_name for col in collections.collections)
        
        if not exists:
            print(f"Creating collection '{collection_name}' in Qdrant with vector size {vector_size}...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
        else:
            print(f"Collection '{collection_name}' already exists in Qdrant.")

    def upsert_chunks(self, chunks):
        """
        Upsert a list of document chunks into Qdrant.
        Each chunk should be a dict or object containing:
        - id: unique identifier (int or uuid)
        - vector: list of floats
        - payload: dict containing text and other metadata
        """
        points = []
        for chunk in chunks:
            points.append(
                PointStruct(
                    id=chunk["id"],
                    vector=chunk["vector"],
                    payload=chunk["payload"]
                )
            )
        
        self.client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            wait=True,
            points=points
        )
        print(f"Successfully upserted {len(chunks)} chunks into Qdrant.")

