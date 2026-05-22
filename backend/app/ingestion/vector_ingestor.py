import glob
import uuid
from pathlib import Path
from app.config import settings
from app.vector.qdrant_client import QdrantClientWrapper

class VectorIngestor:
    def __init__(self):
        self.qdrant_wrapper = QdrantClientWrapper()
        self.embedding_provider = settings.EMBEDDING_PROVIDER
        
        # Initialize embedding model
        if self.embedding_provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY environment variable is not set, but provider is 'openai'")
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.vector_size = 1536
        else:
            print("Loading local SentenceTransformer model (all-MiniLM-L6-v2)...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            self.vector_size = 384

    def get_embedding(self, text: str):
        if self.embedding_provider == "openai":
            response = self.openai_client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        else:
            # Use local SentenceTransformer
            embedding = self.model.encode(text)
            return embedding.tolist()

    def chunk_text(self, text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
        """
        Split text into overlapping chunks.
        """
        if len(text) <= chunk_size:
            return [text.strip()]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end].strip())
            start += chunk_size - chunk_overlap
        return chunks

    def ingest(self, data_dir: Path):
        """
        Ingest all .txt files from the data directory into Qdrant.
        """
        # Ensure collection is initialized in Qdrant
        self.qdrant_wrapper.init_collection(vector_size=self.vector_size)

        txt_files = glob.glob(str(data_dir / "*.txt"))
        if not txt_files:
            print(f"No .txt files found in {data_dir}")
            return

        print(f"Found {len(txt_files)} text files to ingest.")
        
        chunks_to_upsert = []
        for file_path_str in txt_files:
            file_path = Path(file_path_str)
            file_name = file_path.name
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            chunks = self.chunk_text(content)
            print(f"Processing '{file_name}' ({len(chunks)} chunk(s))...")
            
            for idx, chunk_text in enumerate(chunks):
                # Generate deterministic UUID based on file name and chunk index
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_name}_{idx}"))
                vector = self.get_embedding(chunk_text)
                
                payload = {
                    "text": chunk_text,
                    "source": file_name,
                    "chunk_index": idx
                }
                
                chunks_to_upsert.append({
                    "id": point_id,
                    "vector": vector,
                    "payload": payload
                })

        if chunks_to_upsert:
            print(f"Upserting {len(chunks_to_upsert)} total points into Qdrant...")
            self.qdrant_wrapper.upsert_chunks(chunks_to_upsert)
            print("Vector ingestion completed successfully.")
        else:
            print("No chunks generated for ingestion.")

