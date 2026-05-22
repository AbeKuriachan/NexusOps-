import sys
from pathlib import Path

# Add the project root to the python path to resolve app imports
sys.path.append(str(Path(__file__).resolve().parent))

from app.ingestion.vector_ingestor import VectorIngestor
from app.config import settings

def main():
    print("--- Starting Vector Ingestion ---")
    try:
        ingestor = VectorIngestor()
        data_dir = settings.BASE_DIR / "data"
        ingestor.ingest(data_dir)
        print("--- Vector Ingestion Complete ---")
    except Exception as e:
        print(f"Error during vector ingestion: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
