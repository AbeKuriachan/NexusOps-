import sys
from pathlib import Path

# Add the project root to the python path to resolve app imports
sys.path.append(str(Path(__file__).resolve().parent))

from app.ingestion.graph_ingestor import GraphIngestor
from app.config import settings

def main():
    print("--- Starting Graph Ingestion ---")
    try:
        ingestor = GraphIngestor()
        data_dir = settings.BASE_DIR / "data"
        ingestor.ingest(data_dir)
        print("--- Graph Ingestion Complete ---")
    except Exception as e:
        print(f"Error during graph ingestion: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
