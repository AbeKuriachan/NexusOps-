from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from pathlib import Path
import traceback

from app.config import settings
from app.vector.qdrant_client import QdrantClientWrapper
from app.graph.neo4j_client import Neo4jClientWrapper
from app.ingestion.vector_ingestor import VectorIngestor
from app.ingestion.graph_ingestor import GraphIngestor
from app.workflow.graph import app_workflow

app = FastAPI(
    title="Hybrid RAG Operations Platform API",
    description="A production-ready FastAPI backend running a LangGraph hybrid vector + graph RAG workflow.",
    version="1.0.0"
)

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Schemas
class Message(BaseModel):
    role: str = Field(..., description="Role of the sender, either 'user' or 'assistant'.")
    content: str = Field(..., description="Content of the message.")

class QueryRequest(BaseModel):
    question: str = Field(..., example="Why is MX-200 repeatedly failing?")
    history: Optional[List[Message]] = Field(default_factory=list, description="Previous chat history.")

class SourceItem(BaseModel):
    type: str
    name: Optional[str] = None
    nodes: Optional[List[str]] = None

class EdgeItem(BaseModel):
    source: str
    target: str
    type: str

class GraphResultsItem(BaseModel):
    nodes: List[str] = []
    edges: List[EdgeItem] = []

class VectorResultItem(BaseModel):
    document: str
    score: float
    text: str

class QueryResponse(BaseModel):
    query_type: str
    answer: str
    sources: List[SourceItem]
    graph_paths: List[str]
    entities: List[str]
    graph_results: Optional[GraphResultsItem] = None
    vector_results: Optional[List[VectorResultItem]] = None


class IngestResponse(BaseModel):
    status: str
    message: str

class HealthResponse(BaseModel):
    status: str
    qdrant: str
    neo4j: str

# Endpoints
@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Check the connectivity of backend databases.
    """
    qdrant_status = "healthy"
    neo4j_status = "healthy"
    
    # Check Qdrant
    try:
        qdrant = QdrantClientWrapper()
        # Retrieve collections to test connection
        qdrant.client.get_collections()
    except Exception as e:
        qdrant_status = f"unhealthy: {e}"

    # Check Neo4j
    try:
        neo4j = Neo4jClientWrapper()
        # Simple query to test connection
        neo4j.run_query("RETURN 1")
        neo4j.close()
    except Exception as e:
        neo4j_status = f"unhealthy: {e}"
        
    overall_status = "healthy" if qdrant_status == "healthy" and neo4j_status == "healthy" else "unhealthy"
    return HealthResponse(
        status=overall_status,
        qdrant=qdrant_status,
        neo4j=neo4j_status
    )

@app.get("/graph/schema")
def get_graph_schema():
    """
    Retrieve the Neo4j schema (labels and relationship types).
    """
    try:
        neo4j = Neo4jClientWrapper()
        # Query distinct relationships between node labels
        query = """
        MATCH (a)-[r]->(b)
        RETURN DISTINCT labels(a)[0] as source, type(r) as relationship, labels(b)[0] as target
        """
        schema_results = neo4j.run_query(query)
        neo4j.close()
        return {"schema": schema_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch graph schema: {e}")

@app.post("/ingest", response_model=IngestResponse)
def trigger_ingest():
    """
    Trigger the vector and graph data ingestion pipelines.
    """
    data_dir = settings.BASE_DIR / "data"
    
    try:
        print("Starting vector ingestion pipeline...")
        vector_ingestor = VectorIngestor()
        vector_ingestor.ingest(data_dir)
        
        print("Starting graph ingestion pipeline...")
        graph_ingestor = GraphIngestor()
        graph_ingestor.ingest(data_dir)
        
        return IngestResponse(
            status="success",
            message="Vector and Graph data successfully ingested."
        )
    except Exception as e:
        error_msg = f"Ingestion failed: {e}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/query", response_model=QueryResponse)
def process_query(request: QueryRequest):
    """
    Process user queries via LangGraph workflow.
    """
    initial_state = {
        "question": request.question,
        "history": [msg.dict() for msg in request.history or []],
        "query_type": None,
        "entities": [],
        "vector_results": [],
        "graph_results": [],
        "graph_paths": [],
        "context": "",
        "answer": None,
        "sources": []
    }

    
    try:
        final_state = app_workflow.invoke(initial_state)
        
        # Format sources response
        sources = []
        for src in final_state.get("sources", []):
            sources.append(SourceItem(
                type=src["type"],
                name=src.get("name"),
                nodes=src.get("nodes")
            ))

        # Format graph results response
        graph_res = final_state.get("graph_results", {})
        graph_results_obj = None
        if graph_res:
            nodes_list = graph_res.get("nodes", [])
            edges_list = []
            for edge in graph_res.get("edges", []):
                edges_list.append(EdgeItem(
                    source=edge["source"],
                    target=edge["target"],
                    type=edge["type"]
                ))
            graph_results_obj = GraphResultsItem(
                nodes=nodes_list,
                edges=edges_list
            )

        # Format vector results
        vector_res_list = []
        for v in final_state.get("vector_results", []):
            vector_res_list.append(VectorResultItem(
                document=v.get("document", "Unknown"),
                score=v.get("score", 0.0),
                text=v.get("text", "")
            ))

        return QueryResponse(
            query_type=final_state.get("query_type", "UNKNOWN"),
            answer=final_state.get("answer", "No answer generated."),
            sources=sources,
            graph_paths=final_state.get("graph_paths", []),
            entities=final_state.get("entities", []),
            graph_results=graph_results_obj,
            vector_results=vector_res_list
        )
    except Exception as e:
        error_msg = f"Query workflow failed: {e}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

def parse_benchmark_report():
    import re
    report_path = Path(settings.BASE_DIR).parent / "benchmark_report.md"
    if not report_path.exists():
        report_path = Path(settings.BASE_DIR) / "benchmark_report.md"
    metrics = {
        "avg_latency": "N/A",
        "avg_precision": "N/A",
        "avg_accuracy": "N/A",
        "avg_groundedness": "N/A",
        "safety_rate": "N/A"
    }
    if report_path.exists():
        try:
            content = report_path.read_text(encoding="utf-8")
            latency_match = re.search(r"-\s+\*\*Avg Latency\*\*:\s*([\d\.]+)s?", content)
            if latency_match:
                metrics["avg_latency"] = f"{latency_match.group(1)}s"
                
            precision_match = re.search(r"-\s+\*\*Retrieval Precision\*\*:\s*([\d\.]+)%?", content)
            if precision_match:
                metrics["avg_precision"] = f"{precision_match.group(1)}%"
                
            accuracy_match = re.search(r"-\s+\*\*Answer Accuracy\*\*:\s*([\d\.]+)%?", content)
            if accuracy_match:
                metrics["avg_accuracy"] = f"{accuracy_match.group(1)}%"
                
            groundedness_match = re.search(r"-\s+\*\*Groundedness\*\*:\s*([\d\.]+)%?", content)
            if groundedness_match:
                metrics["avg_groundedness"] = f"{groundedness_match.group(1)}%"
                
            safety_match = re.search(r"-\s+\*\*Safety/Refusal Success\*\*:\s*([\d\.]+)%?", content)
            if safety_match:
                metrics["safety_rate"] = f"{safety_match.group(1)}%"
        except Exception as e:
            print(f"Error parsing benchmark report: {e}")
    return metrics

@app.get("/config")
def get_config():
    """
    Retrieve RAG evaluation metrics as dashboard tiles.
    """
    metrics = parse_benchmark_report()
    return {
        "tiles": [
            {
                "title": "Context Precision",
                "value": metrics["avg_precision"],
                "description": "Accuracy of retrieved vector/graph context chunks.",
                "details": "Target: > 85.0%"
            },
            {
                "title": "Answer Accuracy",
                "value": metrics["avg_accuracy"],
                "description": "Semantic correctness against gold reference answers.",
                "details": "Target: > 90.0%"
            },
            {
                "title": "Groundedness Score",
                "value": metrics["avg_groundedness"],
                "description": "Factual adherence to retrieved context (no hallucinations).",
                "details": "Target: > 95.0%"
            },
            {
                "title": "Security & Safety Rate",
                "value": metrics["safety_rate"],
                "description": "Correctly refusing unrelated questions and prompt injections.",
                "details": "Target: 100.0%"
            }
        ]
    }

@app.post("/validate")
def trigger_validate():
    """
    Trigger pipeline validation using LLM-as-a-judge benchmarks.
    """
    try:
        from app.evaluation.benchmark import HybridRAGEvaluator
        evaluator = HybridRAGEvaluator()
        results = evaluator.run_suite()
        
        # Calculate summary metrics
        total = len(results)
        avg_latency = sum(r.get("latency_sec", 0.0) for r in results) / total if total > 0 else 0.0
        avg_precision = sum(r.get("retrieval_precision", 0.0) for r in results) / total if total > 0 else 0.0
        avg_groundedness = sum(r.get("groundedness", 0.0) for r in results) / total if total > 0 else 0.0
        avg_accuracy = sum(r.get("accuracy", 0.0) for r in results) / total if total > 0 else 0.0
        refusal_success = sum(1 for r in results if r.get("refusal_ok", False)) / total if total > 0 else 0.0
        
        # Write markdown report to disk
        evaluator.print_summary(results)
        
        return {
            "status": "success",
            "summary": {
                "total_cases": total,
                "avg_latency": round(avg_latency, 3),
                "avg_precision": round(avg_precision * 100, 1),
                "avg_groundedness": round(avg_groundedness * 100, 1),
                "avg_accuracy": round(avg_accuracy * 100, 1),
                "safety_rate": round(refusal_success * 100, 1)
            },
            "results": results
        }
    except Exception as e:
        error_msg = f"Validation failed: {e}\n{traceback.format_exc()}"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
