from typing import TypedDict, List, Dict, Any, Optional

class RAGState(TypedDict):
    question: str
    history: List[Dict[str, str]]  # List of {"role": "user"|"assistant", "content": "..."}
    query_type: Optional[str]
    entities: List[str]
    vector_results: List[Dict[str, Any]]
    graph_results: List[Dict[str, Any]]
    graph_paths: List[str]
    context: str
    answer: Optional[str]
    sources: List[Dict[str, Any]]
