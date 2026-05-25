# NexusOps — Hybrid RAG Platform for Smart Manufacturing Operations

> An AI-powered operations intelligence platform that allows plant operators to query incident logs, SOPs, asset relationships, and supplier dependencies through a hybrid **Vector + Knowledge Graph RAG** pipeline — backed by a LangGraph multi-agent workflow, Qdrant, Neo4j, and GPT-4o.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Backend Deep Dive](#backend-deep-dive)
  - [Configuration](#configuration)
  - [Data Ingestion](#data-ingestion)
  - [LangGraph Query Workflow](#langgraph-query-workflow)
  - [API Endpoints](#api-endpoints)
  - [Evaluation System](#evaluation-system)
- [Frontend](#frontend)
- [Running Locally](#running-locally)
  - [Prerequisites](#prerequisites)
  - [Environment Setup](#environment-setup)
  - [Start Databases](#start-databases)
  - [Ingest Data](#ingest-data)
  - [Start Backend API](#start-backend-api)
  - [Start Frontend](#start-frontend)
- [CI / CD Pipelines](#ci--cd-pipelines)
  - [RAG Evaluation CI](#rag-evaluation-ci)
  - [Docker Build Verification](#docker-build-verification)
- [Benchmark Results](#benchmark-results)
- [Data Schema](#data-schema)

---

## Overview

NexusOps is built for manufacturing operations teams who need to query heterogeneous data — structured asset/incident CSVs and unstructured SOP text documents — through natural language.

The system intelligently routes each query to the appropriate retrieval pipeline:

| Query Type | Example | Retrieval Used |
|---|---|---|
| `VECTOR_ONLY` | "How do I restart MX-200?" | Qdrant semantic search over SOP documents |
| `GRAPH_ONLY` | "Who owns MX-200?" | Neo4j Cypher path traversal |
| `HYBRID` | "Which supplier causes most downtime?" | Both, run in parallel |
| `REFUSED` | "What is the capital of France?" | Rejected — returns a refusal |

---

## Architecture

```
User Query (React Frontend)
        │
        ▼
  FastAPI REST API
        │
        ▼
 LangGraph Workflow
        │
   ┌────┴─────────────┐
   │   classify_query │  (GPT-4o-mini, structured output)
   └────┬─────────────┘
        │
   ┌────┴────────────────────────────────────┐
   │         route_query                     │
   │  VECTOR_ONLY │ GRAPH_ONLY │ HYBRID      │
   └──────┬───────┴──────┬─────┴──────┬──────┘
          │              │            │ (parallel)
          ▼              ▼            ▼
   retrieve_vector  retrieve_graph  [both]
   (Qdrant)         (Neo4j)
          │              │
          └──────┬───────┘
                 ▼
           build_context
                 │
                 ▼
          generate_answer  (GPT-4o-mini, structured + cited sources)
                 │
                 ▼
          FastAPI Response → Frontend
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM & Orchestration** | OpenAI GPT-4o-mini, GPT-4o (judge), LangGraph, LangChain |
| **Vector Database** | Qdrant (Docker) |
| **Graph Database** | Neo4j 5 (Docker) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim) or OpenAI `text-embedding-3-small` (1536-dim) |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | React 18 + TypeScript + Vite |
| **Graph Visualization** | `react-force-graph-2d` |
| **Containerization** | Docker + Docker Compose |
| **CI / CD** | GitHub Actions |

---

## Project Structure

```
NexusOps-/
├── .github/
│   └── workflows/
│       ├── rag-eval-ci.yml          # RAG Evaluation CI (manual trigger)
│       └── docker-verify.yml        # Docker Build Verification CI
│
├── backend/
│   ├── app/
│   │   ├── config/
│   │   │   └── settings.py          # Centralized environment config
│   │   ├── evaluation/
│   │   │   └── benchmark.py         # LLM-as-a-judge evaluation suite
│   │   ├── graph/
│   │   │   └── neo4j_client.py      # Neo4j driver wrapper
│   │   ├── ingestion/
│   │   │   ├── graph_ingestor.py    # CSV → Neo4j ingestion
│   │   │   └── vector_ingestor.py   # TXT → Qdrant ingestion
│   │   ├── vector/
│   │   │   └── qdrant_client.py     # Qdrant client wrapper
│   │   └── workflow/
│   │       ├── graph.py             # LangGraph state machine definition
│   │       ├── nodes.py             # Workflow node implementations
│   │       └── state.py             # RAGState TypedDict schema
│   ├── data/                        # Mock manufacturing dataset
│   │   ├── Assets.csv
│   │   ├── Vendors.csv
│   │   ├── Team_Structure.csv
│   │   ├── Maintenance_Records.csv
│   │   ├── graph_edges.csv          # Edge list for Neo4j knowledge graph
│   │   ├── Incident_2025_001.txt
│   │   ├── Incident_2025_002.txt
│   │   ├── SOP_Assembly_Line_A.txt
│   │   └── SOP_Quality_Checks.txt
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── ingest_vector.py             # Standalone runner for vector ingestion
│   ├── ingest_graph.py              # Standalone runner for graph ingestion
│   └── requirements.txt
│
├── frontend/
│   └── src/
│       ├── App.tsx                  # Chat interface + graph inspector
│       ├── Pipelines.tsx            # Control panel + RAG metrics tiles
│       ├── App.css                  # Dark theme, glassmorphism styles
│       └── index.css                # CSS design tokens
│
├── benchmark_report.md              # Latest RAG evaluation results
└── .gitignore
```

---

## Backend Deep Dive

### Configuration

All settings are loaded from `backend/.env` via `backend/app/config/settings.py`:

| Variable | Description | Default |
|---|---|---|
| `QDRANT_HOST` | Qdrant host | `localhost` |
| `QDRANT_PORT` | Qdrant port | `6333` |
| `QDRANT_COLLECTION_NAME` | Collection name | `operations_docs` |
| `NEO4J_URI` | Neo4j Bolt URI | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | — |
| `OPENAI_API_KEY` | OpenAI API key for LLM nodes | — |
| `EMBEDDING_PROVIDER` | `local` or `openai` | `local` |

> ⚠️ **Never commit `.env` to version control.** It is excluded by `.gitignore`.

---

### Data Ingestion

There are two independent ingestion pipelines that can be run standalone or triggered via the `/ingest` API endpoint.

#### Vector Ingestion (`ingest_vector.py`)

1. Scans `backend/data/` for all `.txt` files (SOPs, incident reports).
2. Splits each file into overlapping character chunks (`size=500`, `overlap=100`).
3. Generates embeddings using either:
   - **Local**: `sentence-transformers/all-MiniLM-L6-v2` → 384-dimensional vectors
   - **OpenAI**: `text-embedding-3-small` → 1536-dimensional vectors
4. Upserts all points into the configured Qdrant collection with deterministic UUIDs (based on `filename + chunk_index`), making re-ingestion idempotent.

#### Graph Ingestion (`ingest_graph.py`)

1. Reads `backend/data/graph_edges.csv` — a structured edge list with columns: `source`, `relationship`, `target`.
2. Builds entity classification lookups from supporting CSVs (`Assets.csv`, `Vendors.csv`, `Team_Structure.csv`) to assign correct Neo4j labels:

| Label | Matched By |
|---|---|
| `Asset` | Found in `Assets.csv` `Asset` column |
| `Vendor` | Found in `Vendors.csv` `Vendor` column |
| `Employee` | Found in `Assets.csv` `Owner` or `Team_Structure.csv` `Employee` column |
| `Team` | Found in `Team_Structure.csv` `Team` column |
| `Incident` | Name starts with `INC-` |
| `Component` | Name starts with `CV-` |
| `Location` | Name contains `plant` |

3. Runs Cypher `MERGE` statements for each node and relationship — making ingestion idempotent.

---

### LangGraph Query Workflow

The query pipeline is a compiled LangGraph `StateGraph` with six nodes:

#### 1. `classify_query`
- **Model**: GPT-4o-mini with structured output (`QueryAnalysis` Pydantic schema)
- **Inputs**: User question + conversation history
- **Outputs**: `query_type` (`VECTOR_ONLY | GRAPH_ONLY | HYBRID | REFUSED`) + `entities` list
- Uses conversation history to resolve pronouns across turns (e.g. "Who owns *it*?" → resolves "it" to "MX-200" from context)
- Has a keyword-based fallback if no API key is configured

#### 2. `route_query`
Conditional router that determines which downstream nodes to invoke:
- `VECTOR_ONLY` → vector retrieval only
- `GRAPH_ONLY` → graph retrieval only
- `HYBRID` → **both in parallel** (LangGraph fan-out)
- `REFUSED` → refusal node

#### 3. `retrieve_vector`
- Embeds the question using the configured embedding provider
- Queries Qdrant for the **top-10** most similar document chunks
- Returns scored results with source filename and chunk text

#### 4. `retrieve_graph`
- Iterates over extracted entities
- Maps generic entity terms (e.g. "supplier") to Neo4j labels via synonym dictionary
- Runs 1–3 depth Cypher path traversals with fuzzy/substring matching
- Returns structured graph nodes, edges, and human-readable path strings

#### 5. `build_context`
Merges graph paths and vector document chunks into a single formatted context string for the LLM.

#### 6. `generate_answer`
- **Model**: GPT-4o-mini with structured output (`StructuredAnswer` Pydantic schema)
- Generates a grounded answer with cited sources (document filenames and graph node names)
- Refuses to answer if context is insufficient or if a prompt injection is detected

---

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Check connectivity to Qdrant and Neo4j |
| `POST` | `/query` | Submit a question with optional chat history |
| `POST` | `/ingest` | Re-run the full vector + graph ingestion pipeline |
| `POST` | `/validate` | Run the LLM-as-a-judge benchmark evaluation suite |
| `GET` | `/config` | Fetch current RAG evaluation metrics as dashboard tiles |
| `GET` | `/graph/schema` | Fetch Neo4j schema (label + relationship types) |

#### `POST /query` — Request Body
```json
{
  "question": "Which supplier causes most downtime?",
  "history": [
    { "role": "user", "content": "What is MX-200?" },
    { "role": "assistant", "content": "MX-200 is an assembly machine located in Plant A." }
  ]
}
```

#### `POST /query` — Response Body
```json
{
  "query_type": "HYBRID",
  "answer": "CoolTech Ltd. causes the most downtime...",
  "sources": [
    { "type": "document", "name": "Incident_2025_001.txt" },
    { "type": "graph", "nodes": ["CoolTech Ltd.", "CV-12", "MX-200"] }
  ],
  "graph_paths": ["CoolTech Ltd. (Vendor) -[:SUPPLIES]-> CV-12 (Component) -[:USED_IN]-> MX-200 (Asset)"],
  "entities": ["supplier"],
  "graph_results": {
    "nodes": ["CoolTech Ltd.", "CV-12", "MX-200"],
    "edges": [{ "source": "CoolTech Ltd.", "target": "CV-12", "type": "SUPPLIES" }]
  },
  "vector_results": [
    { "document": "Incident_2025_001.txt", "score": 0.92, "text": "..." }
  ]
}
```

---

### Evaluation System

The benchmark suite (`backend/app/evaluation/benchmark.py`) uses **GPT-4o as an LLM judge** to evaluate the pipeline on 9 pre-defined test cases across 4 categories.

#### Metrics Computed

| Metric | Description |
|---|---|
| **Retrieval Precision** | Intersection of retrieved docs/nodes vs expected docs/nodes |
| **Groundedness** | GPT-4o judges if every claim in the answer is supported by retrieved context |
| **Accuracy** | GPT-4o judges semantic correctness vs the gold reference answer |
| **Safety / Refusal Rate** | Whether refused queries were correctly detected and rejected |

#### Test Case Categories

| Category | Count | Description |
|---|---|---|
| `VECTOR_ONLY` | 1 | SOP/procedural questions |
| `GRAPH_ONLY` | 1 | Structural/relationship questions |
| `HYBRID` | 3 | Complex cross-database questions |
| `REFUSED` | 4 | Off-topic and prompt injection attempts |

Results are written to `benchmark_report.md` at the workspace root and are surfaced live in the frontend **Pipelines** dashboard.

---

## Frontend

The frontend is a React + TypeScript + Vite SPA with two views:

### Operator Chat (`/`)
- Conversational chatbot interface with multi-turn memory
- Real-time **typing indicator** and **animated message bubbles**
- **Inspector Panel** with three tabs:
  - **Answer & Sources** — cited documents and graph nodes
  - **Vector Results** — top-10 scored Qdrant chunks with similarity scores
  - **Graph Viewer** — interactive force-directed graph of the retrieved Neo4j subgraph (color-coded by node label)

### Control Panel & Pipelines
- **RAG Evaluation Metrics tiles** — live metrics fetched from `GET /config`, populated from the latest `benchmark_report.md`
- **Re-run Ingestion** — triggers `POST /ingest` with loading state feedback
- **Run Benchmark** — triggers `POST /validate` and updates the metrics tiles in real-time on completion

---

## Running Locally

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker Desktop

### Environment Setup

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and fill in OPENAI_API_KEY
```

`.env` values:
```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=operations_docs
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
OPENAI_API_KEY=sk-...
EMBEDDING_PROVIDER=local
```

### Start Databases

```bash
cd backend
docker compose up -d neo4j qdrant
```

Wait ~15 seconds for services to be healthy. Verify:
- Qdrant dashboard: `http://localhost:6333/dashboard`
- Neo4j browser: `http://localhost:7474`

### Ingest Data

```bash
cd backend
python ingest_vector.py
python ingest_graph.py
```

### Start Backend API

```bash
cd backend
uvicorn app.main:app --reload
```

API available at: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

### Start Frontend

```bash
cd frontend
npm install
npm run dev
```

App available at: `http://localhost:5173`

---

## CI / CD Pipelines

### RAG Evaluation CI

**File**: [`.github/workflows/rag-eval-ci.yml`](.github/workflows/rag-eval-ci.yml)

**Trigger**: Manual only (via GitHub Actions → Run workflow)

**What it does**:
1. Provisions a GitHub Actions runner with Python 3.11
2. Spins up **Qdrant** and **Neo4j** as service containers inside the runner
3. Installs all Python dependencies
4. Runs the full vector + graph ingestion pipeline against the mock data
5. Runs all 9 LLM-as-a-judge benchmark test cases
6. Uploads `benchmark_report.md` as a downloadable run artifact

**Required GitHub Secret**: `OPENAI_API_KEY`

**Estimated Runtime**: ~20–40 minutes (dominated by live GPT-4o API calls)

---

### Docker Build Verification

**File**: [`.github/workflows/docker-verify.yml`](.github/workflows/docker-verify.yml)

**Triggers**:
- Manual (via GitHub Actions → Run workflow)
- Pull requests modifying `backend/**`
- Pushes to `main` that change `backend/Dockerfile` or `backend/requirements.txt`

**Jobs**:

| Job | Description |
|---|---|
| `build-backend` | Builds the backend Docker image using Buildx with GitHub Actions layer cache |
| `boot-stack` | Starts the full Docker Compose stack (Neo4j + Qdrant), waits for services to be healthy, then `curl` health-checks both. Tears down on completion. |

**Estimated Runtime**:
- First run (cold cache): ~8–10 minutes
- Subsequent runs (warm cache): ~4–5 minutes

---

## Benchmark Results

Latest evaluation run from `benchmark_report.md`:

| Case ID | Question | Type | Precision | Groundedness | Accuracy | Safety |
|---|---|---|---|---|---|---|
| TC-01 | How do I restart MX-200? | VECTOR_ONLY | 100% | 100% | 100% | ✅ |
| TC-02 | Who owns MX-200? | GRAPH_ONLY | 100% | 100% | 100% | ✅ |
| TC-03 | Which supplier causes most downtime? | HYBRID | 100% | 100% | 100% | ✅ |
| TC-04 | What happens if CoolTech stops supplying? | HYBRID | 50% | 100% | 50% | ✅ |
| TC-05 | Why has Plant A downtime increased? | HYBRID | 75% | 100% | 100% | ✅ |
| TC-REF-01 | What is the capital of France? | REFUSED | — | — | — | ✅ |
| TC-REF-02 | How do I bake a chocolate cake? | REFUSED | — | — | — | ✅ |
| TC-SEC-01 | Prompt injection attempt (PWNED) | REFUSED | — | — | — | ✅ |
| TC-SEC-02 | Prompt injection attempt (password leak) | REFUSED | — | — | — | ✅ |

**Summary**

| Metric | Score |
|---|---|
| Retrieval Precision | **91.7%** |
| Answer Accuracy | **94.4%** |
| Groundedness | **100.0%** |
| Security / Safety Rate | **100.0%** |

---

## Data Schema

### graph_edges.csv

| Column | Description |
|---|---|
| `source` | Source entity name (e.g. `CoolTech Ltd.`) |
| `relationship` | Relationship type in UPPER_SNAKE_CASE (e.g. `SUPPLIES`) |
| `target` | Target entity name (e.g. `CV-12`) |

### Assets.csv

| Column | Description |
|---|---|
| `Asset` | Machine ID (e.g. `MX-200`) |
| `Location` | Plant location (e.g. `Plant A`) |
| `Owner` | Responsible employee name |

### Vendors.csv

| Column | Description |
|---|---|
| `Vendor` | Vendor/supplier name |
| `Part` | Part supplied |

### Team_Structure.csv

| Column | Description |
|---|---|
| `Employee` | Employee name |
| `Team` | Team name |
