# Semantic Search Based LLM Routing: Case Study

A Python case study in semantic search based LLM routing. User queries are classified into 5 business verticals (HR, Legal, Compliance, Sales, Engineering) and routed to the best LLM provider from 5 options (Anthropic, DeepSeek, Gemini, Llama, Kimi). Uses HuggingFace all-MiniLM-L6-v2 embeddings and cosine similarity through a two-stage pipeline. Built with FastAPI, NumPy, and Docker Compose.

Every routing decision returns the full trace of similarity scores for educational transparency. No real LLM API calls are made. All responses are generated from per-provider, per-vertical mock templates with simulated latency.

---

## Quick Start

```bash
# Option 1: Docker (recommended)
docker compose up --build -d
curl localhost:8000/health

# Option 2: CLI (offline, no server)
python -m src.cli "draft a non-disclosure agreement"

# Option 3: Full demo script (builds, runs, tests, tears down)
bash demo.sh
```

---

## Architecture Overview

```
User Query
    |
    v
[Embedding Service] -- all-MiniLM-L6-v2 --> 384-dim vector
    |
    v
[Vector Store] -- cosine similarity (np.dot)
    |
    v
[Stage 1: Vertical Classifier] -- 5 vertical scores
    |
    +-- threshold >= 0.20? --No--> Fallback response
    |
    Yes
    |
    v
[Stage 2: Provider Selector] -- 5 provider scores (within chosen vertical)
    |
    v
[Mock LLM Service] -- per-provider response template + simulated latency
    |
    v
[RouteResponse] -- full trace: query, vertical, provider, all scores, response, latency
```

---

## Project Structure

```
├── src/
│   ├── main.py                  # FastAPI server with /health and /route
│   ├── cli.py                   # Standalone CLI demo (offline)
│   ├── config.py                # All configuration constants
│   ├── models.py                # Pydantic v2 request/response models
│   ├── embedding_service.py     # HuggingFace embedding with mean pooling
│   ├── vector_store.py          # In-memory NumPy vector store
│   ├── router.py                # Two-stage semantic router
│   ├── llm_service.py           # Mock LLM with 25 response templates
│   └── data/
│       ├── verticals.py         # 5 business verticals with exemplar queries
│       └── providers.py         # 5 LLM providers with capability profiles
├── Dockerfile                   # Multi-stage build with model pre-cache
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── demo.sh                      # End-to-end demo script
└── README.md
```

---

## How It Works

### Stage 1: Vertical Classification

The user's query is embedded using all-MiniLM-L6-v2, producing a 384-dimensional L2-normalized vector. This vector is compared via cosine similarity (implemented as `np.dot()` since all embeddings are unit-norm) against 5 vertical centroids. Each centroid is computed from the vertical's description plus exemplar queries.

The 5 vertical scores are sorted. If the top score is below the configurable threshold (0.20), the router returns a fallback response immediately. This prevents routing unrelated queries (like "what is the weather today") to a forced vertical.

### Stage 2: Provider Selection

Within the chosen vertical, the same query embedding is compared against 5 provider capability descriptions. Providers are described in terms of their real-world strengths (e.g., Anthropic excels at nuanced legal reasoning, DeepSeek at code generation). The best matching provider is selected.

### Response Generation

A mock LLM response is generated using one of 25 response templates (5 providers x 5 verticals). Each provider writes in a distinct style:

| Provider | Style |
|----------|-------|
| Anthropic | Verbose, careful language with caveats |
| DeepSeek | Technical, structured, code-like |
| Gemini | Balanced, informative, bullet points |
| Llama | Direct, concise, practical |
| Kimi | Thorough, detail-oriented with contextual analysis |

Simulated latency (100-500ms via `asyncio.sleep(random.uniform(0.1, 0.5))`) is injected before returning.

### Full Trace

Every response includes all similarity scores:

- `all_vertical_scores`: 5 keys (HR, Legal, Compliance, Sales, Engineering), all computed
- `all_provider_scores`: 5 keys (Anthropic, DeepSeek, Gemini, Llama, Kimi), all computed
- `vertical_confidence`: the winning vertical's score
- `provider_confidence`: the winning provider's score
- `fallback_used`: whether the threshold was not met

---

## API Reference

### `GET /health`

Returns service status and seed data counts.

```bash
curl localhost:8000/health
```

```json
{
  "status": "ok",
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "verticals_count": 5,
  "providers_count": 5
}
```

### `POST /route`

Routes a query through the two-stage pipeline.

```bash
curl -X POST localhost:8000/route \
  -H 'Content-Type: application/json' \
  -d '{"query": "draft a non-disclosure agreement"}'
```

```json
{
  "query": "draft a non-disclosure agreement",
  "vertical": "Legal",
  "vertical_confidence": 0.492,
  "all_vertical_scores": {
    "HR": 0.123,
    "Legal": 0.492,
    "Compliance": 0.341,
    "Sales": 0.098,
    "Engineering": 0.087
  },
  "provider": "Anthropic",
  "provider_confidence": 0.208,
  "all_provider_scores": {
    "Anthropic": 0.208,
    "DeepSeek": 0.165,
    "Gemini": 0.182,
    "Llama": 0.171,
    "Kimi": 0.194
  },
  "response": "I want to provide a thorough analysis of your legal question...",
  "latency_ms": 234.5,
  "fallback_used": false
}
```

#### Example queries for each vertical

```bash
# HR
curl -X POST localhost:8000/route -H 'Content-Type: application/json' \
  -d '{"query": "hire a software engineer"}'

# Legal
curl -X POST localhost:8000/route -H 'Content-Type: application/json' \
  -d '{"query": "draft a non-disclosure agreement"}'

# Compliance
curl -X POST localhost:8000/route -H 'Content-Type: application/json' \
  -d '{"query": "how do I become GDPR compliant"}'

# Sales
curl -X POST localhost:8000/route -H 'Content-Type: application/json' \
  -d '{"query": "write a cold email for enterprise sales"}'

# Engineering
curl -X POST localhost:8000/route -H 'Content-Type: application/json' \
  -d '{"query": "design a microservices architecture"}'
```

#### Fallback (out-of-scope query)

```bash
curl -X POST localhost:8000/route -H 'Content-Type: application/json' \
  -d '{"query": "what is the weather today"}'
```

```json
{
  "query": "what is the weather today",
  "vertical": "unknown",
  "vertical_confidence": 0.045,
  "all_vertical_scores": {
    "HR": 0.045,
    "Legal": 0.032,
    "Compliance": 0.028,
    "Sales": 0.041,
    "Engineering": 0.039
  },
  "provider": "none",
  "provider_confidence": 0.0,
  "all_provider_scores": {},
  "response": "I'm unable to route this query to a specific vertical. Please rephrase your request with more context about HR, Legal, Compliance, Sales, or Engineering topics.",
  "latency_ms": 0.0,
  "fallback_used": true
}
```

---

## Configuration

All configuration lives in `src/config.py` as plain Python constants. No YAML, no `.env` files.

| Constant | Value | Description |
|----------|-------|-------------|
| `EMBEDDING_MODEL_NAME` | `"sentence-transformers/all-MiniLM-L6-v2"` | HuggingFace model for embeddings |
| `SIMILARITY_THRESHOLD` | `0.20` | Minimum cosine similarity for vertical match |
| `TOP_K_VERTICALS` | `1` | Number of top verticals to return |
| `TOP_K_PROVIDERS` | `1` | Number of top providers to return |
| `FALLBACK_VERTICAL` | `"general"` | Label when no vertical scores above threshold |
| `FALLBACK_RESPONSE` | see config.py | Message returned on fallback |
| `MOCK_LATENCY_MIN` | `0.1` | Minimum simulated latency (seconds) |
| `MOCK_LATENCY_MAX` | `0.5` | Maximum simulated latency (seconds) |
| `API_HOST` | `"0.0.0.0"` | Server bind address |
| `API_PORT` | `8000` | Server port |

---

## Seed Data

### Verticals (5)

Each vertical is defined by a `VerticalDef` dataclass with a description and 5 exemplar queries:

| Vertical | Focus Area | Example Query |
|----------|-----------|---------------|
| HR | Employee management, recruitment, payroll | "Write a job description for a senior software engineer position" |
| Legal | Contract law, IP, dispute resolution | "Draft a non-disclosure agreement" |
| Compliance | Data privacy, regulations, audits | "What steps do I need to take to become GDPR compliant?" |
| Sales | Lead generation, CRM, forecasting | "Write a cold email template for enterprise SaaS prospects" |
| Engineering | Architecture, DevOps, testing | "Design a scalable microservices architecture" |

### Providers (5)

Each provider is defined by a `ProviderDef` dataclass with a capability description and per-vertical strength scores (0-1):

| Provider | Best Vertical | Strength |
|----------|--------------|----------|
| Anthropic | Legal | 0.95 |
| DeepSeek | Engineering | 0.95 |
| Gemini | Sales | 0.85 |
| Llama | Sales | 0.80 |
| Kimi | Compliance | 0.80 |

---

## Docker

Multi-stage Docker build for minimal production images:

- **Builder stage** (`python:3.11-slim`): Installs all Python dependencies including CPU-only PyTorch from the official CPU index. Pre-caches the all-MiniLM-L6-v2 model via `AutoModel.from_pretrained()` so it is baked into the image at build time.
- **Runtime stage** (`python:3.11-slim`): Copies installed site-packages and HuggingFace cache from the builder. Runs as a non-root user (`appuser`). Exposes port 8000.

```bash
# Build and start
docker compose up --build -d

# Stop
docker compose down
```

---

## Tech Stack

- **Python 3.11+**: target runtime
- **FastAPI**: async API server
- **HuggingFace Transformers**: model loading and inference
- **PyTorch (CPU)**: tensor operations for embedding computation
- **NumPy**: cosine similarity via dot product
- **Pydantic v2**: request/response validation
- **Uvicorn**: ASGI server
- **Docker Compose**: container orchestration

---

## License

MIT
