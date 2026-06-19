# Architecture: Semantic Search Based LLM Routing

## Overview

This project implements a two-stage semantic search routing pipeline. User queries are classified into one of 5 business verticals (HR, Legal, Compliance, Sales, Engineering), then routed to the best LLM provider from 5 options (Anthropic, DeepSeek, Gemini, Llama, Kimi). The decision is driven entirely by cosine similarity on HuggingFace all-MiniLM-L6-v2 embeddings, with no ML model training, no real LLM API calls, and no external dependencies beyond standard PyData libraries.

---

## Two-Stage Routing Algorithm

### Stage 1: Vertical Classification

The router performs a single `search_with_scores()` call against the vector store, which returns scores for every stored item regardless of type. Items are stored with label prefixes:

- Verticals use the prefix `vertical:` (e.g., `vertical:HR`, `vertical:Legal`)
- Providers use the prefix `provider:` (e.g., `provider:Anthropic`, `provider:DeepSeek`)

The router filters the flat score dictionary by prefix to separate vertical and provider scores:

```python
# One call gets everything
all_scores = vector_store.search_with_scores(query)

# Stage 1: extract vertical scores
vertical_scores = {
    k.removeprefix("vertical:"): v
    for k, v in all_scores.items()
    if k.startswith("vertical:")
}
top_vertical, top_score = max(vertical_scores.items(), key=lambda x: x[1])

# Threshold check
if top_score < SIMILARITY_THRESHOLD:
    return fallback_response()  # confidence too low
```

If the top vertical score is below `SIMILARITY_THRESHOLD` (0.20), the router returns a fallback response immediately. This prevents forced routing of unrelated queries.

### Stage 2: Provider Selection

Within the chosen vertical, the same query embedding is compared against provider descriptions:

```python
# Stage 2: extract provider scores (same query, same vector store)
provider_scores = {
    k.removeprefix("provider:"): v
    for k, v in all_scores.items()
    if k.startswith("provider:")
}
top_provider, top_provider_score = max(provider_scores.items(), key=lambda x: x[1])
```

There is no secondary threshold check at the provider level. If a vertical was chosen, a provider is always selected within it.

### Response Assembly

The winning provider's template is looked up in a 5x5 grid (5 providers x 5 verticals = 25 templates), the query is interpolated into the template, simulated latency is applied, and the full `RouteResponse` is returned.

### Simplified Pseudo-Code

```
function route(query):
    all_scores = vector_store.search_with_scores(query)

    # Stage 1
    vert_scores = filter(prefix="vertical:", all_scores)
    best_vert, best_vert_score = top(vert_scores)

    if best_vert_score < 0.20:
        return { fallback_used: true, ... }

    # Stage 2
    prov_scores = filter(prefix="provider:", all_scores)
    best_prov, best_prov_score = top(prov_scores)

    # Generate response
    response, latency = llm_service.generate(best_prov, best_vert, query)

    return {
        vertical: best_vert,
        provider: best_prov,
        all_vertical_scores: vert_scores,
        all_provider_scores: prov_scores,
        response: response,
        latency_ms: latency,
        fallback_used: false
    }
```

---

## Embedding Pipeline

The `EmbeddingService` class wraps HuggingFace's `AutoModel` and `AutoTokenizer` directly, bypassing the `sentence-transformers` library. This keeps dependencies minimal. Just `transformers` plus `torch`.

### Model

- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Output dimension**: 384
- **Context length**: 128 tokens (truncated)
- **Device**: CPU (`torch.device("cpu")`)

### Mean Pooling

The raw transformer output is a tensor of shape `(1, seq_len, 384)` where each token position has a 384-dimensional vector. Not all tokens are meaningful. Padding tokens should contribute zero to the final embedding. Mean pooling handles this:

```python
token_embeddings = outputs.last_hidden_state  # (1, seq_len, 384)
attention_mask = inputs["attention_mask"]      # (1, seq_len)

# Expand mask and apply
mask = attention_mask.unsqueeze(-1).float()    # (1, seq_len, 1)
masked_embeddings = token_embeddings * mask    # zero out padding

# Sum over tokens
summed = masked_embeddings.sum(dim=1)          # (1, 384)
counts = mask.sum(dim=1)                       # (1, 1)

# Divide by count of non-padding tokens
mean_pooled = summed / counts                  # (1, 384)
```

### L2 Normalization

After mean pooling, the vector is L2-normalized so that cosine similarity reduces to a simple dot product (`np.dot(a, b)` when `||a|| = ||b|| = 1`):

```python
norm = torch.norm(mean_pooled, p=2, dim=1, keepdim=True)
normalized = mean_pooled / norm
```

### Empty String Handling

An empty query returns a zero vector of shape `(384,)` to avoid division-by-zero in normalization.

---

## Vector Store Internals

The `VectorStore` class is deliberately minimal. No FAISS, no Annoy, no sklearn. Just Python dicts and NumPy.

### Storage

```python
class VectorStore:
    def __init__(self, embedding_service):
        self._embedding_service = embedding_service
        self._items: dict[str, np.ndarray] = {}  # label -> 384-dim embedding
```

Items are stored with label prefixes so a single `search_with_scores()` call returns scores for both verticals and providers simultaneously:

```python
store.add_item("vertical:HR", hr_description + " " + " ".join(hr_exemplars))
store.add_item("provider:Anthropic", anthropic_description)
```

### Cosine Similarity

Since all stored embeddings and query embeddings are L2-normalized, cosine similarity is exactly `np.dot()`:

```python
def search_with_scores(self, query: str) -> dict[str, float]:
    query_embedding = self._embedding_service.embed(query)
    return {
        label: float(np.dot(query_embedding, emb))
        for label, emb in self._items.items()
    }
```

### How Centroids Are Computed

Vertical embeddings are not stored as individual exemplars. Instead, the description plus first 3 exemplar queries (in the server) or all 5 exemplar queries (in the CLI) are concatenated into a single text string and embedded once:

```python
# main.py: description + first 3 exemplars
exemplar_text = v.description + " " + " ".join(v.exemplar_queries[:3])
store.add_item(f"vertical:{v.label}", exemplar_text)

# cli.py: description + all 5 exemplars
exemplar_text = v.description + " " + " ".join(v.exemplar_queries)
store.add_item(f"vertical:{v.label}", exemplar_text)
```

This creates a single "centroid" embedding that represents the semantic center of that vertical. Provider descriptions are embedded directly. They are short capability profiles, not computed centroids.

---

## Similarity Threshold Tuning

### Why 0.20?

The threshold was empirically determined with all-MiniLM-L6-v2 embeddings. Initial testing with 0.30 caused legitimate queries to fall back:

| Query | HR Score (0.30 threshold) | HR Score (0.20 threshold) |
|-------|--------------------------|--------------------------|
| "hire a software engineer" | 0.218 (fallback) | 0.218 (passes) |
| "draft a non-disclosure agreement" | 0.492 (passes) | 0.492 (passes) |
| "what is the weather today" | 0.045 (fallback) | 0.045 (fallback) |

With all-MiniLM-L6-v2, cosine similarity scores for semantically related text tend to cluster in the 0.15-0.50 range (not the 0.70-0.95 range seen with larger models like OpenAI ada-002). A 0.30 threshold falsely rejected relevant queries. Lowering to 0.20 was the minimum viable threshold that still caught truly unrelated queries.

### Trade-Off

A lower threshold means more false positives. Queries that are vaguely business-related but don't clearly belong to any vertical will still be routed somewhere. The project favors functional routing over strict rejection. For a production system, this threshold would need tuning against your specific domain and embedding model.

---

## Mock LLM Service

### Template Grid

The `MockLLMService` contains a dictionary of 25 response templates, keyed by `(provider, vertical)` tuple:

```python
_TEMPLATES: dict[tuple[str, str], str] = {}
_TEMPLATES[("Anthropic", "Legal")] = "I want to provide a thorough analysis..."
_TEMPLATES[("DeepSeek", "Engineering")] = "Here is a structured breakdown..."
# ... 23 more templates
```

### Per-Provider Writing Styles

Each provider's templates follow a distinct tone and structure to simulate real differences in LLM output:

| Provider | Style Characteristics |
|----------|----------------------|
| **Anthropic** | Verbose, hedging language with caveats. Opens with "I want to provide a thorough analysis..." |
| **DeepSeek** | Technical and structured. Uses markdown headings, code blocks, tables, pseudo-code snippets |
| **Gemini** | Balanced and informative. Uses bullet-point lists with a summary sentence |
| **Llama** | Direct and concise. Short paragraphs with numbered action items |
| **Kimi** | Thorough and contextual. Adds a "Context:" section before the detailed breakdown |

### Latency Simulation

Each generation call injects random latency via `asyncio.sleep()`:

```python
latency_seconds = random.uniform(MOCK_LATENCY_MIN, MOCK_LATENCY_MAX)  # 0.1 - 0.5s
await asyncio.sleep(latency_seconds)
```

This simulates real LLM inference time without using threading or async tasks. The actual latency value is returned in the response as `latency_ms`.

### Missing Template Fallback

If a (provider, vertical) pair has no template (shouldn't happen with the current 5x5 grid, but handled defensively):

```python
template = _TEMPLATES.get((provider, vertical))
if template is None:
    response = f"Response from {provider} regarding your {vertical} query..."
else:
    response = template.format(query=query)
```

---

## Docker Setup

### Multi-Stage Build

The Dockerfile uses two stages to minimize the runtime image size:

**Stage 1: Builder** (`python:3.11-slim`):

1. Installs build dependencies (`gcc` for compiling native extensions)
2. Installs CPU-only PyTorch from the official CPU wheel index
3. Installs all Python dependencies from `requirements.txt`
4. Pre-caches the embedding model by running `AutoModel.from_pretrained()` and `AutoTokenizer.from_pretrained()`. This ensures the model is baked into the image at build time and does not need to download at container startup.

```dockerfile
RUN python -c "\
from transformers import AutoModel, AutoTokenizer; \
AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')"
```

**Stage 2: Runtime** (`python:3.11-slim`):

1. Copies installed site-packages and binaries from the builder
2. Copies the HuggingFace model cache (`/root/.cache/huggingface`) from the builder
3. Copies only the application source code (`src/`)
4. Creates a non-root user (`appuser`) for security
5. Exposes port 8000 and runs `uvicorn src.main:app`

```dockerfile
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
COPY src/ src/
```

### Docker Compose

```yaml
services:
  router:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
```

Single service, single port mapping, no health check, no override files.

### CPU-Only PyTorch

The Dockerfile installs PyTorch from the CPU-only wheel index to avoid pulling CUDA libraries:

```
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.5.0,<2.6.0"
```

---

## Design Decisions

### `transformers` Directly Instead of `sentence-transformers`

The `sentence-transformers` library adds convenience wrappers (like `encode()` and built-in pooling) but also pulls in additional dependencies. Using `AutoModel` and `AutoTokenizer` directly from `transformers` with manual mean pooling achieves the same result with fewer transitive dependencies and more explicit, educational code.

### No ABCs or Interfaces

The vector store, embedding service, and LLM service are concrete classes used directly. No abstract base classes, no protocols, no factory patterns. This is a deliberate choice for a case study. Abstraction layers would add complexity without benefit for a project this size.

### No Real LLM API Calls

All responses are generated from templates. This keeps the project self-contained (no API keys needed), zero-cost to run, and deterministic for demonstration purposes. The focus is on the routing logic, not the LLM output quality.

### In-Memory Only

The vector store is a plain Python dict. No SQLite, no Redis, no file persistence. On restart, all embeddings are recomputed from the seed data. This is acceptable because:
- Seed data is small (10 items total: 5 verticals + 5 providers)
- Model loading is the bottleneck, not embedding computation
- Persistence adds complexity with no benefit for a demo

### Label Prefix Filtering Instead of Separate Stores

A single `search_with_scores()` call returns all scores. The router separates vertical and provider scores by filtering on the `vertical:` and `provider:` label prefixes. This avoids the need for multiple vector store instances or a separate `category` parameter on the search method. The trade-off is that every query computes scores against all 10 items regardless of type. That is trivial at this scale.

### Single `config.py` Instead of YAML/Env Files

All configuration is in one Python module with plain module-level constants. No YAML parsing, no `.env` loading, no Pydantic Settings. This keeps the config immediately readable, importable, and type-checkable by the Python interpreter. Environment-specific configuration is unnecessary for a single-deployment case study.
