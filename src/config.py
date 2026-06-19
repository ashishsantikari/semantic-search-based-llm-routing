"""
Configuration module for the semantic search-based LLM router.

This module contains all constants used across the routing system,
including model names, similarity thresholds, API server settings,
and fallback responses. All values are plain Python globals —
no Pydantic, no YAML, no .env files.
"""

# ── Embedding model ──────────────────────────────────────────────────────────
# Sentence-transformer model used to embed queries and vertical descriptions
# for semantic similarity matching.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── Routing thresholds ───────────────────────────────────────────────────────
# Minimum cosine-similarity score required to consider a vertical match.
# Values range from 0 (everything matches) to 1 (exact match only).
SIMILARITY_THRESHOLD = 0.20

# Number of top-matching verticals to consider when routing a query.
TOP_K_VERTICALS = 1

# Number of top-matching LLM providers to select within the chosen vertical.
TOP_K_PROVIDERS = 1

# ── Fallbacks ─────────────────────────────────────────────────────────────────
# Vertical name to use when no vertical exceeds the similarity threshold.
FALLBACK_VERTICAL = "general"

# User-facing message returned when the router cannot confidently assign a
# vertical to the incoming query.
FALLBACK_RESPONSE = (
    "I'm unable to route this query to a specific vertical. "
    "Please rephrase your request with more context about HR, Legal, "
    "Compliance, Sales, or Engineering topics."
)

# ── Mock latency simulation ───────────────────────────────────────────────────
# Bounds (in seconds) for random latency injected during mock embedding /
# routing calls, simulating real inference time.
MOCK_LATENCY_MIN = 0.1
MOCK_LATENCY_MAX = 0.5

# ── API server ───────────────────────────────────────────────────────────────
# Host and port for the FastAPI / Flask server that exposes the router
# endpoint.
API_HOST = "0.0.0.0"
API_PORT = 8000
