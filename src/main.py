"""FastAPI application for semantic LLM routing."""

import uvicorn
from fastapi import FastAPI

from src.config import API_HOST, API_PORT, EMBEDDING_MODEL_NAME
from src.data.providers import get_all_providers
from src.data.verticals import get_all_verticals
from src.embedding_service import EmbeddingService
from src.llm_service import MockLLMService
from src.models import HealthResponse, RouteRequest, RouteResponse
from src.router import SemanticRouter
from src.vector_store import VectorStore

app = FastAPI(title="Semantic LLM Router", version="0.1.0")

# Global service instances (populated on startup)
router: SemanticRouter | None = None
llm_service: MockLLMService | None = None


@app.on_event("startup")
async def startup() -> None:
    """Initialize all services and seed the vector store."""
    global router, llm_service

    # 1. Embedding service
    emb_svc = EmbeddingService()

    # 2. Vector store
    store = VectorStore(emb_svc)

    # 3. Seed verticals (description + first 3 exemplar queries)
    for v in get_all_verticals():
        exemplar_text = v.description + " " + " ".join(v.exemplar_queries[:3])
        store.add_item(f"vertical:{v.label}", exemplar_text)

    # 4. Seed providers (description only)
    for p in get_all_providers():
        store.add_item(f"provider:{p.label}", p.description)

    # 5. LLM service and router
    llm_service = MockLLMService()
    router = SemanticRouter(store, llm_service)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return health check with model name and seed counts."""
    return HealthResponse(
        status="ok",
        model=EMBEDDING_MODEL_NAME,
        verticals_count=5,
        providers_count=5,
    )


@app.post("/route", response_model=RouteResponse)
async def route(request: RouteRequest) -> RouteResponse:
    """Route a user query to the best vertical and LLM provider."""
    return await router.route(request.query)


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
