"""
Pydantic v2 request/response models for the semantic LLM router API.
"""

from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    """Request model for the semantic LLM route endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="User query to route",
    )


class RouteResponse(BaseModel):
    """Response model for the semantic LLM route endpoint."""

    query: str
    vertical: str
    vertical_confidence: float
    all_vertical_scores: dict[str, float]
    provider: str
    provider_confidence: float
    all_provider_scores: dict[str, float]
    response: str
    latency_ms: float
    fallback_used: bool


class HealthResponse(BaseModel):
    """Response model for the health check endpoint."""

    status: str = "ok"
    model: str = "all-MiniLM-L6-v2"
    verticals_count: int = 5
    providers_count: int = 5
