"""Two-stage semantic router: query → vertical → provider → response."""

import importlib

from src.models import RouteResponse


class SemanticRouter:
    """Routes a query to the best vertical and provider via two-stage similarity search.

    Stage 1 filters ``vertical:*`` scores from the vector store to pick the best
    vertical.  If the top score is below the confidence threshold a fallback
    response is returned immediately.  Stage 2 filters ``provider:*`` scores to
    pick the best provider within the chosen vertical, then delegates generation
    to the LLM service.
    """

    def __init__(self, vector_store, llm_service, config_module=None):
        self._vector_store = vector_store
        self._llm_service = llm_service

        if config_module is None:
            config_module = importlib.import_module("src.config")

        self._threshold = config_module.SIMILARITY_THRESHOLD
        self._fallback_message = config_module.FALLBACK_RESPONSE

    async def route(self, query: str) -> RouteResponse:
        scores = self._vector_store.search_with_scores(query)

        # ── Stage 1: vertical routing ──────────────────────────────────────
        vertical_scores = {
            k.removeprefix("vertical:"): v
            for k, v in scores.items()
            if k.startswith("vertical:")
        }
        sorted_verticals = sorted(
            vertical_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_vertical, top_vertical_score = (
            sorted_verticals[0] if sorted_verticals else ("unknown", 0.0)
        )

        # Fallback: confidence too low → return immediately.
        if not sorted_verticals or top_vertical_score < self._threshold:
            return RouteResponse(
                query=query,
                vertical="unknown",
                vertical_confidence=top_vertical_score,
                all_vertical_scores=vertical_scores,
                provider="none",
                provider_confidence=0.0,
                all_provider_scores={},
                response=self._fallback_message,
                latency_ms=0.0,
                fallback_used=True,
            )

        # ── Stage 2: provider routing ──────────────────────────────────────
        provider_scores = {
            k.removeprefix("provider:"): v
            for k, v in scores.items()
            if k.startswith("provider:")
        }
        sorted_providers = sorted(
            provider_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_provider, top_provider_score = (
            sorted_providers[0] if sorted_providers else ("none", 0.0)
        )

        # Generate response via the chosen provider.
        response_text, latency_ms = await self._llm_service.generate(
            top_provider, top_vertical, query
        )

        return RouteResponse(
            query=query,
            vertical=top_vertical,
            vertical_confidence=top_vertical_score,
            all_vertical_scores=vertical_scores,
            provider=top_provider,
            provider_confidence=top_provider_score,
            all_provider_scores=provider_scores,
            response=response_text,
            latency_ms=latency_ms,
            fallback_used=False,
        )
