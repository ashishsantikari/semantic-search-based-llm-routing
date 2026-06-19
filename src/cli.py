#!/usr/bin/env python3
"""Standalone CLI demo for the semantic LLM router.

Usage:
    python src/cli.py "your query here"
    python src/cli.py  (defaults to "hire a software engineer")
"""

import asyncio
import sys

from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.router import SemanticRouter
from src.llm_service import MockLLMService
from src.data.verticals import get_all_verticals
from src.data.providers import get_all_providers


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else "hire a software engineer"

    # 1. Embedding service
    emb_svc = EmbeddingService()

    # 2. Vector store
    store = VectorStore(emb_svc)

    # 3. Seed verticals (description + all exemplar queries)
    for v in get_all_verticals():
        store.add_item(
            f"vertical:{v.label}",
            v.description + " " + " ".join(v.exemplar_queries),
        )

    # 4. Seed providers (description only)
    for p in get_all_providers():
        store.add_item(f"provider:{p.label}", p.description)

    # 5. Router
    router = SemanticRouter(store, MockLLMService())

    # 6. Route the query
    result = await router.route(query)

    # 7. Print results
    print(f"Query: {result.query}")
    print(f"Vertical: {result.vertical} (confidence: {result.vertical_confidence:.3f})")
    print(f"Provider: {result.provider} (confidence: {result.provider_confidence:.3f})")
    print(f"Fallback: {result.fallback_used}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    print()
    print("All vertical scores:")
    for v, s in sorted(result.all_vertical_scores.items(), key=lambda x: -x[1]):
        print(f"  {v}: {s:.3f}")
    if result.all_provider_scores:
        print()
        print("All provider scores:")
        for p, s in sorted(result.all_provider_scores.items(), key=lambda x: -x[1]):
            print(f"  {p}: {s:.3f}")
    print()
    print("=== Generated Response ===")
    print(result.response)


if __name__ == "__main__":
    asyncio.run(main())
