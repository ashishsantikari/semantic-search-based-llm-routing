import numpy as np


class VectorStore:
    """In-memory vector store using NumPy cosine similarity."""

    def __init__(self, embedding_service):
        self._embedding_service = embedding_service
        self._items: dict[str, np.ndarray] = {}

    def add_item(self, label: str, text: str) -> None:
        embedding = self._embedding_service.embed(text)
        self._items[label] = embedding

    def add_items(self, items: dict[str, str]) -> None:
        for label, text in items.items():
            self.add_item(label, text)

    def search(self, query: str, top_k: int = 1) -> list[tuple[str, float]]:
        if not self._items:
            return []

        query_embedding = self._embedding_service.embed(query)
        scores = {label: float(np.dot(query_embedding, emb)) for label, emb in self._items.items()}
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]

    def search_with_scores(self, query: str) -> dict[str, float]:
        if not self._items:
            return {}

        query_embedding = self._embedding_service.embed(query)
        return {label: float(np.dot(query_embedding, emb)) for label, emb in self._items.items()}
