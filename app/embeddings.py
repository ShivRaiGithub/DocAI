from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.config import Settings, get_settings


class Embedder:
    """Turns text into vectors using a local sentence-transformers model.

    The model (~90 MB) is downloaded on first use and loaded lazily, so
    importing this module or starting the app stays fast.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True makes every vector length 1, so the dot
        # product of two vectors *is* their cosine similarity.
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()


@lru_cache
def _embedder_for(model_name: str) -> Embedder:
    return Embedder(model_name)


def get_embedder(settings: Annotated[Settings, Depends(get_settings)]) -> Embedder:
    return _embedder_for(settings.embedding_model)


EmbedderDep = Annotated[Embedder, Depends(get_embedder)]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Dot product of two unit-length vectors (see Embedder.embed)."""
    return sum(x * y for x, y in zip(a, b))
