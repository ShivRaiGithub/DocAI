import uuid
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.config import Settings, get_settings
from app.schemas import Chunk

# Fixed namespace so the same (document, version, chunk) always gets the same point id.
_NAMESPACE = uuid.UUID("6f1c1c60-4d7e-4b8e-9d57-5b6d1a1e0d01")


def point_id(doc_id: str, version: int, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{doc_id}:{version}:{chunk_index}"))


def _doc_filter(doc_id: str) -> Filter:
    return Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))])


class VectorStore:
    """Stores chunk vectors (plus the chunk text and citation info) in Qdrant."""

    def __init__(self, client: QdrantClient, collection: str, dimension: int):
        self.client = client
        self.collection = collection
        self.dimension = dimension

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
        )
        # an index on document_id makes "all points of document X" lookups fast
        self.client.create_payload_index(
            self.collection, "document_id", PayloadSchemaType.KEYWORD
        )

    def replace_document(
        self,
        doc_id: str,
        version: int,
        filename: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> int:
        """Make the index hold exactly this version's chunks for the document."""
        self.ensure_collection()
        self.delete_document(doc_id)  # drops older versions and stale chunks
        points = [
            PointStruct(
                id=point_id(doc_id, version, c.index),
                vector=vec,
                payload={
                    "document_id": doc_id,
                    "version": version,
                    "filename": filename,
                    "chunk_index": c.index,
                    "page": c.page,
                    "start": c.start,
                    "end": c.end,
                    "text": c.text,
                },
            )
            for c, vec in zip(chunks, vectors, strict=True)
        ]
        if points:
            self.client.upsert(self.collection, points=points)
        return len(points)

    def delete_document(self, doc_id: str) -> None:
        self.ensure_collection()
        self.client.delete(self.collection, points_selector=_doc_filter(doc_id))

    def count(self, doc_id: str) -> int:
        self.ensure_collection()
        return self.client.count(
            self.collection, count_filter=_doc_filter(doc_id), exact=True
        ).count

    def indexed_version(self, doc_id: str) -> int | None:
        self.ensure_collection()
        points, _ = self.client.scroll(
            self.collection, scroll_filter=_doc_filter(doc_id), limit=1, with_payload=["version"]
        )
        return points[0].payload["version"] if points else None


@lru_cache
def _client_for(url: str) -> QdrantClient:
    return QdrantClient(url=url)


def get_vector_store(settings: Annotated[Settings, Depends(get_settings)]) -> VectorStore:
    return VectorStore(
        _client_for(settings.qdrant_url), settings.qdrant_collection, settings.embedding_dimension
    )


VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
