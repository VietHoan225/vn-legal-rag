"""Vector store abstraction and Qdrant implementation."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import logging
from typing import Any, Sequence
import uuid
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Standardized search hit returned by VectorStore."""

    chunk_id: str
    doc_id: int
    article_id: str 
    article_number: int
    title: str
    category: str
    text: str
    score: float
    clause_number: int | None = None

    def location(self) -> str:
        """Human-readable citation, e.g. 'Điều 6, Khoản 2' or 'Điều 6'."""
        if self.clause_number is not None:
            return f"Điều {self.article_number}, Khoản {self.clause_number}"
        return f"Điều {self.article_number}"

    def to_dict(self) -> dict[str, Any]:
        """Convert search result to dictionary."""
        return asdict(self)


class VectorStore(ABC):
    """Abstract interface for vector database implementations."""

    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        recreate: bool = False,
    ) -> None:
        """Create or initialize a vector collection."""
        pass

    @abstractmethod
    def upsert_chunks(
        self,
        collection_name: str,
        chunks: Sequence[dict[str, Any]],
        vectors: np.ndarray | Sequence[Sequence[float]],
        batch_size: int = 500,
    ) -> int:
        """Upsert chunk payloads and vectors into the collection idempotently."""
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        limit: int = 4,
        category: str | None = None,
    ) -> list[SearchResult]:
        """Search top-k most similar chunks for a given query vector."""
        pass

    @abstractmethod
    def count(self, collection_name: str) -> int:
        """Return total number of vectors in collection."""
        pass


class QdrantVectorStore(VectorStore):
    """Qdrant implementation of VectorStore."""

    def __init__(self, url: str | None = None, client: QdrantClient | None = None) -> None:
        self.url = url or settings.qdrant_url
        self.client = client or QdrantClient(url=self.url)

    @staticmethod
    def chunk_id_to_uuid(chunk_id: str) -> str:
        """Generate a deterministic UUID from chunk_id for idempotency."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        recreate: bool = False,
    ) -> None:
        """Create Qdrant collection if not exists, or recreate if requested."""
        collections_response = self.client.get_collections()
        existing = [c.name for c in collections_response.collections]

        if collection_name in existing:
            if recreate:
                logger.info("Dropping existing collection '%s'...", collection_name)
                self.client.delete_collection(collection_name=collection_name)
            else:
                logger.info("Collection '%s' already exists. Skipping creation.", collection_name)
                return

        logger.info(
            "Creating collection '%s' (dim=%d, distance=COSINE)...",
            collection_name,
            vector_size,
        )
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: Sequence[dict[str, Any]],
        vectors: np.ndarray | Sequence[Sequence[float]],
        batch_size: int = 500,
    ) -> int:
        """Upsert chunks and vectors in batches using deterministic UUIDs."""
        total = len(chunks)
        if total != len(vectors):
            raise ValueError(
                f"Chunks count ({total}) does not match vectors count ({len(vectors)})"
            )

        upserted = 0
        for i in range(0, total, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_vectors = vectors[i : i + batch_size]

            points = []
            for chunk, vec in zip(batch_chunks, batch_vectors):
                point_id = self.chunk_id_to_uuid(str(chunk["chunk_id"]))
                payload = {
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "article_id": chunk["article_id"],
                    "article_number": chunk["article_number"],
                    "title": chunk["title"],
                    "category": chunk["category"],
                    "text": chunk["text"],
                    "clause_number": chunk.get("clause_number"),
                }
                vector = vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            self.client.upsert(collection_name=collection_name, points=points)
            upserted += len(points)
            logger.info("Upserted %d/%d chunks into '%s'", upserted, total, collection_name)

        return upserted

    def search(
        self,
        collection_name: str,
        query_vector: Sequence[float],
        limit: int = 4,
        category: str | None = None,
    ) -> list[SearchResult]:
        """Search top-k most similar chunks with optional category filter."""
        query_filter = None
        if category:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value=category),
                    )
                ]
            )

        # qdrant-client search / query_points
        res = self.client.query_points(
            collection_name=collection_name,
            query=list(query_vector),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )

        results: list[SearchResult] = []
        for hit in res.points:
            p = hit.payload or {}
            results.append(
                SearchResult(
                    chunk_id=str(p.get("chunk_id", "")),
                    doc_id=int(p.get("doc_id", 0)),
                    article_id=str(p.get("article_id", "")),
                    article_number=int(p.get("article_number", 0)),
                    title=str(p.get("title", "")),
                    category=str(p.get("category", "")),
                    text=str(p.get("text", "")),
                    score=float(hit.score),
                    clause_number=p.get("clause_number"),
                )
            )
        return results

    def count(self, collection_name: str) -> int:
        """Return total point count in collection."""
        res = self.client.count(collection_name=collection_name)
        return res.count
