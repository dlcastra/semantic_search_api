import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from src.core.settings import settings

client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_HTTP_PORT, grpc_port=settings.QDRANT_GRPC_PORT)


def collection_exists() -> bool:
    """Check if the collection exists in Qdrant."""

    return client.collection_exists(settings.QDRANT_COLLECTION_NAME)


def create_collection(vector_size: int) -> None:
    """Create a collection in Qdrant if it does not exist."""

    if collection_exists():
        return

    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


def add_embedding(vector: list[float], payload: dict[str, Any]) -> None:
    """Add an embedding to the Qdrant collection."""

    point_id = str(uuid.uuid4())
    point = PointStruct(id=point_id, vector=vector, payload=payload)
    client.upsert(collection_name=settings.QDRANT_COLLECTION_NAME, points=[point])


def search_similar(vector: list[float], limit: int = 5):
    """Search for similar embeddings in the Qdrant collection."""

    search_result = client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query_vector=vector,
        limit=limit,
    )

    return search_result
