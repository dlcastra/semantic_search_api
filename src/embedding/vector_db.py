import uuid
from typing import Any

from qdrant_client import models
from qdrant_client.models import VectorParams, Distance, PointStruct

from src.core.settings import settings
from src.clients.qdrant import client


async def collection_exists() -> bool:
    """Check if the collection exists in Qdrant."""

    return await client.collection_exists(settings.QDRANT_COLLECTION_NAME)


async def create_collection(vector_size: int) -> None:
    """Create a collection in Qdrant if it does not exist."""

    if await collection_exists():
        return

    await client.create_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


async def add_embedding(vector: list[float], payload: dict[str, Any]) -> None:
    """Add an embedding to the Qdrant collection."""

    point_id = str(uuid.uuid4())
    point = PointStruct(id=point_id, vector=vector, payload=payload)
    await client.upsert(collection_name=settings.QDRANT_COLLECTION_NAME, points=[point])


async def search_similar(vector: list[float], limit: int = 5):
    """Search for similar embeddings in the Qdrant collection."""

    search_result = await client.search(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query_vector=vector,
        limit=limit,
        with_payload=True,
    )

    return search_result


async def get_all_user_embeddings(user_id: str, limit: int = 50) -> list[PointStruct]:
    points = await client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))]
        ),
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    return points
