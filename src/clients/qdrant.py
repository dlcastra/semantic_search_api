from qdrant_client import AsyncQdrantClient

from src.core.settings import settings

client = AsyncQdrantClient(
    host=settings.QDRANT_HOST, port=settings.QDRANT_HTTP_PORT, grpc_port=settings.QDRANT_GRPC_PORT
)
