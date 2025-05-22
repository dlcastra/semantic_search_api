from qdrant_client import QdrantClient

from src.core.settings import settings

client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_HTTP_PORT, grpc_port=settings.QDRANT_GRPC_PORT)
