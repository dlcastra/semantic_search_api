from pydantic_settings import BaseSettings
from decouple import config

class ModelSettings(BaseSettings):
    EMBEDDING_MODEL: str = config("EMBEDDING_MODEL")


class DatabaseSettings(BaseSettings):
    QDRANT_HOST: str = config("QDRANT_HOST")
    QDRANT_HTTP_PORT: str = config("QDRANT_HTTP_PORT")
    QDRANT_GRPC_PORT: str = config("QDRANT_GRPC_PORT")
    QDRANT_VECTOR_SIZE: str = 384
    QDRANT_COLLECTION_NAME: str = config("QDRANT_COLLECTION_NAME")


class AppSettings(BaseSettings):
    PORT: int = 8000


class Settings(ModelSettings, DatabaseSettings, AppSettings):
    DEBUG: bool = False


settings = Settings()
