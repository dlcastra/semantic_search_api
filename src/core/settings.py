import logging

from colorama import Fore, Style
from decouple import config
from pydantic_settings import BaseSettings
from redis.asyncio import Redis


class ModelSettings(BaseSettings):
    AZURE_OPENAI_API_KEY: str = config("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: str = config("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_MODEL_NAME: str = config("AZURE_OPENAI_MODEL_NAME")
    AZURE_OPENAI_DEPLOYMENT_NAME: str = config("AZURE_OPENAI_DEPLOYMENT_NAME")


class QdrantSettings(BaseSettings):
    QDRANT_HOST: str = config("QDRANT_HOST")
    QDRANT_HTTP_PORT: str = config("QDRANT_HTTP_PORT")
    QDRANT_GRPC_PORT: str = config("QDRANT_GRPC_PORT")
    QDRANT_VECTOR_SIZE: str = config("QDRANT_VECTOR_SIZE")
    QDRANT_COLLECTION_NAME: str = config("QDRANT_COLLECTION_NAME")


class PostgresSettings(BaseSettings):
    POSTGRES_HOST: str = config("POSTGRES_HOST")
    POSTGRES_PORT: int = config("POSTGRES_PORT", cast=int, default=5432)
    POSTGRES_USER: str = config("POSTGRES_USER")
    POSTGRES_PASSWORD: str = config("POSTGRES_PASSWORD")
    POSTGRES_DB: str = config("POSTGRES_DB")

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def LOCAL_DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}" f"@localhost:5433/{self.POSTGRES_DB}"
        )


class AzureStorageSettings(BaseSettings):
    AZURE_CONNECTION_STRING: str = config("AZURE_CONNECTION_STRING")
    CONTAINER_NAME: str = config("CONTAINER_NAME")


class AppSettings(BaseSettings):
    PORT: int = 8000


class Settings(AppSettings, AzureStorageSettings, ModelSettings, PostgresSettings, QdrantSettings):
    DEBUG: bool = False
    SECRET_KEY: str = config("SECRET_KEY")
    NLTK_DATA_DIR: str = "/app/nltk_data"
    DCOCKER_ENV: str = config("DOCKER_ENV", default="false")


class ColorLogFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.LIGHTYELLOW_EX,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"


console_handler = logging.StreamHandler()
console_handler.setFormatter(ColorLogFormatter("%(levelname)s: %(message)s"))
logging.basicConfig(level=logging.DEBUG, handlers=[console_handler])

settings = Settings()
logger = logging.getLogger(__name__)


def get_redis():
    """Create a Redis connection."""
    if settings.DCOCKER_ENV == "true":
        redis_url = "redis://redis:6379/1"
    else:
        redis_url = "redis://localhost:6379/1"

    return Redis.from_url(redis_url, decode_responses=True)
