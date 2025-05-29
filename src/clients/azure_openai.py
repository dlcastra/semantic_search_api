from openai import AzureOpenAI
from azure.storage.blob.aio import BlobServiceClient

from src.core.settings import settings

embedding_client = AzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version="2024-12-01-preview",
)

blob_service_client = BlobServiceClient.from_connection_string(settings.AZURE_CONNECTION_STRING)
