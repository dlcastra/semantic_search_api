from decouple import config

from src.clients.azure_openai import blob_service_client


async def upload_file_to_azure_blob(file_id: str, file_bytes: bytes, file_name: str) -> str:
    """Upload a file to Azure Blob Storage."""

    container_name = config("CONTAINER_NAME")
    blob_name = f"{file_id}/{file_name}"

    blob_client = await blob_service_client.get_blob_client(container_name=container_name, blob_name=blob_name)
    await blob_client.upload_blob(file_bytes)

    return blob_client.url
