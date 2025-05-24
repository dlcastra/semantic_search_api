import asyncio

from fastapi import APIRouter
from pydantic import BaseModel

from src.clients.azure_openai import embedding_client
from src.core.settings import settings, logger
from src.embedding.vector_db import search_similar, add_embedding

router = APIRouter()


class TextInput(BaseModel):
    text: str


@router.post("/add-embedding")
async def add_embedding_router(text_input: TextInput):
    response = await asyncio.to_thread(
        embedding_client.embeddings.create,
        input=[text_input.text],
        model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
    )

    embedding = response.data[0].embedding
    await add_embedding(vector=embedding, payload={"text": text_input.text})
    logger.info(f"Added embedding {text_input.text}")

    return {"status": "success", "points": embedding}


@router.post("/search-embedding")
async def search_text_embedding_router(text_input: TextInput) -> dict:
    response = await asyncio.to_thread(
        embedding_client.embeddings.create,
        input=[text_input.text],
        model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
    )

    embedding = response.data[0].embedding
    search_result = await search_similar(vector=embedding, limit=5)
    response = [{"id": r.id, "score": r.score, "text": r.payload.get("text")} for r in search_result]

    return {"status": "success", "results": response}
