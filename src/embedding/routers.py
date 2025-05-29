import asyncio

from fastapi import APIRouter, Query
from fastapi.params import Depends
from pydantic import BaseModel

from src.auth.utils import get_current_user
from src.clients.azure_openai import embedding_client
from src.core.settings import settings
from src.embedding.vector_db import search_similar, add_embedding, get_all_user_embeddings

router = APIRouter()


class TextInput(BaseModel):
    text: str


@router.post("/add-embedding")
async def add_embedding_router(text_input: TextInput, auth_payload: dict = Depends(get_current_user)):
    """Add an embedding for the provided text input."""

    user_id = auth_payload.get("user").get("sub")

    response = await asyncio.to_thread(
        embedding_client.embeddings.create,
        input=[text_input.text],
        model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
    )

    embedding = response.data[0].embedding
    await add_embedding(vector=embedding, payload={"user_id": user_id, "text": text_input.text})

    return {"status": "success", "points": embedding}


@router.post("/search-embedding", dependencies=[Depends(get_current_user)])
async def search_text_embedding_router(text_input: TextInput) -> dict:
    """Search for similar embeddings based on the provided text input."""

    response = await asyncio.to_thread(
        embedding_client.embeddings.create,
        input=[text_input.text],
        model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
    )

    embedding = response.data[0].embedding
    search_result = await search_similar(vector=embedding, limit=5)
    response = [{"id": r.id, "score": r.score, "text": r.payload.get("text")} for r in search_result]

    return {"status": "success", "results": response}


@router.get("/get-all-embeddings/", status_code=200)
async def get_all_embeddings_router(
    limit: int = Query(default=50), auth_payload: dict = Depends(get_current_user)
) -> dict:
    """Fetch all embeddings for the authenticated user."""

    user_id = auth_payload.get("user").get("sub")

    embeddings = await get_all_user_embeddings(user_id, limit=limit)
    response = [r for r in embeddings[0]]

    return {"status": "success", "embeddings": response}
