import asyncio
from typing import Optional

import tiktoken
from fastapi import APIRouter, Query, File, UploadFile, Form
from fastapi.params import Depends
from pydantic import BaseModel

from src.auth.utils import get_current_user
from src.clients.azure_openai import embedding_client
from src.core.settings import settings
from src.embedding.services import get_text_extractor_service, get_embedding_service
from src.embedding.vector_db import search_similar, get_all_user_embeddings

router = APIRouter()


class SearchEmbeddingRequest(BaseModel):
    text: str
    limit: int = Form(default=5)
    score: float = Form(None)


@router.post("/add-embedding")
async def add_embedding_router(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    auth_payload: dict = Depends(get_current_user),
):
    """Add an embedding for the provided text input."""

    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    text_extractor = await get_text_extractor_service()
    embedding_service = await get_embedding_service(
        embedding_client=embedding_client,
        text_extractor=text_extractor,
        tokenizer=tokenizer,
        max_tokens=50,
    )

    user_id = auth_payload.get("user").get("sub")
    return await embedding_service.create_embeddings(user_id, text, file)


@router.post("/search-embedding", dependencies=[Depends(get_current_user)])
async def search_text_embedding_router(request_data: SearchEmbeddingRequest) -> dict:
    """Search for similar embeddings based on the provided text input."""

    response = await asyncio.to_thread(
        embedding_client.embeddings.create,
        input=[request_data.text],
        model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
    )

    embedding = response.data[0].embedding
    search_result = await search_similar(vector=embedding, limit=request_data.limit)
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
