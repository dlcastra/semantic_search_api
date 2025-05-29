import asyncio
import uuid

import tiktoken
from fastapi import APIRouter, Query, File, UploadFile, Form
from fastapi.params import Depends
from pydantic import BaseModel

from src.auth.utils import get_current_user
from src.clients.azure_openai import embedding_client
from src.core.settings import settings
from src.embedding.services import get_text_extractor_service, get_text_tokenization_service
from src.embedding.vector_db import search_similar, add_embedding, get_all_user_embeddings

router = APIRouter()


class AddEmbeddingRequest(BaseModel):
    text: str = Form(None)
    file: UploadFile = File(None)


class SearchEmbeddingRequest(BaseModel):
    text: str
    limit: int = Form(default=5)
    score: float = Form(None)


@router.post("/add-embedding")
async def add_embedding_router(request_data: AddEmbeddingRequest, auth_payload: dict = Depends(get_current_user)):
    """Add an embedding for the provided text input."""

    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

    text_extractor = await get_text_extractor_service()
    text_tokenization = await get_text_tokenization_service(tokenizer, max_tokens=50)
    user_id = auth_payload.get("user").get("sub")
    text_chunks = []

    if request_data.file:
        ...

    if request_data.text:
        text_chunks = await text_tokenization.chunk_text(request_data.text)

    response = await asyncio.to_thread(
        embedding_client.embeddings.create,
        input=text_chunks,
        model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
    )

    for idx, (chunk, embedding_data) in enumerate(zip(text_chunks, response.data)):
        embedding = embedding_data.embedding
        point_id = str(uuid.uuid4())
        await add_embedding(vector=embedding, payload={"id": point_id, "user_id": user_id, "text": chunk})

    return {"status": "success", "chunks_saved": ""}


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
