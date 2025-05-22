from fastapi import APIRouter
from pydantic import BaseModel

from src.clients.azure_openai import client
from src.core.settings import settings
from src.embedding.vector_db import create_collection, search_similar, add_embedding

router = APIRouter()
create_collection(vector_size=settings.QDRANT_VECTOR_SIZE)


class TextInput(BaseModel):
    text: str


@router.post("/add-embedding")
def add_embedding_router(text_input: TextInput):
    response = client.embeddings.create(input=[text_input.text], model=settings.AZURE_OPENAI_DEPLOYMENT_NAME)

    embedding = response.data[0].embedding
    add_embedding(vector=embedding, payload={"text": text_input.text})

    return {"status": "success", "points": embedding}


@router.post("/search-embedding")
def search_text_embedding_router(text_input: TextInput) -> dict:
    response = client.embeddings.create(input=[text_input.text], model=settings.AZURE_OPENAI_DEPLOYMENT_NAME)

    embedding = response.data[0].embedding
    search_result = search_similar(vector=embedding, limit=5)
    response = [{"id": r.id, "score": r.score, "text": r.payload.get("text")} for r in search_result]

    return {"status": "success", "results": response}
