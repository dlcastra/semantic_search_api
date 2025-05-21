from fastapi import APIRouter
from pydantic import BaseModel

from src.embedding.vector_db import create_collection, search_similar, add_embedding

router = APIRouter()
create_collection(vector_size=384)


class TextInput(BaseModel):
    text: str


@router.post("/add-embedding")
def add_embedding_router(text_input: TextInput):
    embedding = []
    add_embedding(vector=embedding, payload={"text": text_input.text})

    return {"status": "success"}


@router.post("/search-embedding")
def search_text_embedding_router(text_input: TextInput) -> dict:
    embedding = []

    search_result = search_similar(vector=embedding, limit=5)
    response = [
        {"id": r.id, "score": r.score, "text": r.payload.get("text")}
        for r in search_result
    ]

    return {"status": "success", "results": response}
