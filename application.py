from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from src.core.settings import settings
from src.embedding import routers as embedding_routers
from src.auth import routers as auth_routers
from src.embedding.vector_db import create_collection
import nltk


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_collection(vector_size=settings.QDRANT_VECTOR_SIZE)
    nltk.download("punkt_tab", download_dir=settings.NLTK_DATA_DIR)
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(embedding_routers.router, prefix="/api/v1/embedding", tags=["embedding"])
app.include_router(auth_routers.router, prefix="/api/v1/auth", tags=["auth"])


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    errors = [{"field": err["loc"][-1], "msg": err["msg"]} for err in exc.errors()]
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": errors},
    )
