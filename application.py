from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

from src.core.settings import settings, logger
from src.embedding import routers as embedding_routers
from src.embedding.vector_db import create_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_collection(vector_size=settings.QDRANT_VECTOR_SIZE)
    logger.info(settings.DATABASE_URL)
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(embedding_routers.router, prefix="/api/v1/embedding", tags=["embedding"])


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    errors = [{"field": err["loc"][-1], "msg": err["msg"]} for err in exc.errors()]
    return JSONResponse(
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": errors},
    )
