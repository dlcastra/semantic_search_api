import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.auth.models import UserCreate, UserLogin
from src.auth.services import AuthService
from src.auth.utils import AuthUtils, store_session
from src.core.settings import logger, get_redis
from src.database.engine.config import get_db
from src.validators.password_validator import PasswordValidator, invalid_password

router = APIRouter()


class RegistrationResponse(BaseModel):
    message: str = "Registration successfully completed"


class LogoutResponse(BaseModel):
    message: str = "Session logged out"


@router.post("/registration", response_model=RegistrationResponse, status_code=201)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    auth_service = AuthService(db)
    psw_validator = PasswordValidator()

    try:
        is_valid_psw = psw_validator.password_validator(user_data.model_dump())
        if not is_valid_psw:
            raise HTTPException(status_code=400, detail=invalid_password)

        new_user = await auth_service.register_user(user_data.model_dump())
        if not new_user:
            raise HTTPException(status_code=400, detail="User registration failed")

        return RegistrationResponse()

    except HTTPException as e:
        raise e

    except Exception as e:
        logger.error("Error during user registration", exc_info=e)
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/login")
async def login(response: Response, user_data: UserLogin, db: AsyncSession = Depends(get_db), redis=Depends(get_redis)):
    auth_service = AuthService(db)
    username = user_data.username
    password = user_data.password
    user = await auth_service.authenticate_user(username, password)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    session_id = str(uuid.uuid4())
    logger.info(f"user {user.id} type: {type(user.id)}; logged in with session ID: {session_id}")
    token = AuthUtils.create_access_token(user.id)

    await store_session(redis, user.id, session_id)

    response.set_cookie(key="access_token", value=token, httponly=True, samesite="lax", secure=False)

    return {"message": "Logged in", "access_token": token}
