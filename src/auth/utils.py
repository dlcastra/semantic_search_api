from datetime import timedelta, datetime

from fastapi import HTTPException
from fastapi.requests import Request
from jose import jwt
from passlib.context import CryptContext

from src.core.constants import SESSION_AGE
from src.core.settings import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthUtils:
    @staticmethod
    def hash_password(password: str) -> str:
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def create_access_token(user_id: int) -> str:
        expire = datetime.now() + timedelta(seconds=SESSION_AGE)
        payload = {"sub": str(user_id), "exp": int(expire.timestamp())}
        return jwt.encode(payload, settings.SECRET_KEY, algorithm="HS256")


async def get_current_user_from_session(request: Request):
    session_id = request.session.get("session_id")
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return session_id


async def store_session(redis, user_id, session_id):
    key = f"user:{user_id}:session:{session_id}"
    await redis.set(key, "active", ex=SESSION_AGE)
