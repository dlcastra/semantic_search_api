from datetime import timedelta, datetime

from fastapi import HTTPException
from fastapi.requests import Request
from jose import jwt, JWTError
from passlib.context import CryptContext
from starlette import status

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


async def get_user_from_cookies(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return None

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms="HS256")
        return {"auth_type": "session", "user": payload}
    except JWTError:
        return None


async def get_user_from_header(request: Request):
    token = request.headers.get("Authorization")
    if not token or not token.startswith("Bearer "):
        return None

    token = token.split(" ")[1]
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms="HS256")
        return {"auth_type": "token", "user": payload}
    except JWTError:
        return None


async def get_current_user(request: Request):
    user = await get_user_from_cookies(request)
    if user:
        return user

    user = await get_user_from_header(request)
    if user:
        return user

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")


async def store_session(redis, user_id, session_id):
    key = f"user:{user_id}:session:{session_id}"
    await redis.set(key, "active", ex=SESSION_AGE)
