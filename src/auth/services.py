from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models.users import Users
from src.auth.utils import AuthUtils


class AuthService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.auth_util = AuthUtils()

    async def register_user(self, user_data):
        existing_user = await self.db.execute(select(Users).filter(Users.email == user_data["email"]))
        existing_user = existing_user.scalars().first()

        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")

        hashed_password = self.auth_util.hash_password(user_data["password"])
        new_user = Users(
            username=user_data["username"],
            email=user_data["email"],
            password=hashed_password,
        )

        self.db.add(new_user)
        try:
            await self.db.commit()
            return new_user
        except Exception:
            await self.db.rollback()
            raise HTTPException(status_code=500, detail="Failed to create user")

    async def authenticate_user(self, username: str, password: str):
        user = await self.db.execute(select(Users).filter(Users.username == username))
        user = user.scalars().first()

        if not user:
            return None

        if not self.auth_util.verify_password(password, user.password):
            return None

        return user
