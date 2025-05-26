import re

from pydantic import BaseModel, EmailStr, field_validator, model_validator


class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    password1: str

    @field_validator("username")
    def validate_username(cls, username):
        if len(username) < 5:
            raise ValueError("Username must be at least 5 characters long")
        if len(username) > 100:
            raise ValueError("Username cannot contain more than 100 characters")
        if " " in username:
            raise ValueError("Username cannot contain spaces")
        if re.search(r"[;\'\"\-\-]", username):
            raise ValueError("Username contains forbidden characters")

        return username

    @model_validator(mode="after")
    def validate_password(self):
        if self.password != self.password1:
            raise ValueError("The passwords must match")

        return self


class UserLogin(BaseModel):
    username: str
    password: str
