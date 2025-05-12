from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr

    class Config:
        from_attributes = True
        
    class LoginRequest(BaseModel):
        username: str
        password: str

# Schema for JWT token response
class Token(BaseModel):
    access_token: str
    token_type: str

# Schema for decoding token data
class TokenData(BaseModel):
    username: str | None = None
