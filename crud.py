from sqlalchemy.orm import Session
from models import User
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token settings
SECRET_KEY = "your_secret_key"  # Change this for production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Hash password
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

# Verify password
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# Create a new user
def create_user(db: Session, username: str, email: str, password: str) -> User:
    hashed_password = hash_password(password)
    new_user = User(username=username, email=email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

# Get user by username
def get_user_by_username(db: Session, username: str) -> User:
    return db.query(User).filter(User.username == username).first()

# Get user by email
def get_user_by_email(db: Session, email: str) -> User:
    return db.query(User).filter(User.email == email).first()

# Create JWT token
def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Verify and decode JWT token
def verify_access_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp = payload.get("exp")
        
        if exp and datetime.fromtimestamp(exp, timezone.utc) < datetime.now(timezone.utc):
            return None  # Token expired
        return payload  # Returns the decoded data
    except JWTError:
        return None
# NEW FUNCTION: Check if username exists
def check_username_exists(db: Session, username: str) -> bool:
    return db.query(User).filter(User.username == username).first() is not None

# NEW FUNCTION: Check if email exists
def check_email_exists(db: Session, email: str) -> bool:
    return db.query(User).filter(User.email == email).first() is not None