from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

# MySQL connection details
DATABASE_URL = "mysql+pymysql://root:1234@localhost:3306/login-register"

# Create the database engine
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
# engine.connect()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Ensure tables are created
Base.metadata.create_all(bind=engine)

# FastAPI Dependency for database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()