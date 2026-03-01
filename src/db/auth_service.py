from __future__ import annotations

import bcrypt
from sqlalchemy.orm import Session
from src.db.models import User

def hash_password(password: str) -> str:
    """Hash a password for storing."""
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def check_password(password: str, hashed_password: str) -> bool:
    """Check a hashed password."""
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))

def register_user(session: Session, email: str, password: str, role: str = "customer") -> User | None:
    """Register a new user."""
    # Check if user already exists
    existing_user = session.query(User).filter(User.email == email).first()
    if existing_user:
        return None
    
    password_hash = hash_password(password)
    new_user = User(email=email, password_hash=password_hash, role=role)
    session.add(new_user)
    session.commit()
    return new_user

def authenticate_user(session: Session, email: str, password: str) -> User | None:
    """Authenticate a user."""
    user = session.query(User).filter(User.email == email).first()
    if user and check_password(password, user.password_hash):
        return user
    return None
