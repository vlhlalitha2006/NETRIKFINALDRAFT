import sys
import os
from sqlalchemy.orm import Session
from src.db.session import session_scope, init_db
from src.db.base import Base
from src.db.models import User
from src.db.auth_service import register_user, authenticate_user

def test_auth():
    print("Starting Auth Verification...")
    
    # Ensure DB is initialized
    with session_scope() as session:
        init_db(Base.metadata)
    
    email = "test@example.com"
    password = "password123"
    
    with session_scope() as session:
        # 1. Test Registration
        print(f"Testing registration for {email}...")
        user = register_user(session, email, password)
        if user:
            print(f"Successfully registered user with ID: {user.id}")
        else:
            # Check if user already existed from previous run
            existing = session.query(User).filter(User.email == email).first()
            if existing:
                print(f"User {email} already exists with ID: {existing.id}")
            else:
                print("Registration failed!")
                return False
        
        # 2. Test Authentication (Success)
        print("Testing authentication (Success)...")
        auth_user = authenticate_user(session, email, password)
        if auth_user and auth_user.email == email:
            print("Authentication successful!")
        else:
            print("Authentication failed!")
            return False
            
        # 3. Test Authentication (Failure - Wrong Password)
        print("Testing authentication (Failure - Wrong Password)...")
        fail_user = authenticate_user(session, email, "wrongpassword")
        if fail_user is None:
            print("Correctly rejected wrong password.")
        else:
            print("Failed: Accepted wrong password!")
            return False
            
    print("Auth Verification Completed Successfully!")
    return True

if __name__ == "__main__":
    if test_auth():
        sys.exit(0)
    else:
        sys.exit(1)
