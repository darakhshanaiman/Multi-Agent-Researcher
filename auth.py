import os
import chainlit as cl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def authenticate_user(username, password):
    """
    Verifies user credentials against secure environment variables.
    Returns a Chainlit User object if valid, None otherwise.
    """
    # Get the secure password from .env
    # If not found, default to 'admin' (Safety fallback)
    secure_password = os.getenv("ADMIN_PASSWORD", "admin")
    
    # Define your valid users
    # In a real app, you would check a database here.
    valid_users = {
        "admin": secure_password,
        "researcher": "science",
        "guest": "guest"
    }
    
    # Check credentials
    if username in valid_users and valid_users[username] == password:
        return cl.User(identifier=username)
    
    return None