import streamlit as st
from supabase import create_client
from supabase.client import Client
import re
from typing import Dict, Any, Optional, Tuple
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Initialize Supabase client
def get_supabase_client() -> Client:
    """Get the Supabase client from environment variables or session state"""
    if not hasattr(st.session_state, "supabase"):
        # Check if Supabase URL and key are configured
        if not config.SUPABASE_URL or not config.SUPABASE_KEY:
            st.error("Supabase URL or key not configured. Please check your .env file.")
            return None
        
        try:
            # Create client without additional options to avoid compatibility issues
            st.session_state.supabase = create_client(
                config.SUPABASE_URL, 
                config.SUPABASE_KEY
            )
        except TypeError as e:
            # Handle specific TypeError related to proxy argument
            if "unexpected keyword argument 'proxy'" in str(e):
                st.error("Compatibility issue with Supabase client. Try updating the supabase package: pip install --upgrade supabase")
                return None
            else:
                raise e
    
    return st.session_state.supabase

def is_valid_email(email: str) -> bool:
    """Check if the provided email is valid."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None

def is_valid_password(password: str) -> bool:
    """Check if the provided password meets requirements."""
    # At least 8 characters with 1 uppercase, 1 lowercase, 1 number
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"[0-9]", password):
        return False
    return True

def signup(email: str, password: str) -> Tuple[bool, Dict]:
    """Sign up a new user with email and password."""
    if not is_valid_email(email):
        return False, {"error": "Invalid email format"}
    
    if not is_valid_password(password):
        return False, {"error": "Password must be at least 8 characters with 1 uppercase, 1 lowercase, and 1 number"}
    
    supabase = get_supabase_client()
    if not supabase:
        return False, {"error": "Supabase client not initialized"}
    
    try:
        # Use Supabase auth to create user (it will automatically send a verification email)
        response = supabase.auth.sign_up({
            "email": email,
            "password": password
        })
        
        return True, {"message": "Signup successful! Please check your email to verify your account."}
    except Exception as e:
        return False, {"error": str(e)}

def login_with_email(email: str, password: str) -> Tuple[bool, Dict]:
    """Log in user with email and password."""
    supabase = get_supabase_client()
    if not supabase:
        return False, {"error": "Supabase client not initialized"}
    
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        # Get user data
        user = response.user
        
        # Store user in session state
        st.session_state.user = user
        st.session_state.authenticated = True
        
        return True, {"message": "Login successful!"}
    except Exception as e:
        return False, {"error": str(e)}

def login_with_google() -> str:
    """Initialize Google OAuth login flow."""
    supabase = get_supabase_client()
    if not supabase:
        return ""
    
    try:
        # Use st.query_params instead of st.experimental_get_query_params
        redirect_uri = ""
        if hasattr(st, "query_params") and "redirect_uri" in st.query_params:
            redirect_uri = st.query_params.get("redirect_uri", "")
        
        response = supabase.auth.sign_in_with_oauth({
            "provider": "google",
            "options": {
                "redirect_to": redirect_uri
            }
        })
        
        # Return the authorization URL for the user to visit
        return response.url
    except Exception as e:
        st.error(f"Error initiating Google login: {str(e)}")
        return ""

def handle_oauth_callback() -> bool:
    """Handle OAuth callback."""
    # Use st.query_params instead of st.experimental_get_query_params
    if hasattr(st, "query_params"):
        query_params = st.query_params
    else:
        # Fallback for older Streamlit versions
        query_params = {}
    
    # If we have auth parameters, process them
    if "access_token" in query_params and "refresh_token" in query_params:
        supabase = get_supabase_client()
        if not supabase:
            return False
        
        try:
            # Set session with tokens
            session = supabase.auth.set_session({
                "access_token": query_params["access_token"],
                "refresh_token": query_params["refresh_token"]
            })
            
            # Store user in session state
            st.session_state.user = session.user
            st.session_state.authenticated = True
            
            # Clear query parameters (use the appropriate method)
            if hasattr(st, "query_params"):
                st.query_params.clear()
            
            return True
        except Exception as e:
            st.error(f"Error processing authentication: {str(e)}")
    
    return False

def logout() -> None:
    """Log out the current user."""
    supabase = get_supabase_client()
    if supabase:
        try:
            supabase.auth.sign_out()
        except Exception as e:
            st.error(f"Error during logout: {str(e)}")
    
    # Clear session state
    if "user" in st.session_state:
        del st.session_state.user
    if "authenticated" in st.session_state:
        del st.session_state.authenticated

def is_authenticated() -> bool:
    """Check if a user is authenticated."""
    # First check session state
    if st.session_state.get("authenticated", False):
        return True
    
    # Then check with Supabase
    supabase = get_supabase_client()
    if not supabase:
        return False
    
    try:
        user = supabase.auth.get_user()
        if user and hasattr(user, 'user') and user.user:
            # Update session state
            st.session_state.user = user.user
            st.session_state.authenticated = True
            return True
    except Exception:
        # Error means no valid session
        pass
    
    return False

def get_current_user() -> Optional[Dict]:
    """Get the current authenticated user."""
    if is_authenticated():
        return st.session_state.user
    return None

def init_auth() -> None:
    """Initialize authentication state."""
    # Check for OAuth callback
    if handle_oauth_callback():
        st.rerun()
    
    # Check for existing session
    supabase = get_supabase_client()
    if not supabase:
        return
    
    try:
        session = supabase.auth.get_session()
        if session and hasattr(session, 'user') and session.user:
            st.session_state.user = session.user
            st.session_state.authenticated = True
    except Exception:
        # No valid session
        if "user" in st.session_state:
            del st.session_state.user
        if "authenticated" in st.session_state:
            del st.session_state.authenticated