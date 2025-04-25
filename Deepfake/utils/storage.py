import os
import tempfile
import streamlit as st
from typing import Optional
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def get_supabase_client():
    """Get the Supabase client from session state."""
    if hasattr(st.session_state, "supabase"):
        return st.session_state.supabase
    return None

def upload_file(file_path: str, storage_path: str, media_type: str) -> bool:
    """
    Upload a file to Supabase storage
    
    Args:
        file_path: Local path to the file
        storage_path: Path to store the file in Supabase storage
        media_type: Type of media (image, audio, video)
        
    Returns:
        Boolean indicating success
    """
    supabase = get_supabase_client()
    if not supabase:
        return False
    
    try:
        # Read file
        with open(file_path, "rb") as f:
            file_contents = f.read()
        
        # Upload to storage
        bucket_name = "analysis_files"
        response = supabase.storage.from_(bucket_name).upload(
            storage_path,
            file_contents,
            {"content-type": f"{media_type}/{os.path.splitext(file_path)[1][1:]}"}
        )
        
        return True
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        return False

def get_file_url(storage_path: str) -> Optional[str]:
    """
    Get the URL for a file in Supabase storage
    
    Args:
        storage_path: Path to the file in storage
        
    Returns:
        URL string or None if error
    """
    supabase = get_supabase_client()
    if not supabase:
        return None
    
    try:
        bucket_name = "analysis_files"
        url = supabase.storage.from_(bucket_name).get_public_url(storage_path)
        return url
    except Exception as e:
        print(f"Error getting file URL: {str(e)}")
        return None

def download_file(storage_path: str) -> Optional[str]:
    """
    Download a file from Supabase storage to a temp file
    
    Args:
        storage_path: Path to the file in storage
        
    Returns:
        Path to local temp file or None if error
    """
    supabase = get_supabase_client()
    if not supabase:
        return None
    
    try:
        bucket_name = "analysis_files"
        response = supabase.storage.from_(bucket_name).download(storage_path)
        
        # Save to temp file
        suffix = os.path.splitext(storage_path)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(response)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        print(f"Error downloading file: {str(e)}")
        return None