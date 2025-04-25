import os
import requests
import json
from typing import Dict, Any, Optional
import streamlit as st

def get_huggingface_api_key() -> Optional[str]:
    """Get the Hugging Face API key from environment variables or session state"""
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    if not api_key and "huggingface_api_key" in st.session_state:
        api_key = st.session_state.huggingface_api_key
        
    return api_key

def query_deepfake_model(image_path: str, model_id: str = "deepfakealertproject/deepfake-detection") -> Dict[str, Any]:
    """
    Query a deepfake detection model on Hugging Face Inference API
    
    Args:
        image_path: Path to the image file
        model_id: Model ID on Hugging Face
        
    Returns:
        Dictionary with the model response
    """
    api_key = get_huggingface_api_key()
    
    if not api_key:
        return {
            "error": "No Hugging Face API key found. Please set the HUGGINGFACE_API_KEY environment variable."
        }
    
    api_url = f"https://api-inference.huggingface.co/models/{model_id}"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        with open(image_path, "rb") as file:
            data = file.read()
            
        response = requests.post(api_url, headers=headers, data=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"API request failed with status code: {response.status_code}",
                "message": response.text
            }
    
    except Exception as e:
        return {
            "error": f"Error querying model: {str(e)}"
        }

def set_api_key(api_key: str) -> None:
    """Set the Hugging Face API key in session state"""
    st.session_state.huggingface_api_key = api_key