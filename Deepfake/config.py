# Configuration settings for the Deepfake Detection application

# App name and info
APP_NAME = "Deep Detect"
VERSION = "1.0.0"
DESCRIPTION = "AI-powered deepfake detection tool"

# Supabase configuration - load from environment variables
SUPABASE_URL = ""  # Load from .env
SUPABASE_KEY = ""  # Load from .env

# Model paths
MODEL_PATHS = {
    "image": "models/image_deepfake_detector.pt",
    "audio": "models/audio_deepfake_detector.h5",
    "video": "models/video_deepfake_detector.pt"
}

# Detection thresholds
THRESHOLDS = {
    "image": 0.5,
    "audio": 0.5,
    "video": 0.5
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    "image": ["jpg", "jpeg", "png"],
    "audio": ["mp3", "wav", "ogg"],
    "video": ["mp4", "mov", "avi"]
}

# Maximum file sizes (in MB)
MAX_FILE_SIZES = {
    "image": 5,
    "audio": 10,
    "video": 50
}

# Try to load environment variables
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Update configuration from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL", SUPABASE_URL)
SUPABASE_KEY = os.getenv("SUPABASE_KEY", SUPABASE_KEY)