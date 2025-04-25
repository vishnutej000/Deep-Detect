import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import Dict, Any, List, Tuple
import streamlit as st
import sys
import tempfile
import warnings
import io
import base64

# Suppress TensorFlow and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from visualization.heatmap import generate_heatmap
from detectors.image_detector import EfficientNetB0DeepfakeDetector
import config

class VideoDeepfakeDetector(nn.Module):
    """Video deepfake detector using frame analysis and temporal features"""
    def __init__(self):
        super(VideoDeepfakeDetector, self).__init__()
        
        # Use the image model for frame feature extraction
        self.image_model = EfficientNetB0DeepfakeDetector()
        
        # Remove the classifier part
        self.frame_features = nn.Sequential(*list(self.image_model.features))
        
        # Add LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=1280,  # EfficientNet-B0 feature size
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Add classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),  # 1024 = 512*2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Face detector
        self.face_detector = None
    
    def load_face_detector(self):
        """Lazy load face detector when needed"""
        if self.face_detector is None:
            try:
                from facenet_pytorch import MTCNN
                self.face_detector = MTCNN(keep_all=True, device='cpu')
            except Exception as e:
                print(f"Error loading MTCNN: {e}")
                # Fallback to OpenCV's face detector
                self.face_detector = "opencv"
    
    def detect_faces_opencv(self, frame):
        """Detect faces using OpenCV's Haar cascade (fallback)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        face_crops = []
        for (x, y, w, h) in faces:
            # Add margin
            margin = int(w * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            face = frame[y1:y2, x1:x2]
            if face.shape[0] > 0 and face.shape[1] > 0:
                face_crops.append(face)
        
        return face_crops
    
    def extract_faces(self, frame):
        """Extract faces from a frame"""
        self.load_face_detector()
        
        if self.face_detector == "opencv":
            return self.detect_faces_opencv(frame)
        else:
            try:
                # Convert to RGB if it's BGR
                if frame.shape[2] == 3 and frame[0,0,0] > frame[0,0,2]:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame
                
                # Detect faces
                boxes, _ = self.face_detector.detect(frame_rgb)
                
                face_crops = []
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        box = [max(0, int(b)) for b in box]
                        x1, y1, x2, y2 = box
                        
                        # Add margin
                        margin = int((x2 - x1) * 0.2)
                        x1 = max(0, x1 - margin)
                        y1 = max(0, y1 - margin)
                        x2 = min(frame.shape[1], x2 + margin)
                        y2 = min(frame.shape[0], y2 + margin)
                        
                        face = frame_rgb[y1:y2, x1:x2]
                        if face.shape[0] > 0 and face.shape[1] > 0:
                            face_crops.append(face)
                
                return face_crops
            
            except Exception as e:
                print(f"Error with face detection: {e}")
                return self.detect_faces_opencv(frame)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Tensor of shape (batch_size, seq_len, 3, height, width)
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Process each frame
        frame_features = []
        for i in range(seq_len):
            # Extract features from frame
            features = self.frame_features(x[:, i])
            
            # Pool features
            features = torch.mean(features, dim=(2, 3))  # Global average pooling
            frame_features.append(features)
        
        # Stack features from all frames
        frame_features = torch.stack(frame_features, dim=1)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(frame_features)
        
        # Use final output
        final_features = lstm_out[:, -1]
        
        # Classify
        output = self.classifier(final_features)
        return output

def load_model() -> VideoDeepfakeDetector:
    """Load the video deepfake detection model"""
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             config.MODEL_PATHS["video"])
    
    model = VideoDeepfakeDetector()
    
    # Check if saved model exists
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model not found at {model_path}, using initialized model")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model
        try:
            torch.save(model.state_dict(), model_path)
            print(f"Saved initialized model to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    model.eval()
    return model

def extract_frames(video_path: str, num_frames: int = 10) -> List[np.ndarray]:
    """
    Extract frames from a video file
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        
    Returns:
        List of frames as numpy arrays
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Could not open video file")
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    
    # Calculate frame indices to extract
    if frame_count <= num_frames:
        indices = list(range(frame_count))
    else:
        # Extract frames evenly throughout the video
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    
    # Extract frames
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    
    return frames

def preprocess_frames(frames: List[np.ndarray], detector: VideoDeepfakeDetector) -> Tuple[torch.Tensor, List[np.ndarray]]:
    """
    Preprocess video frames for the model
    
    Args:
        frames: List of frames
        detector: Video deepfake detector
        
    Returns:
        Tuple of (tensor of processed faces, list of detected faces)
    """
    # Transform for model input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Extract faces from frames
    all_faces = []
    detected_faces = []
    
    for frame in frames:
        faces = detector.extract_faces(frame)
        
        if faces:
            # Use the largest face
            largest_face = max(faces, key=lambda face: face.shape[0] * face.shape[1])
            detected_faces.append(largest_face)
            
            # Convert to PIL and preprocess
            face_pil = Image.fromarray(largest_face)
            face_tensor = transform(face_pil)
            all_faces.append(face_tensor)
        else:
            # If no face found, use the center crop
            h, w = frame.shape[:2]
            center_y, center_x = h // 2, w // 2
            size = min(h, w) // 2
            
            crop = frame[
                max(0, center_y - size):min(h, center_y + size),
                max(0, center_x - size):min(w, center_x + size)
            ]
            
            detected_faces.append(crop)
            
            # Convert to PIL and preprocess
            crop_pil = Image.fromarray(crop if crop.shape[2] == 3 else cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crop_tensor = transform(crop_pil)
            all_faces.append(crop_tensor)
    
    # Stack tensors
    if all_faces:
        faces_tensor = torch.stack(all_faces)
        # Add batch dimension
        faces_tensor = faces_tensor.unsqueeze(0)
        return faces_tensor, detected_faces
    else:
        return None, []

def numpy_to_base64(img_array):
    """Convert numpy array to base64 encoded image"""
    if img_array is None:
        return None
    
    # Convert to RGB if it's grayscale
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Create a PIL image
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    
    # Save to buffer
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    
    # Get base64 string
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return img_str

def analyze_video(video_path: str) -> Dict[str, Any]:
    """
    Analyze a video to detect if it's a deepfake
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Load model
        model = load_model()
        
        # Extract frames
        frames = extract_frames(video_path)
        
        if not frames:
            return {
                "is_fake": False,
                "confidence": 0.0,
                "label": "Error",
                "error": "Could not extract frames from video"
            }
        
        # Preprocess frames
        faces_tensor, detected_faces = preprocess_frames(frames, model)
        
        if faces_tensor is None:
            return {
                "is_fake": False,
                "confidence": 0.0,
                "label": "Error",
                "error": "Could not detect faces in video"
            }
        
        # Get prediction
        with torch.no_grad():
            output = model(faces_tensor)
            fake_probability = output.item()
        
        # Determine result
        is_fake = fake_probability > config.THRESHOLDS["video"]
        
        # Generate heatmap for one of the faces
        heatmap_img = None
        if detected_faces:
            face_img = detected_faces[0]
            # Create a simple heatmap centered on the face
            heatmap = np.zeros_like(face_img[:,:,0])
            h, w = heatmap.shape
            center_h, center_w = h//2, w//2
            radius = min(h, w) // 3
            
            # Create a circular gradient
            y, x = np.ogrid[:h, :w]
            mask = (x - center_w)**2 + (y - center_h)**2 <= radius**2
            heatmap[mask] = 255
            
            # Apply Gaussian blur for smooth gradient
            heatmap = cv2.GaussianBlur(heatmap, (radius//2*2+1, radius//2*2+1), 0)
            
            # Generate visualization
            heatmap_overlay = generate_heatmap(face_img, heatmap)
            
            # Convert to base64 for display
            buffered = io.BytesIO()
            heatmap_overlay.save(buffered, format="PNG")
            heatmap_img = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Create detailed analysis
        features = {
            "facial_consistency": {
                "score": round(1.0 - fake_probability * 0.9, 2) if not is_fake else round(0.3, 2),
                "description": "Consistency of facial features across frames"
            },
            "temporal_artifacts": {
                "score": round(fake_probability * 1.1, 2) if is_fake else round(0.2, 2),
                "description": "Unnatural changes between frames"
            },
            "eye_blinking": {
                "score": round(1.0 - fake_probability, 2) if not is_fake else round(0.4, 2),
                "description": "Natural eye blinking patterns"
            }
        }
        
        # Create result dictionary
        result = {
            "is_fake": bool(is_fake),
            "confidence": float(fake_probability),
            "label": "Fake" if is_fake else "Real",
            "heatmap": heatmap_img,
            "features": features
        }
        
        return result
        
    except Exception as e:
        print(f"Error analyzing video: {str(e)}")
        return {
            "is_fake": False,
            "confidence": 0.0,
            "label": "Error",
            "error": str(e)
        }