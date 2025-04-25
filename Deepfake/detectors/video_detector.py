import os
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple
import streamlit as st
from facenet_pytorch import MTCNN
import tempfile
from visualization.heatmap import generate_heatmap
import config
from detectors.image_detector import DeepfakeDetector, preprocess_image

def load_models():
    """Load the pre-trained deepfake detection model and face detector."""
    # Load deepfake detection model (reusing the image model)
    model_path = config.MODEL_PATHS["video"]
    
    # Check if model exists
    if not os.path.exists(model_path):
        # For demo purposes, create a dummy model if not found
        model = DeepfakeDetector()
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save(model.state_dict(), model_path)
        st.warning("Using a dummy model for demonstration purposes. Replace with a properly trained model for production.")
    
    # Load model
    model = DeepfakeDetector()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    # Load face detector
    face_detector = MTCNN(
        keep_all=True,
        device='cpu'
    )
    
    return model, face_detector

def extract_frames(video_path: str, num_frames: int = 5) -> List[np.ndarray]:
    """Extract frames from video file."""
    frames = []
    
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count == 0:
            st.error("Could not read frames from video")
            return frames
        
        # Calculate frame indices to extract
        if frame_count <= num_frames:
            indices = list(range(frame_count))
        else:
            indices = np.linspace(0, frame_count - 1, num_frames, dtype=int).tolist()
        
        # Extract frames at calculated indices
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        
    except Exception as e:
        st.error(f"Error extracting frames: {str(e)}")
    
    return frames

def detect_and_crop_faces(frames: List[np.ndarray], face_detector) -> List[np.ndarray]:
    """Detect and crop faces from frames."""
    face_frames = []
    
    for frame in frames:
        # Convert from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, _ = face_detector.detect(frame_rgb)
        
        if boxes is not None and len(boxes) > 0:
            # Process each detected face
            for box in boxes:
                box = [max(0, int(b)) for b in box]
                x1, y1, x2, y2 = box
                
                # Crop face with some margin
                margin = 20
                h, w = frame.shape[:2]
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    face_frames.append(face_crop)
        else:
            # If no face detected, use the whole frame
            face_frames.append(frame)
    
    return face_frames

def analyze_face_frame(model, face_frame: np.ndarray) -> Tuple[float, np.ndarray]:
    """Analyze a face frame for deepfakes."""
    # Save face frame to temporary file for processing
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        cv2.imwrite(tmp_file.name, face_frame)
        tmp_path = tmp_file.name
    
    # Process with the model
    try:
        # Preprocess image
        img_tensor = preprocess_image(tmp_path)
        
        # Set requires_grad for Grad-CAM
        img_tensor.requires_grad_()
        
        # Get prediction
        with torch.set_grad_enabled(True):
            output = model(img_tensor)
            confidence = output.item()
            
            # Calculate gradients for Grad-CAM
            model.zero_grad()
            output.backward()
            
            # Generate heatmap
            heatmap = generate_grad_cam_heatmap(model, face_frame)
            
        # Delete temporary file
        os.unlink(tmp_path)
        
        return confidence, heatmap
    
    except Exception as e:
        st.error(f"Error analyzing face frame: {str(e)}")
        os.unlink(tmp_path)
        return 0.0, None

def generate_grad_cam_heatmap(model, frame):
    """Generate Grad-CAM heatmap for video frame."""
    # Get the model's activations and gradients
    activations = model.activations
    gradients = model.gradients
    
    # Calculate weights
    weights = torch.mean(gradients, dim=[0, 2, 3])
    
    # Generate heatmap
    heatmap = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * activations, dim=1).squeeze().detach()
    heatmap = torch.relu(heatmap)  # ReLU to keep only positive contributions
    
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = heatmap.cpu().numpy()
    
    # Resize heatmap to frame size
    height, width = frame.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (width, height))
    
    # Create colored heatmap
    return generate_heatmap(frame, heatmap_resized)

def analyze_video(video_path: str) -> Dict:
    """Analyze a video file for deepfake detection."""
    try:
        # Load models
        model, face_detector = load_models()
        
        # Extract frames
        frames = extract_frames(video_path)
        if not frames:
            return {
                "is_fake": False,
                "confidence": 0.0,
                "label": "Error",
                "error": "Failed to extract frames from video"
            }
        
        # Detect and crop faces
        face_frames = detect_and_crop_faces(frames, face_detector)
        if not face_frames:
            return {
                "is_fake": False,
                "confidence": 0.0,
                "label": "Error",
                "error": "No faces detected in video"
            }
        
        # Analyze each face frame
        confidences = []
        heatmaps = []
        
        for face_frame in face_frames:
            confidence, heatmap = analyze_face_frame(model, face_frame)
            confidences.append(confidence)
            if heatmap is not None:
                heatmaps.append(heatmap)
        
        # Calculate average confidence
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Determine result
        is_fake = avg_confidence > config.THRESHOLDS["video"]
        
        # Create result dictionary
        result = {
            "is_fake": bool(is_fake),
            "confidence": float(avg_confidence),
            "label": "Fake" if is_fake else "Real",
            "frame_confidences": [float(c) for c in confidences],
            "heatmaps": heatmaps[:3]  # Limit to first 3 heatmaps
        }
        
        return result
    
    except Exception as e:
        st.error(f"Error analyzing video: {str(e)}")
        return {
            "is_fake": False,
            "confidence": 0.0,
            "label": "Error",
            "error": str(e)
        }