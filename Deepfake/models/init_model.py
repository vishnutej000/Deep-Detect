import os
import sys
import torch
import torch.nn as nn
import tensorflow as tf
import warnings
import numpy as np

# Suppress TensorFlow warnings at module level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detectors.image_detector import EfficientNetB0DeepfakeDetector
import config

def initialize_image_model():
    """Initialize and save the image deepfake detection model"""
    print("Initializing image deepfake detection model...")
    
    model = EfficientNetB0DeepfakeDetector()
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              os.path.basename(config.MODEL_PATHS["image"]))
    
    # Check if model file exists
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    try:
        torch.save(model.state_dict(), model_path)
        print(f"Saved image model to {model_path}")
    except Exception as e:
        print(f"Error saving image model: {e}")

def initialize_audio_model():
    """Initialize and save the audio deepfake detection model"""
    print("Initializing audio deepfake detection model...")
    
    # Create model architecture - avoid TensorFlow warnings with tf.compat API
    with tf.device('/CPU:0'):
        # Temporarily disable TensorFlow logging during model creation
        tf_log_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        
        model = tf.keras.Sequential([
            # Input layer for mel spectrograms
            tf.keras.layers.Input(shape=(128, 128, 1)),
            
            # Convolutional blocks
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.BatchNormalization(),
            
            # Flatten and dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Restore previous log level
        if tf_log_level:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_log_level
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               os.path.basename(config.MODEL_PATHS["audio"]))
    
    # Check if model file exists
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    try:
        # Use tf.compat.v1.reset_default_graph() instead of deprecated function
        tf.compat.v1.reset_default_graph()
        model.save(model_path, save_format='h5')
        print(f"Saved audio model to {model_path}")
    except Exception as e:
        print(f"Error saving audio model: {e}")

def initialize_video_model():
    """Initialize and save the video deepfake detection model"""
    print("Initializing video deepfake detection model...")
    
    # First, make sure image model is initialized
    initialize_image_model()
    
    # Create a simple VideoDeepfakeDetector
    class SimpleVideoDeepfakeDetector(nn.Module):
        def __init__(self):
            super(SimpleVideoDeepfakeDetector, self).__init__()
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
    
    model = SimpleVideoDeepfakeDetector()
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              os.path.basename(config.MODEL_PATHS["video"]))
    
    # Check if model file exists
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    try:
        torch.save(model.state_dict(), model_path)
        print(f"Saved video model to {model_path}")
    except Exception as e:
        print(f"Error saving video model: {e}")

def initialize_models():
    """Initialize all models"""
    print("Initializing all deepfake detection models...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize each model
    initialize_image_model()
    initialize_audio_model()
    initialize_video_model()
    
    print("All models initialized successfully!")

if __name__ == "__main__":
    # Suppress additional warnings
    os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning:tensorflow'
    
    # Run initialization
    initialize_models()