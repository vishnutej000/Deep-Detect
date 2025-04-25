import os
import numpy as np
import cv2
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from typing import Dict, Tuple
import streamlit as st
import matplotlib.pyplot as plt
from visualization.heatmap import generate_heatmap
import config

# Define a simple CNN model for deepfake detection
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        # Use a pre-trained ResNet as the backbone
        self.backbone = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Add custom classification layers
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # Grad-CAM hooks
        self.gradients = None
        self.activations = None
        
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        # Get features from the backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Register hooks for Grad-CAM
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
            self.activations = x
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

# Define image preprocessing
def preprocess_image(image_path):
    """Preprocess image for the model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
    return img_tensor.unsqueeze(0)  # Add batch dimension

def load_model():
    """Load the pre-trained deepfake detection model."""
    model_path = config.MODEL_PATHS["image"]
    
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
    
    return model

def analyze_image(image_path: str) -> Dict:
    """Analyze an image for deepfake detection."""
    try:
        # Load model
        model = load_model()
        
        # Preprocess image
        img_tensor = preprocess_image(image_path)
        
        # Set requires_grad for Grad-CAM
        img_tensor.requires_grad_()
        
        # Get prediction
        with torch.set_grad_enabled(True):
            output = model(img_tensor)
            confidence = output.item()
            
            # Calculate gradients for Grad-CAM
            model.zero_grad()
            output.backward()
            
            # Generate heatmap visualization
            heatmap_img = generate_grad_cam_heatmap(model, img_tensor, image_path)
            
        # Determine result
        is_fake = confidence > config.THRESHOLDS["image"]
        
        # Create result dictionary
        result = {
            "is_fake": bool(is_fake),
            "confidence": float(confidence),
            "label": "Fake" if is_fake else "Real",
            "heatmap": heatmap_img
        }
        
        return result
    
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return {
            "is_fake": False,
            "confidence": 0.0,
            "label": "Error",
            "error": str(e)
        }

def generate_grad_cam_heatmap(model, img_tensor, original_image_path):
    """Generate a Grad-CAM heatmap from the model's activations."""
    import matplotlib.pyplot as plt
    import cv2
    from visualization.heatmap import generate_heatmap
    
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
    
    # Resize heatmap to original image size
    img = cv2.imread(original_image_path)
    height, width, _ = img.shape
    heatmap = cv2.resize(heatmap, (width, height))
    
    # Create colored heatmap using visualization module
    return generate_heatmap(img, heatmap)