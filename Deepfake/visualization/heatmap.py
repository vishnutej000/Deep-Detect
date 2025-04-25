import numpy as np
import cv2
from PIL import Image
import io
import base64
from typing import Union, Optional
import matplotlib.pyplot as plt

def generate_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    Generate a heatmap overlay on an image
    
    Args:
        image: Input image as numpy array (RGB)
        heatmap: Heatmap data as numpy array (grayscale)
        
    Returns:
        Numpy array with heatmap overlay
    """
    # Make sure image is RGB
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]  # Just use RGB channels
    
    # Ensure image is uint8 type
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # Make sure heatmap is grayscale
    if len(heatmap.shape) > 2:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
    
    # Resize heatmap if needed
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Normalize heatmap to range 0-255
    if np.max(heatmap) > np.min(heatmap):
        heatmap = ((heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) * 255).astype(np.uint8)
    else:
        heatmap = np.zeros_like(heatmap, dtype=np.uint8)
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on image with alpha blending
    alpha = 0.4
    overlay = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    
    return overlay

def heatmap_to_base64(image: np.ndarray, heatmap: np.ndarray) -> Optional[str]:
    """
    Generate a heatmap and convert to base64 for displaying in HTML
    
    Args:
        image: Input image as numpy array
        heatmap: Heatmap data as numpy array
        
    Returns:
        Base64 encoded string of the heatmap image
    """
    try:
        # Generate heatmap overlay
        overlay = generate_heatmap(image, heatmap)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(overlay)
        
        # Save to buffer
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        
        # Convert to base64
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return img_str
    
    except Exception as e:
        print(f"Error generating heatmap: {str(e)}")
        return None

def heatmap_from_probabilities(probabilities: np.ndarray, size: tuple = (224, 224)) -> np.ndarray:
    """
    Generate a heatmap from probability values
    
    Args:
        probabilities: Array of probability values (e.g., from attention maps)
        size: Size to resize heatmap to (height, width)
        
    Returns:
        Numpy array with heatmap normalized to 0-255 range
    """
    # Normalize probabilities
    if np.max(probabilities) > np.min(probabilities):
        heatmap = (probabilities - np.min(probabilities)) / (np.max(probabilities) - np.min(probabilities))
    else:
        heatmap = np.zeros_like(probabilities)
    
    # Resize to required dimensions
    heatmap_resized = cv2.resize(heatmap, (size[1], size[0]))
    
    # Scale to 0-255 range
    heatmap_scaled = (heatmap_resized * 255).astype(np.uint8)
    
    return heatmap_scaled

def save_heatmap(image: np.ndarray, heatmap: np.ndarray, output_path: str) -> bool:
    """
    Generate a heatmap overlay and save to file
    
    Args:
        image: Input image as numpy array
        heatmap: Heatmap data as numpy array
        output_path: Path to save the output image
        
    Returns:
        Boolean indicating success
    """
    try:
        # Generate heatmap overlay
        overlay = generate_heatmap(image, heatmap)
        
        # Save to file
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        return True
    
    except Exception as e:
        print(f"Error saving heatmap: {str(e)}")
        return False

def plot_attention_heatmap(image: np.ndarray, attention_map: np.ndarray, title: str = "Attention Heatmap") -> plt.Figure:
    """
    Create a figure with original image, attention heatmap, and overlay
    
    Args:
        image: Original image
        attention_map: Attention weights
        title: Title for the plot
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Plot attention heatmap
    heatmap = cv2.resize(attention_map, (image.shape[1], image.shape[0]))
    im = axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Attention Map")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot overlay
    overlay = generate_heatmap(image, attention_map)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    return fig