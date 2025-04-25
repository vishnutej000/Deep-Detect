import os
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from typing import Optional, Union, Tuple

def generate_heatmap(image: np.ndarray, heatmap: np.ndarray) -> Image.Image:
    """
    Generate a heatmap overlay on an image
    
    Args:
        image: Input image as numpy array (RGB)
        heatmap: Heatmap data as numpy array (grayscale)
        
    Returns:
        PIL Image with heatmap overlay
    """
    # Make sure image is RGB
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]  # Just use RGB channels
    
    # Make sure heatmap is grayscale
    if len(heatmap.shape) > 2:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
    
    # Resize heatmap if needed
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Normalize heatmap to range 0-255
    if heatmap.max() > 0:
        heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
    
    # Apply colormap to heatmap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on image with alpha blending
    alpha = 0.4
    overlay = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(overlay)
    return pil_img

def generate_spectrogram(audio_path: str) -> Image.Image:
    """
    Generate a spectrogram visualization for audio
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        PIL Image with spectrogram
    """
    try:
        import librosa
        import librosa.display
        
        # Temporarily disable warnings during librosa load
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050, duration=5)
        
        # Generate mel spectrogram
        plt.figure(figsize=(10, 4))
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Plot
        librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close()
        
        # Open as PIL Image
        pil_img = Image.open(buf)
        return pil_img
        
    except Exception as e:
        # Create a simple "Error" image
        img = Image.new('RGB', (400, 200), color = (240, 240, 240))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.text((20, 90), f"Error generating spectrogram:\n{str(e)}", fill=(200, 0, 0))
        return img

def image_to_base64(image: Union[np.ndarray, Image.Image, str]) -> Optional[str]:
    """
    Convert an image to base64 encoding for display in HTML
    
    Args:
        image: Input image (numpy array, PIL Image, or path to file)
        
    Returns:
        Base64 encoded string or None if conversion fails
    """
    try:
        pil_image = None
        
        # Convert based on input type
        if isinstance(image, np.ndarray):
            # For numpy arrays
            if len(image.shape) == 2:
                # Convert grayscale to RGB
                image = np.stack([image, image, image], axis=2)
            elif image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]  # Just use RGB channels
                
            pil_image = Image.fromarray(image.astype(np.uint8))
            
        elif isinstance(image, Image.Image):
            # Already a PIL image
            pil_image = image
            
        elif isinstance(image, str):
            # Path to an image file
            if os.path.isfile(image):
                pil_image = Image.open(image)
            
        # If conversion succeeded, encode to base64
        if pil_image:
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return img_str
            
    except Exception as e:
        print(f"Error converting image to base64: {str(e)}")
        
    return None

def numpy_to_displayable(img_array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to displayable PIL Image
    
    Args:
        img_array: Input image as numpy array
        
    Returns:
        PIL Image ready for display
    """
    if img_array is None:
        return None
        
    # Convert to RGB if it's grayscale
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Convert to uint8 if needed
    if img_array.dtype != np.uint8:
        if img_array.max() > 1.0:
            img_array = img_array.astype(np.uint8)
        else:
            img_array = (img_array * 255).astype(np.uint8)
    
    # Convert numpy array to PIL Image
    img_pil = Image.fromarray(img_array)
    return img_pil

def create_comparison_image(original: np.ndarray, analyzed: np.ndarray, 
                          original_title: str = "Original", analyzed_title: str = "Analysis") -> Image.Image:
    """
    Create a side-by-side comparison of original and analyzed images
    
    Args:
        original: Original image as numpy array
        analyzed: Analyzed/processed image as numpy array
        original_title: Title for the original image
        analyzed_title: Title for the analyzed image
        
    Returns:
        PIL Image with side-by-side comparison
    """
    # Ensure both images are in RGB format
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    elif original.shape[2] == 4:
        original = original[:, :, :3]
        
    if len(analyzed.shape) == 2:
        analyzed = cv2.cvtColor(analyzed, cv2.COLOR_GRAY2RGB)
    elif analyzed.shape[2] == 4:
        analyzed = analyzed[:, :, :3]
    
    # Resize if dimensions don't match
    if original.shape[:2] != analyzed.shape[:2]:
        analyzed = cv2.resize(analyzed, (original.shape[1], original.shape[0]))
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display original image
    ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax1.set_title(original_title)
    ax1.axis('off')
    
    # Display analyzed image
    ax2.imshow(cv2.cvtColor(analyzed, cv2.COLOR_BGR2RGB))
    ax2.set_title(analyzed_title)
    ax2.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)
    
    # Return as PIL Image
    return Image.open(buf)