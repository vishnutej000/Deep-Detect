import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from typing import Optional, Union
import warnings

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def generate_spectrogram(audio_path: str) -> np.ndarray:
    """
    Generate a spectrogram visualization for audio
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Numpy array containing the spectrogram image
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, duration=10, mono=True)
        
        # Set up figure
        plt.figure(figsize=(10, 4))
        
        # Generate mel spectrogram
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
        
        # Convert to numpy array
        img = Image.open(buf)
        img_array = np.array(img)
        
        plt.close()
        
        return img_array
        
    except Exception as e:
        # Create a simple "Error" image
        plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, f"Error generating spectrogram: {str(e)}", 
                ha='center', va='center', fontsize=12, color='red')
        plt.tight_layout()
        
        # Convert to numpy array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        img_array = np.array(img)
        
        plt.close()
        
        return img_array

def spectrogram_to_base64(audio_path: str) -> Optional[str]:
    """
    Generate a spectrogram and convert to base64 for displaying in HTML
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Base64 encoded string of the spectrogram image
    """
    try:
        # Generate spectrogram
        spec_img = generate_spectrogram(audio_path)
        
        # Convert to PIL Image
        pil_img = Image.fromarray(spec_img)
        
        # Save to buffer
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        
        # Convert to base64
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return img_str
    
    except Exception as e:
        print(f"Error generating spectrogram: {str(e)}")
        return None

def extract_audio_features(audio_path: str) -> dict:
    """
    Extract audio features for analysis
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary with audio features
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, duration=30, mono=True)
        
        # Extract features
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # Zero crossing rate
        zero_crossings = librosa.feature.zero_crossing_rate(y=y)[0]
        zcr_mean = np.mean(zero_crossings)
        zcr_std = np.std(zero_crossings)
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        sc_mean = np.mean(spectral_centroids)
        sc_std = np.std(spectral_centroids)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        sr_mean = np.mean(spectral_rolloff)
        sr_std = np.std(spectral_rolloff)
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        
        # Collect features
        features = {
            "rms_mean": float(rms_mean),
            "rms_std": float(rms_std),
            "zcr_mean": float(zcr_mean),
            "zcr_std": float(zcr_std),
            "sc_mean": float(sc_mean),
            "sc_std": float(sc_std),
            "sr_mean": float(sr_mean),
            "sr_std": float(sr_std),
            "mfcc_means": mfcc_means.tolist(),
            "mfcc_stds": mfcc_stds.tolist(),
            "duration": float(len(y) / sr)
        }
        
        return features
    
    except Exception as e:
        print(f"Error extracting audio features: {str(e)}")
        return {"error": str(e)}

def plot_audio_features(audio_path: str) -> plt.Figure:
    """
    Create a plot of audio features for analysis
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Matplotlib figure with audio visualizations
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, duration=30, mono=True)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Plot waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[0])
        axes[0].set_title('Waveform')
        axes[0].set(xlabel=None)
        
        # Plot spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[1])
        fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
        axes[1].set_title('Mel Spectrogram')
        axes[1].set(xlabel=None)
        
        # Plot chromagram
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        img = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=axes[2])
        fig.colorbar(img, ax=axes[2])
        axes[2].set_title('Chromagram')
        
        plt.tight_layout()
        
        return fig
        
    except Exception as e:
        # Create error figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"Error plotting audio features: {str(e)}", 
               ha='center', va='center', fontsize=12, color='red')
        ax.axis('off')
        return fig