import os
import sys
import warnings
import streamlit as st
import tempfile
from PIL import Image
import numpy as np
import base64
import time
import uuid

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress TensorFlow and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# Import custom modules
from utils.auth import init_auth, is_authenticated, get_current_user
from utils.storage import upload_file, get_file_url
from detectors.image_detector import analyze_image
from detectors.audio_detector import analyze_audio
from detectors.video_detector import analyze_video
import config

# Page config
st.set_page_config(
    page_title=f"{config.APP_NAME} - Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    /* Main UI Elements */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2E86C1, #3498DB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.2rem;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.6rem;
        color: #3498DB;
        margin-bottom: 1.2rem;
        border-bottom: 2px solid rgba(52, 152, 219, 0.3);
        padding-bottom: 0.5rem;
    }
    
    /* Result Boxes */
    .result-box {
        padding: 1.8rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .result-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    
    .fake-result {
        background-color: rgba(231, 76, 60, 0.1);
        border-left: 5px solid #E74C3C;
    }
    
    .real-result {
        background-color: rgba(46, 204, 113, 0.1);
        border-left: 5px solid #2ECC71;
    }
    
    /* Feature Analysis */
    .feature-box {
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 0.8rem;
        background-color: rgba(52, 152, 219, 0.05);
        border: 1px solid rgba(52, 152, 219, 0.1);
        transition: all 0.2s ease;
    }
    
    .feature-box:hover {
        background-color: rgba(52, 152, 219, 0.1);
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #2E86C1;
        color: white;
        font-weight: 500;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #3498DB;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* File Uploader */
    .uploadedFile {
        border: 1px solid #3498DB;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize authentication
init_auth()

# Redirect if not authenticated
if not is_authenticated():
    st.warning("Please login to access the detector")
    if st.button("Go to Login"):
        st.switch_page("pages/login.py")
    st.stop()

# Helper function for image conversion
def numpy_to_base64(img_array):
    """Convert numpy array to base64 encoded image for display"""
    if img_array is None:
        return None
        
    # Convert to RGB if it's grayscale
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array, img_array, img_array], axis=2)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Just use RGB channels
    
    # Convert numpy array to PIL Image
    img_pil = Image.fromarray(img_array.astype(np.uint8))
    return img_pil

# Functions to display results
def display_image_results(result):
    if result.get("error"):
        st.error(f"Error during analysis: {result['error']}")
        return
    
    # Display result with styling
    result_class = "fake-result" if result["is_fake"] else "real-result"
    confidence = result["confidence"] * 100
    
    st.markdown(f"""
    <div class="result-box {result_class}">
        <h2 style="margin-top: 0;">Detection Result: {result["label"]}</h2>
        <div style="display: flex; align-items: center; margin: 1rem 0;">
            <div style="flex-grow: 1; height: 10px; background-color: #EDF0F1; border-radius: 5px; overflow: hidden;">
                <div style="width: {confidence}%; height: 100%; background: {'linear-gradient(90deg, #E74C3C, #E57373)' if result["is_fake"] else 'linear-gradient(90deg, #2ECC71, #81C784)'};"></div>
            </div>
            <div style="margin-left: 15px; font-weight: bold; min-width: 80px; font-size: 1.2rem;">
                {confidence:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display heatmap if available
    if "heatmap" in result and result["heatmap"] is not None:
        st.markdown("### Detection Visualization")
        st.markdown("The heatmap shows areas that contributed to the classification decision:")
        
        # Process the heatmap to ensure it's displayable
        if isinstance(result["heatmap"], np.ndarray):
            heatmap_img = numpy_to_base64(result["heatmap"])
            if heatmap_img:
                st.image(heatmap_img, use_container_width=True)
        else:
            # Try to display as is if it's not a numpy array
            try:
                st.image(result["heatmap"], use_container_width=True)
            except Exception as e:
                st.error(f"Could not display heatmap: {str(e)}")
    
    # Display feature analysis
    if "features" in result:
        st.markdown("### Feature Analysis")
        
        for feature_name, feature_data in result["features"].items():
            score = feature_data["score"]
            description = feature_data["description"]
            
            st.markdown(f"""
            <div class="feature-box">
                <h4 style="margin-top: 0;">{feature_name.replace('_', ' ').title()}</h4>
                <p style="margin-bottom: 0.8rem;">{description}</p>
                <div style="display: flex; align-items: center;">
                    <div style="flex-grow: 1;">
                        <div style="height: 8px; background-color: #EDF0F1; border-radius: 4px; overflow: hidden;">
                            <div style="width: {score*100}%; height: 100%; background: {'linear-gradient(90deg, #E74C3C, #E57373)' if score > 0.5 else 'linear-gradient(90deg, #2ECC71, #81C784)'};"></div>
                        </div>
                    </div>
                    <div style="margin-left: 10px; min-width: 50px; font-weight: bold;">
                        {score:.2f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def display_audio_results(result):
    if result.get("error"):
        st.error(f"Error during analysis: {result['error']}")
        return
    
    # Display result with styling
    result_class = "fake-result" if result["is_fake"] else "real-result"
    confidence = result["confidence"] * 100
    
    st.markdown(f"""
    <div class="result-box {result_class}">
        <h2 style="margin-top: 0;">Detection Result: {result["label"]}</h2>
        <div style="display: flex; align-items: center; margin: 1rem 0;">
            <div style="flex-grow: 1; height: 10px; background-color: #EDF0F1; border-radius: 5px; overflow: hidden;">
                <div style="width: {confidence}%; height: 100%; background: {'linear-gradient(90deg, #E74C3C, #E57373)' if result["is_fake"] else 'linear-gradient(90deg, #2ECC71, #81C784)'};"></div>
            </div>
            <div style="margin-left: 15px; font-weight: bold; min-width: 80px; font-size: 1.2rem;">
                {confidence:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display spectrogram if available
    if "spectrogram" in result and result["spectrogram"] is not None:
        st.markdown("### Audio Analysis")
        st.markdown("The spectrogram visualization shows patterns that may indicate deepfake audio:")
        
        # Process the spectrogram to ensure it's displayable
        if isinstance(result["spectrogram"], np.ndarray):
            spec_img = numpy_to_base64(result["spectrogram"])
            if spec_img:
                st.image(spec_img, use_container_width=True)
        else:
            # Try to display as is if it's not a numpy array
            try:
                st.image(result["spectrogram"], use_container_width=True)
            except Exception as e:
                st.error(f"Could not display spectrogram: {str(e)}")
    
    # Display feature analysis
    if "features" in result:
        st.markdown("### Feature Analysis")
        
        for feature_name, feature_data in result["features"].items():
            score = feature_data["score"]
            description = feature_data["description"]
            
            st.markdown(f"""
            <div class="feature-box">
                <h4 style="margin-top: 0;">{feature_name.replace('_', ' ').title()}</h4>
                <p style="margin-bottom: 0.8rem;">{description}</p>
                <div style="display: flex; align-items: center;">
                    <div style="flex-grow: 1;">
                        <div style="height: 8px; background-color: #EDF0F1; border-radius: 4px; overflow: hidden;">
                            <div style="width: {score*100}%; height: 100%; background: {'linear-gradient(90deg, #E74C3C, #E57373)' if score > 0.5 else 'linear-gradient(90deg, #2ECC71, #81C784)'};"></div>
                        </div>
                    </div>
                    <div style="margin-left: 10px; min-width: 50px; font-weight: bold;">
                        {score:.2f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def display_video_results(result):
    if result.get("error"):
        st.error(f"Error during analysis: {result['error']}")
        return
    
    # Display result with styling
    result_class = "fake-result" if result["is_fake"] else "real-result"
    confidence = result["confidence"] * 100
    
    st.markdown(f"""
    <div class="result-box {result_class}">
        <h2 style="margin-top: 0;">Detection Result: {result["label"]}</h2>
        <div style="display: flex; align-items: center; margin: 1rem 0;">
            <div style="flex-grow: 1; height: 10px; background-color: #EDF0F1; border-radius: 5px; overflow: hidden;">
                <div style="width: {confidence}%; height: 100%; background: {'linear-gradient(90deg, #E74C3C, #E57373)' if result["is_fake"] else 'linear-gradient(90deg, #2ECC71, #81C784)'};"></div>
            </div>
            <div style="margin-left: 15px; font-weight: bold; min-width: 80px; font-size: 1.2rem;">
                {confidence:.1f}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display heatmap if available
    if "heatmap" in result and result["heatmap"] is not None:
        st.markdown("### Detection Visualization")
        st.markdown("The heatmap shows areas that contributed to the classification decision:")
        
        # Process the heatmap to ensure it's displayable
        if isinstance(result["heatmap"], np.ndarray):
            heatmap_img = numpy_to_base64(result["heatmap"])
            if heatmap_img:
                st.image(heatmap_img, use_container_width=True)
        else:
            # Try to display as is if it's not a numpy array
            try:
                st.image(result["heatmap"], use_container_width=True)
            except Exception as e:
                st.error(f"Could not display heatmap: {str(e)}")
    
    # Display feature analysis
    if "features" in result:
        st.markdown("### Feature Analysis")
        
        for feature_name, feature_data in result["features"].items():
            score = feature_data["score"]
            description = feature_data["description"]
            
            st.markdown(f"""
            <div class="feature-box">
                <h4 style="margin-top: 0;">{feature_name.replace('_', ' ').title()}</h4>
                <p style="margin-bottom: 0.8rem;">{description}</p>
                <div style="display: flex; align-items: center;">
                    <div style="flex-grow: 1;">
                        <div style="height: 8px; background-color: #EDF0F1; border-radius: 4px; overflow: hidden;">
                            <div style="width: {score*100}%; height: 100%; background: {'linear-gradient(90deg, #E74C3C, #E57373)' if score > 0.5 else 'linear-gradient(90deg, #2ECC71, #81C784)'};"></div>
                        </div>
                    </div>
                    <div style="margin-left: 10px; min-width: 50px; font-weight: bold;">
                        {score:.2f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def save_analysis_result(media_type, file_path, file_name, result):
    """Save analysis result to Supabase database"""
    if is_authenticated():
        user = get_current_user()
        if user:
            try:
                # Get file size
                file_size = os.path.getsize(file_path)
                
                # Upload file to storage if needed
                storage_path = f"{user.id}/{str(uuid.uuid4())}/{file_name}"
                upload_file(file_path, storage_path, media_type)
                
                # Create result JSON for database
                db_result = {
                    "is_fake": result["is_fake"],
                    "confidence": result["confidence"],
                    "features": result.get("features", {}),
                    "timestamp": time.time()
                }
                
                # Add record to Supabase
                supabase = st.session_state.supabase
                data = {
                    "user_id": user.id,
                    "media_type": media_type,
                    "result": db_result,
                    "file_path": storage_path,
                    "file_name": file_name,
                    "file_size": file_size
                }
                
                supabase.table("analysis_results").insert(data).execute()
                st.success("Analysis result saved to your history")
            except Exception as e:
                st.error(f"Error saving analysis result: {str(e)}")

# Sidebar
def sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        user = get_current_user()
        if user:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                <div style="background-color: #3498DB; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 18px;">
                    {user.email[0].upper() if user and user.email else "?"}
                </div>
                <div>
                    <p style="margin: 0; font-weight: bold;">Welcome</p>
                    <p style="margin: 0; font-size: 0.9rem; color: #7F8C8D; overflow: hidden; text-overflow: ellipsis; max-width: 180px;">{user.email if user else ""}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("""
        <div style="margin: 1.5rem 0; padding-bottom: 1rem; border-bottom: 1px solid rgba(49, 51, 63, 0.2);">
            <h3 style="margin-bottom: 0.8rem; color: #2E86C1;">Navigation</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🏠 Home", key="home_button", use_container_width=True):
            st.switch_page("app.py")
            
        if st.button("📊 History", key="history_button", use_container_width=True):
            st.switch_page("pages/history.py")
            
        if st.button("👤 Profile", key="profile_button", use_container_width=True):
            st.switch_page("pages/profile.py")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Main detector functionality
def detector_page():
    st.markdown('<h1 class="main-header">Media Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload media to detect potential deepfakes</p>', unsafe_allow_html=True)
    
    # Media type selection
    media_type = st.radio("Select media type to analyze:", ["Image", "Audio", "Video"], horizontal=True)
    
    # Render the appropriate detector based on selection
    if media_type == "Image":
        image_detector_page()
    elif media_type == "Audio":
        audio_detector_page()
    elif media_type == "Video":
        video_detector_page()

def image_detector_page():
    st.markdown('<p class="sub-header">Image Deepfake Detection</p>', unsafe_allow_html=True)
    
    # More user-friendly instructions
    st.markdown("""
    <div style="padding: 1rem; background-color: rgba(52, 152, 219, 0.05); border-radius: 8px; margin-bottom: 1.5rem;">
        <h4 style="margin-top: 0; color: #2E86C1;">How it works</h4>
        <p>Upload an image to detect if it contains manipulated faces or other artificial elements.</p>
        <ul>
            <li>Supported formats: JPG, JPEG, PNG</li>
            <li>Best results with clear facial images</li>
            <li>Maximum file size: 5MB</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=config.ALLOWED_EXTENSIONS["image"])
    
    if uploaded_file is not None:
        # Display uploaded image
        st.markdown("### Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Save uploaded file to temp location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        
        # Add details about the image
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Size", f"{round(os.path.getsize(temp_file.name) / 1024, 1)} KB")
        with col2:
            st.metric("Dimensions", f"{image.width} × {image.height}")
        with col3:
            st.metric("Format", uploaded_file.name.split('.')[-1].upper())
        
        # Analyze button
        if st.button("🔍 Analyze Image", key="analyze_image_button"):
            with st.spinner("Analyzing image for potential manipulation..."):
                # Run analysis
                result = analyze_image(temp_file.name)
                
                # Fix heatmap display issue by converting numpy array to PIL Image
                if "heatmap" in result and isinstance(result["heatmap"], np.ndarray):
                    result["heatmap"] = numpy_to_base64(result["heatmap"])
                
                # Display results
                display_image_results(result)
                
                # Save analysis result
                save_analysis_result("image", temp_file.name, uploaded_file.name, result)
        
        # Clean up temp file when done
        try:
            os.unlink(temp_file.name)
        except:
            pass
    else:
        # Placeholder when no image is uploaded
        st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; background-color: rgba(52, 152, 219, 0.05); border-radius: 10px; border: 1px dashed #3498DB;">
            <img src="https://img.icons8.com/color/96/000000/add-image.png" style="opacity: 0.7; margin-bottom: 15px;">
            <p style="color: #7F8C8D; text-align: center;">Drag and drop an image file<br>or click the "Browse files" button above.</p>
        </div>
        """, unsafe_allow_html=True)

def audio_detector_page():
    st.markdown('<p class="sub-header">Audio Deepfake Detection</p>', unsafe_allow_html=True)
    
    # More user-friendly instructions
    st.markdown("""
    <div style="padding: 1rem; background-color: rgba(52, 152, 219, 0.05); border-radius: 8px; margin-bottom: 1.5rem;">
        <h4 style="margin-top: 0; color: #2E86C1;">How it works</h4>
        <p>Upload an audio file to detect if it contains synthetic or manipulated voices.</p>
        <ul>
            <li>Supported formats: MP3, WAV, OGG</li>
            <li>Best results with clear speech recordings</li>
            <li>Maximum file size: 10MB</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=config.ALLOWED_EXTENSIONS["audio"])
    
    if uploaded_file is not None:
        # Create a nice container for the audio player
        st.markdown("""
        <div style="padding: 1.5rem; background-color: rgba(52, 152, 219, 0.05); border-radius: 8px; margin-bottom: 1.5rem;">
            <h3 style="margin-top: 0; color: #2E86C1;">Uploaded Audio</h3>
        """, unsafe_allow_html=True)
        
        # Display audio player
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Save uploaded file to temp location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        
        # Add file details
        col1, col2 = st.columns(2)
        with col1:
            st.metric("File Size", f"{round(os.path.getsize(temp_file.name) / 1024, 1)} KB")
        with col2:
            st.metric("Format", uploaded_file.name.split('.')[-1].upper())
        
        # Analyze button
        if st.button("🔍 Analyze Audio", key="analyze_audio_button"):
            with st.spinner("Analyzing audio for potential manipulation..."):
                # Run analysis
                result = analyze_audio(temp_file.name)
                
                # Fix spectrogram display issue by converting numpy array to PIL Image
                if "spectrogram" in result and isinstance(result["spectrogram"], np.ndarray):
                    result["spectrogram"] = numpy_to_base64(result["spectrogram"])
                
                # Display results
                display_audio_results(result)
                
                # Save analysis result
                save_analysis_result("audio", temp_file.name, uploaded_file.name, result)
        
        # Clean up temp file when done
        try:
            os.unlink(temp_file.name)
        except:
            pass
    else:
        # Placeholder when no audio is uploaded
        st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; background-color: rgba(52, 152, 219, 0.05); border-radius: 10px; border: 1px dashed #3498DB;">
            <img src="https://img.icons8.com/color/96/000000/audio-wave--v1.png" style="opacity: 0.7; margin-bottom: 15px;">
            <p style="color: #7F8C8D; text-align: center;">Drag and drop an audio file<br>or click the "Browse files" button above.</p>
        </div>
        """, unsafe_allow_html=True)

def video_detector_page():
    st.markdown('<p class="sub-header">Video Deepfake Detection</p>', unsafe_allow_html=True)
    
    # More user-friendly instructions
    st.markdown("""
    <div style="padding: 1rem; background-color: rgba(52, 152, 219, 0.05); border-radius: 8px; margin-bottom: 1.5rem;">
        <h4 style="margin-top: 0; color: #2E86C1;">How it works</h4>
        <p>Upload a video file to detect facial manipulation, inconsistencies, and other deepfake indicators.</p>
        <ul>
            <li>Supported formats: MP4, MOV, AVI</li>
            <li>Best results with clear facial footage</li>
            <li>Maximum file size: 50MB</li>
            <li>Analysis can take longer for video files</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a video file", type=config.ALLOWED_EXTENSIONS["video"])
    
    if uploaded_file is not None:
        # Create a nice container for the video player
        st.markdown("""
        <div style="padding: 1.5rem; background-color: rgba(52, 152, 219, 0.05); border-radius: 8px; margin-bottom: 1.5rem;">
            <h3 style="margin-top: 0; color: #2E86C1;">Uploaded Video</h3>
        """, unsafe_allow_html=True)
        
        # Display video player
        st.video(uploaded_file, format=f"video/{uploaded_file.name.split('.')[-1]}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Save uploaded file to temp location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}")
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()
        
        # Add file details
        col1, col2 = st.columns(2)
        with col1:
            st.metric("File Size", f"{round(os.path.getsize(temp_file.name) / 1024 / 1024, 2)} MB")
        with col2:
            st.metric("Format", uploaded_file.name.split('.')[-1].upper())
        
        # Analyze button with improved styling
        if st.button("🔍 Analyze Video", key="analyze_video_button", help="Click to start deepfake detection analysis"):
            with st.spinner("Analyzing video for potential manipulation... This may take several moments."):
                # Run analysis
                result = analyze_video(temp_file.name)
                
                # Fix heatmap display issue by converting numpy array to PIL Image
                if "heatmap" in result and isinstance(result["heatmap"], np.ndarray):
                    result["heatmap"] = numpy_to_base64(result["heatmap"])
                
                # Display results
                display_video_results(result)
                
                # Save analysis result
                save_analysis_result("video", temp_file.name, uploaded_file.name, result)
        
        # Clean up temp file when done
        try:
            os.unlink(temp_file.name)
        except:
            pass
    else:
        # Placeholder when no video is uploaded
        st.markdown("""
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; background-color: rgba(52, 152, 219, 0.05); border-radius: 10px; border: 1px dashed #3498DB;">
            <img src="https://img.icons8.com/color/96/000000/video.png" style="opacity: 0.7; margin-bottom: 15px;">
            <p style="color: #7F8C8D; text-align: center;">Drag and drop a video file<br>or click the "Browse files" button above.</p>
        </div>
        """, unsafe_allow_html=True)

# Run the sidebar and page
if __name__ == "__main__":
    sidebar()
    detector_page()