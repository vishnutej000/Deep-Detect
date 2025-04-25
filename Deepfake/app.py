import os
import sys
import warnings
import base64
import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import time

# Suppress TensorFlow and other warnings - add these at the very top
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging (0=all, 1=info, 2=warnings, 3=errors)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN
warnings.filterwarnings('ignore')  # Suppress warnings from other libraries
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning:tensorflow'

# Import custom modules
from utils.auth import init_auth, is_authenticated, get_current_user
import config

# Page config
st.set_page_config(
    page_title=f"{config.APP_NAME} - Home",
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
    
    /* Authentication Forms */
    .auth-form {
        max-width: 500px;
        margin: 0 auto;
        padding: 2rem;
        background-color: rgba(52, 152, 219, 0.05);
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #2E86C1, #3498DB);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 4rem;
        padding-top: 2rem;
        color: #7F8C8D;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
    }
    
    /* Sidebar */
    .sidebar-content {
        padding: 1.2rem;
        background-color: rgba(52, 152, 219, 0.03);
        border-radius: 8px;
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
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        border-radius: 4px 4px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(52, 152, 219, 0.1);
    }
    
    /* Card effect for sections */
    div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
        padding: 1rem 0;
    }
    
    /* Custom animation for loading spinner */
    div[data-testid="stSpinner"] > div {
        border-color: #2E86C1 transparent transparent transparent;
    }
    
    /* Hide login and signup links in the sidebar */
    [data-testid="stSidebarNav"] ul li:has(a[href="login"]),
    [data-testid="stSidebarNav"] ul li:has(a[href="signup"]) {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize authentication
init_auth()

# Helper function to convert numpy array to displayable image
def numpy_to_base64(img_array):
    """Convert numpy array to base64 encoded image for display"""
    if img_array is None:
        return None
        
    # Convert to RGB if it's grayscale
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Convert numpy array to PIL Image
    img_pil = Image.fromarray(img_array)
    
    return img_pil

# Hide sidebar for non-authenticated users
if not is_authenticated():
    st.markdown("""
    <style>
        /* Hide sidebar completely for non-authenticated users */
        section[data-testid="stSidebar"] {
            display: none;
        }
        
        /* Ensure hamburger menu shows (not arrow) */
        button[kind="header"] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
def sidebar():
    if is_authenticated():
        with st.sidebar:
            st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
            
            user = get_current_user()
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
                    
            if st.button("🔍 Detector", key="detector_button", use_container_width=True):
                st.switch_page("pages/detector.py")
                    
            if st.button("📊 History", key="history_button", use_container_width=True):
                st.switch_page("pages/history.py")
                    
            if st.button("👤 Profile", key="profile_button", use_container_width=True):
                st.switch_page("pages/profile.py")
                
            st.markdown("""
            <div style="margin-top: 2rem;">
            </div>
            """, unsafe_allow_html=True)
                
            if st.button("🚪 Logout", key="logout_button", use_container_width=True):
                st.session_state['logout_initiated'] = True
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# Home page content
def home_page():
    st.markdown('<h1 class="main-header">Welcome to Is This Fake?</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header" style="text-align: center;">Your AI-Powered Deepfake Detection Assistant</p>', unsafe_allow_html=True)
    
    # Create a more visually appealing layout with a hero image
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Hero image placeholder - you can replace this with your logo
        st.image("https://img.icons8.com/color/344/facial-recognition-scan.png", width=150, use_container_width=False)
    
    # Main content based on authentication status
    if is_authenticated():
        st.markdown("""
        <div style="max-width: 800px; margin: 2rem auto; padding: 1.5rem; background-color: rgba(52, 152, 219, 0.05); border-radius: 10px;">
            <h3 style="color: #2E86C1;">Getting Started</h3>
            <p>Welcome to Is This Fake? Use the navigation menu to access the features:</p>
            <ul>
                <li><strong>Detector</strong>: Upload and analyze media files for potential deepfakes</li>
                <li><strong>History</strong>: View your past analysis results</li>
                <li><strong>Profile</strong>: Manage your account settings</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick access buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔍 Go to Detector", use_container_width=True):
                st.switch_page("pages/detector.py")
        with col2:
            if st.button("📊 View History", use_container_width=True):
                st.switch_page("pages/history.py")
        with col3:
            if st.button("👤 My Profile", use_container_width=True):
                st.switch_page("pages/profile.py")
    else:
        # Login/signup options for non-authenticated users
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="padding: 1.5rem; background-color: rgba(52, 152, 219, 0.05); border-radius: 10px; text-align: center;">
                <h3 style="color: #2E86C1;">Already have an account?</h3>
                <p>Sign in to access your detector and analysis history.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔑 Sign In", use_container_width=True):
                st.switch_page("pages/login.py")
        
        with col2:
            st.markdown("""
            <div style="padding: 1.5rem; background-color: rgba(52, 152, 219, 0.05); border-radius: 10px; text-align: center;">
                <h3 style="color: #2E86C1;">New here?</h3>
                <p>Create an account to start detecting deepfakes.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("✏️ Sign Up", use_container_width=True):
                st.switch_page("pages/signup.py")

    # Add information about the app
    st.markdown("""
    <div style="max-width: 800px; margin: 2rem auto; padding: 1.5rem; background-color: rgba(52, 152, 219, 0.05); border-radius: 10px;">
        <h3 style="color: #2E86C1;">How It Works</h3>
        <p>Our deepfake detector uses advanced AI to analyze:</p>
        <ul>
            <li><strong>Images</strong>: Detect manipulated faces and altered images</li>
            <li><strong>Audio</strong>: Identify synthetic or cloned voices</li>
            <li><strong>Video</strong>: Spot inconsistencies in deepfake videos</li>
        </ul>
        <p>Simply upload your media, and our AI will determine if it's authentic or manipulated.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Is This Fake? helps identify potentially manipulated media.</p>
        <p>© 2025 Deepfake Detector. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

# Logout handler
if st.session_state.get('logout_initiated', False):
    from utils.auth import logout
    logout()
    st.session_state.pop('logout_initiated')
    st.rerun()

# Main execution
if __name__ == "__main__":
    sidebar()
    home_page()