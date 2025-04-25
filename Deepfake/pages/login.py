import os
import sys
import warnings
import streamlit as st
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress TensorFlow and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning:tensorflow'

# Import custom modules
from utils.auth import init_auth, is_authenticated, login_with_email, login_with_google
import config

# Page config with hidden sidebar
st.set_page_config(
    page_title=f"{config.APP_NAME} - Login",
    page_icon="🔑",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Hide the sidebar completely and set menu icon to hamburger
st.markdown("""
<style>
    /* Hide sidebar completely */
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    /* Ensure hamburger menu shows (not arrow) */
    button[kind="header"] {
        display: none;
    }
    
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
    
    /* Authentication Forms */
    .auth-form {
        max-width: 500px;
        margin: 0 auto;
        padding: 2rem;
        background-color: rgba(52, 152, 219, 0.05);
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
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
    
    /* Footer */
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        color: #7F8C8D;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# Initialize authentication
init_auth()

# Redirect if already logged in
if is_authenticated():
    st.switch_page("app.py")

def login_page():
    st.markdown('<h1 class="main-header">Welcome Back</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; margin-bottom: 2rem;">Sign in to continue to Is This Fake?</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="auth-form">', unsafe_allow_html=True)
    
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Sign In", key="login_button", use_container_width=True):
            if email and password:
                with st.spinner("Signing in..."):
                    success, response = login_with_email(email, password)
                    if success:
                        st.success(response["message"])
                        time.sleep(1)
                        st.session_state.authenticated = True
                        st.switch_page("app.py")
                    else:
                        st.error(response["error"])
            else:
                st.warning("Please enter your email and password")
    
    google_login_url = login_with_google()
    if google_login_url:
        st.markdown("<div style='text-align: center; margin: 1rem 0;'><strong>OR</strong></div>", unsafe_allow_html=True)
        
        google_btn = f'<a href="{google_login_url}" target="_self"><button style="background-color: #4285F4; color: white; padding: 10px 16px; border: none; border-radius: 4px; cursor: pointer; width: 100%; font-weight: bold; display: flex; align-items: center; justify-content: center; gap: 8px;"><svg width="18" height="18" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 48 48"><path fill="#FFC107" d="M43.611,20.083H42V20H24v8h11.303c-1.649,4.657-6.08,8-11.303,8c-6.627,0-12-5.373-12-12c0-6.627,5.373-12,12-12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C12.955,4,4,12.955,4,24c0,11.045,8.955,20,20,20c11.045,0,20-8.955,20-20C44,22.659,43.862,21.35,43.611,20.083z"/><path fill="#FF3D00" d="M6.306,14.691l6.571,4.819C14.655,15.108,18.961,12,24,12c3.059,0,5.842,1.154,7.961,3.039l5.657-5.657C34.046,6.053,29.268,4,24,4C16.318,4,9.656,8.337,6.306,14.691z"/><path fill="#4CAF50" d="M24,44c5.166,0,9.86-1.977,13.409-5.192l-6.19-5.238C29.211,35.091,26.715,36,24,36c-5.202,0-9.619-3.317-11.283-7.946l-6.522,5.025C9.505,39.556,16.227,44,24,44z"/><path fill="#1976D2" d="M43.611,20.083H42V20H24v8h11.303c-0.792,2.237-2.231,4.166-4.087,5.571c0.001-0.001,0.002-0.001,0.003-0.002l6.19,5.238C36.971,39.205,44,34,44,24C44,22.659,43.862,21.35,43.611,20.083z"/></svg>Sign in with Google</button></a>'
        st.markdown(google_btn, unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: center; margin-top: 1.5rem;'>Don't have an account? <a href='/pages/signup.py' target='_self' style='color: #3498DB; text-decoration: none;'>Sign up</a></div>", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: center; margin-top: 0.5rem;'><a href='#' style='color: #3498DB; text-decoration: none;'>Forgot Password?</a></div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Back to home button
    st.markdown("<div style='text-align: center; margin-top: 1.5rem;'>", unsafe_allow_html=True)
    if st.button("← Back to Home", key="back_home"):
        st.switch_page("app.py")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Is This Fake? - Your AI-Powered Deepfake Detection Assistant</p>
        <p>© 2025 Deepfake Detector. All rights reserved.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    login_page()