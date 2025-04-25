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
from utils.auth import init_auth, is_authenticated, signup
import config

# Page config with hidden sidebar
st.set_page_config(
    page_title=f"{config.APP_NAME} - Sign Up",
    page_icon="✏️",
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
    
    /* Progress indicator */
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
    }
    
    .step {
        text-align: center;
    }
    
    .step-number {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background-color: #eee;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .active .step-number {
        background-color: #3498DB;
        color: white;
    }
    
    .completed .step-number {
        background-color: #2ECC71;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize authentication
init_auth()

# Redirect if already logged in
if is_authenticated():
    st.switch_page("app.py")

def signup_page():
    st.markdown('<h1 class="main-header">Create Your Account</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; margin-bottom: 2rem;">Join Is This Fake? to start detecting deepfakes</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="auth-form">', unsafe_allow_html=True)
    
    # Multi-step form
    current_step = st.session_state.get('signup_step', 1)
    
    # Step indicator
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="step {'active' if current_step == 1 else 'completed' if current_step > 1 else ''}">
            <div class="step-number">{'✓' if current_step > 1 else '1'}</div>
            <div class="step-label">Account Details</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="step {'active' if current_step == 2 else 'completed' if current_step > 2 else ''}">
            <div class="step-number">{'✓' if current_step > 2 else '2'}</div>
            <div class="step-label">Profile Information</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Step 1: Account details
    if current_step == 1:
        # Save form values to session state
        if 'signup_email' not in st.session_state:
            st.session_state.signup_email = ''
        if 'signup_password' not in st.session_state:
            st.session_state.signup_password = ''
        if 'signup_confirm_password' not in st.session_state:
            st.session_state.signup_confirm_password = ''
        
        st.subheader("Account Details")
        
        email = st.text_input("Email Address", value=st.session_state.signup_email, key="email_input")
        password = st.text_input("Create Password", type="password", value=st.session_state.signup_password, key="password_input",
                              help="Must be at least 8 characters with 1 uppercase, 1 lowercase, and 1 number")
        confirm_password = st.text_input("Confirm Password", type="password", value=st.session_state.signup_confirm_password, key="confirm_password_input")
        
        # Store values in session state
        st.session_state.signup_email = email
        st.session_state.signup_password = password
        st.session_state.signup_confirm_password = confirm_password
        
        # Validate and proceed
        if st.button("Continue", use_container_width=True):
            if not email or not password or not confirm_password:
                st.warning("Please fill all required fields")
            elif password != confirm_password:
                st.error("Passwords do not match")
            else:
                st.session_state.signup_step = 2
                st.rerun()
    
    # Step 2: Profile information
    elif current_step == 2:
        # Save form values to session state
        if 'signup_first_name' not in st.session_state:
            st.session_state.signup_first_name = ''
        if 'signup_last_name' not in st.session_state:
            st.session_state.signup_last_name = ''
        
        st.subheader("Profile Information")
        
        col1, col2 = st.columns(2)
        with col1:
            first_name = st.text_input("First Name", value=st.session_state.signup_first_name, key="first_name_input")
        with col2:
            last_name = st.text_input("Last Name", value=st.session_state.signup_last_name, key="last_name_input")
        
        # Store values in session state
        st.session_state.signup_first_name = first_name
        st.session_state.signup_last_name = last_name
        
        # Terms and conditions
        agree = st.checkbox("I agree to the Terms of Service and Privacy Policy")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Back", use_container_width=True):
                st.session_state.signup_step = 1
                st.rerun()
                
        with col2:
            if st.button("Create Account", key="create_account_button", disabled=not agree, use_container_width=True):
                if not first_name or not last_name:
                    st.warning("Please fill all required fields")
                else:
                    with st.spinner("Creating your account..."):
                        # Get values from session state
                        email = st.session_state.signup_email
                        password = st.session_state.signup_password
                        
                        # Call signup function
                        success, response = signup(email, password)
                        
                        if success:
                            # Store first and last name in session state (for future use)
                            st.session_state.first_name = first_name
                            st.session_state.last_name = last_name
                            
                            st.success(response["message"])
                            st.info("You can now sign in with your credentials")
                            time.sleep(2)
                            
                            # Reset step
                            st.session_state.pop('signup_step', None)
                            st.session_state.pop('signup_email', None)
                            st.session_state.pop('signup_password', None)
                            st.session_state.pop('signup_confirm_password', None)
                            st.session_state.pop('signup_first_name', None)
                            st.session_state.pop('signup_last_name', None)
                            
                            # Redirect to login
                            st.switch_page("pages/login.py")
                        else:
                            st.error(response["error"])
    
    st.markdown("<div style='text-align: center; margin-top: 1.5rem;'>Already have an account? <a href='/pages/login.py' target='_self' style='color: #3498DB; text-decoration: none;'>Sign in</a></div>", unsafe_allow_html=True)
    
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
    signup_page()