import os
import sys
import warnings
import streamlit as st
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress TensorFlow and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning:tensorflow'

# Import custom modules
from utils.auth import init_auth, is_authenticated, get_current_user, logout
import config

# Page config
st.set_page_config(
    page_title=f"{config.APP_NAME} - Profile",
    page_icon="👤",
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
    
    /* Profile section */
    .profile-section {
        padding: 1.5rem;
        background-color: rgba(52, 152, 219, 0.05);
        border-radius: 10px;
        margin-bottom: 1.5rem;
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
</style>
""", unsafe_allow_html=True)

# Initialize authentication
init_auth()

# Redirect if not authenticated
if not is_authenticated():
    st.warning("Please login to access your profile")
    if st.button("Go to Login"):
        st.switch_page("pages/login.py")
    st.stop()

# Init session state for profile page
if "show_password_form" not in st.session_state:
    st.session_state.show_password_form = False

# Helper function to format created_at date safely
def format_date(date_obj):
    """Format a date object in YYYY-MM-DD format, handling various data types"""
    if isinstance(date_obj, str):
        # Handle string format from API
        try:
            if 'T' in date_obj:
                return date_obj.split('T')[0]
            return date_obj[:10]
        except:
            return date_obj
    elif isinstance(date_obj, datetime):
        # Handle datetime object
        return date_obj.strftime("%Y-%m-%d")
    elif date_obj is not None:
        # Handle any other type by converting to string
        return str(date_obj)[:10]
    else:
        return "N/A"

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
            
        if st.button("🔍 Detector", key="detector_button", use_container_width=True):
            st.switch_page("pages/detector.py")
            
        if st.button("📊 History", key="history_button", use_container_width=True):
            st.switch_page("pages/history.py")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Profile page
def profile_page():
    # Get user data
    user = get_current_user()
    
    # Get first/last name from session state if available
    first_name = st.session_state.get("first_name", "")
    last_name = st.session_state.get("last_name", "")
    
    st.markdown('<h1 class="main-header">My Profile</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Manage your account settings</p>', unsafe_allow_html=True)
    
    # User info section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Format the created_at date safely using our helper function
        member_since = format_date(user.created_at) if hasattr(user, "created_at") else "N/A"
        last_sign_in = format_date(user.last_sign_in_at) if hasattr(user, "last_sign_in_at") else "N/A"
        
        # Profile picture/avatar
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="background-color: #3498DB; color: white; width: 100px; height: 100px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 36px; margin: 0 auto;">
                {user.email[0].upper() if user and user.email else "?"}
            </div>
            <h2 style="margin-top: 1rem; margin-bottom: 0.5rem;">{first_name} {last_name}</h2>
            <p style="color: #7F8C8D;">{user.email if user else ""}</p>
            
            <div style="margin-top: 1.5rem;">
                <p><strong>Member since:</strong> {member_since}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Account actions
        st.markdown("### Account Actions")
        if st.button("🔄 Change Password", use_container_width=True):
            # Toggle password change form
            st.session_state.show_password_form = not st.session_state.show_password_form
        
        if st.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()
            
        # Danger zone (collapsible section)
        with st.expander("⚠️ Danger Zone"):
            st.warning("Actions in this section can permanently affect your account.")
            if st.button("🗑️ Delete Account", use_container_width=True):
                if st.session_state.get("confirm_delete_account", False):
                    try:
                        # Delete logic would go here
                        # This is a placeholder - in a real app, you'd call an API to delete the account
                        st.success("Account scheduled for deletion. You will be logged out now.")
                        time.sleep(2)
                        logout()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting account: {str(e)}")
                else:
                    st.session_state.confirm_delete_account = True
                    st.error("⚠️ Are you sure you want to delete your account? This action CANNOT be undone. Click 'Delete Account' again to confirm.")
    
    with col2:
        # Account details
        st.markdown('<div class="profile-section">', unsafe_allow_html=True)
        st.markdown("### Account Details")
        
        # Display user info
        info_data = [
            {"label": "Email", "value": user.email if user else "N/A"},
            {"label": "Account Created", "value": member_since},
            {"label": "Last Sign In", "value": last_sign_in}
        ]
        
        for item in info_data:
            st.markdown(f"""
            <div style="margin-bottom: 15px;">
                <p style="color: #7F8C8D; margin-bottom: 5px; font-size: 0.9em;">{item["label"]}</p>
                <p style="font-weight: bold; font-size: 1.1em; margin-top: 0;">{item["value"]}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Password change form
        if st.session_state.get("show_password_form", False):
            st.markdown('<div class="profile-section">', unsafe_allow_html=True)
            st.markdown("### Change Password")
            
            with st.form("password_change_form"):
                current_password = st.text_input("Current Password", type="password")
                new_password = st.text_input("New Password", type="password",
                                          help="Must be at least 8 characters with 1 uppercase, 1 lowercase, and 1 number")
                confirm_password = st.text_input("Confirm New Password", type="password")
                
                submit = st.form_submit_button("Update Password")
                
                if submit:
                    if not current_password or not new_password or not confirm_password:
                        st.error("Please fill all fields")
                    elif new_password != confirm_password:
                        st.error("New passwords do not match")
                    else:
                        # Password change logic would go here
                        # This is a placeholder - in a real app, you'd call an API to change the password
                        st.success("Password updated successfully!")
                        st.session_state.show_password_form = False
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Personal information section
        st.markdown('<div class="profile-section">', unsafe_allow_html=True)
        st.markdown("### Personal Information")
        
        # Use placeholders or session state values
        first_name_input = st.text_input("First Name", value=first_name)
        last_name_input = st.text_input("Last Name", value=last_name)
        
        if st.button("Save Changes"):
            # Save to session state (in a real app, you'd save to the database)
            st.session_state.first_name = first_name_input
            st.session_state.last_name = last_name_input
            st.success("Profile updated successfully!")
            # Refresh the page to show updated values
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Usage statistics section - assuming you're using the database
        st.markdown('<div class="profile-section">', unsafe_allow_html=True)
        st.markdown("### Usage Statistics")
        
        try:
            # Try to get usage statistics
            supabase = st.session_state.supabase
            
            # Query the analysis_results table
            response = None
            try:
                response = supabase.table("analysis_results").select("media_type").eq("user_id", user.id).execute()
            except Exception as e:
                # Table might not exist yet
                if "does not exist" not in str(e):
                    st.error(f"Error querying database: {str(e)}")
            
            # Display metrics based on response
            if response and response.data:
                # Count by media type
                media_types = [item['media_type'] for item in response.data]
                image_count = media_types.count('image') if 'image' in media_types else 0
                audio_count = media_types.count('audio') if 'audio' in media_types else 0
                video_count = media_types.count('video') if 'video' in media_types else 0
                total = len(media_types)
                
                # Display as metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Analyses", total)
                with col2:
                    st.metric("Most Analyzed", "Images" if image_count >= audio_count and image_count >= video_count else
                                             "Audio" if audio_count >= image_count and audio_count >= video_count else "Videos")
                with col3:
                    st.metric("Analysis This Month", "Coming soon")
                
                # Create a simple bar chart
                st.markdown("### Media Type Breakdown")
                
                # Use altair or matplotlib for more advanced charts
                st.bar_chart({"Images": image_count, "Audio": audio_count, "Videos": video_count})
                
            else:
                st.info("No analysis data available yet. Start analyzing media to see statistics here.")
                
        except Exception as e:
            st.info("Usage statistics will appear here once you start analyzing media.")
            
        st.markdown('</div>', unsafe_allow_html=True)

# Run the sidebar and page
if __name__ == "__main__":
    sidebar()
    profile_page()