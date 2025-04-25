import os
import sys
import warnings
import streamlit as st
import pandas as pd
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress TensorFlow and other warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning:tensorflow'

# Import custom modules
from utils.auth import init_auth, is_authenticated, get_current_user
from utils.storage import get_file_url
import config

# Page config
st.set_page_config(
    page_title=f"{config.APP_NAME} - History",
    page_icon="📊",
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
    
    /* Empty state styling */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 4rem 2rem;
        background-color: rgba(52, 152, 219, 0.05);
        border-radius: 10px;
        text-align: center;
        margin: 2rem 0;
    }
    
    .empty-state-icon {
        font-size: 48px;
        color: #3498DB;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize authentication
init_auth()

# Redirect if not authenticated
if not is_authenticated():
    st.warning("Please login to access your history")
    if st.button("Go to Login"):
        st.switch_page("pages/login.py")
    st.stop()

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
            
        if st.button("👤 Profile", key="profile_button", use_container_width=True):
            st.switch_page("pages/profile.py")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Function to check if table exists in database
def check_table_exists(table_name):
    try:
        # Try to query the table with a limit of 0 to just check existence
        st.session_state.supabase.table(table_name).select("id").limit(0).execute()
        return True
    except Exception as e:
        if "does not exist" in str(e):
            return False
        # For other errors, re-raise them
        raise e

# Display setup guide when table doesn't exist
def display_setup_guide():
    st.markdown("""
    <div style="padding: 2rem; background-color: rgba(52, 152, 219, 0.05); border-radius: 10px; margin-bottom: 1.5rem;">
        <h3 style="color: #2E86C1; margin-top: 0;">Database Setup Required</h3>
        <p>The analysis history database table needs to be created. You can run the database initialization script to create it.</p>
        
        <h4>How to set up the database:</h4>
        <ol>
            <li>Run the database initialization utility: <code>streamlit run utils/init_db.py</code></li>
            <li>Click the "Initialize Database" button</li>
            <li>Wait for confirmation that the table was created</li>
            <li>Return to this page</li>
        </ol>
        
        <p>After setting up the database, refresh this page to view your analysis history.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔧 Run Database Setup", use_container_width=True):
            # If the init_db.py file exists at the expected path
            try:
                file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utils", "init_db.py")
                if os.path.exists(file_path):
                    st.info("Opening database setup in a new tab...")
                    # We can't directly run it, but we can guide the user
                    st.markdown(f"<script>window.open('http://localhost:8501/utils/init_db.py', '_blank');</script>", unsafe_allow_html=True)
                else:
                    st.error(f"Database setup file not found at {file_path}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        if st.button("🔍 Go to Detector", use_container_width=True):
            st.switch_page("pages/detector.py")

# Empty state display for when there's no history
def display_empty_state():
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">📊</div>
        <h2>No Analysis History Yet</h2>
        <p>Once you analyze images, audio, or videos, your results will appear here.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔍 Start Analyzing", use_container_width=True):
        st.switch_page("pages/detector.py")

# History page
def history_page():
    st.markdown('<h1 class="main-header">Analysis History</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">View and manage your past analysis results</p>', unsafe_allow_html=True)
    
    try:
        # Check if table exists first
        if not check_table_exists("analysis_results"):
            display_setup_guide()
            return
            
        # Fetch user's analysis history from Supabase
        supabase = st.session_state.supabase
        user = get_current_user()
        
        response = supabase.table("analysis_results").select("*").eq("user_id", user.id).order("created_at", desc=True).execute()
        
        if response and response.data:
            # Create tabs for different media types
            tab_all, tab_image, tab_audio, tab_video = st.tabs(["All", "Images", "Audio", "Videos"])
            
            # Process data for display
            df = pd.DataFrame(response.data)
            df["created_at"] = pd.to_datetime(df["created_at"])
            df["date"] = df["created_at"].dt.strftime("%Y-%m-%d %H:%M")
            
            # Safely extract nested values
            def safe_get_is_fake(result):
                if isinstance(result, dict):
                    return result.get("is_fake", False)
                return False
                
            def safe_get_confidence(result):
                if isinstance(result, dict):
                    return result.get("confidence", 0)
                return 0
            
            df["is_fake"] = df["result"].apply(safe_get_is_fake)
            df["confidence"] = df["result"].apply(safe_get_confidence)
            
            # Format for display
            display_df = df[["date", "media_type", "file_name", "is_fake", "confidence"]].copy()
            display_df["confidence"] = (display_df["confidence"] * 100).round(2).astype(str) + "%"
            display_df["is_fake"] = display_df["is_fake"].apply(lambda x: "🚫 Fake" if x else "✅ Real")
            display_df.columns = ["Date", "Type", "Filename", "Result", "Confidence"]
            
            # Custom styling function
            def highlight_fake(val):
                if val == "🚫 Fake":
                    return 'background-color: rgba(231, 76, 60, 0.1); color: #C0392B; font-weight: bold'
                elif val == "✅ Real":
                    return 'background-color: rgba(46, 204, 113, 0.1); color: #27AE60; font-weight: bold'
                return ''
            
            # All tab
            with tab_all:
                if not display_df.empty:
                    st.dataframe(
                        display_df.style.applymap(highlight_fake, subset=["Result"]),
                        use_container_width=True,
                        height=400
                    )
                    
                    # Add summary statistics
                    st.markdown("### Analysis Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total = len(display_df)
                        st.metric("Total Analyses", total)
                    
                    with col2:
                        fake_count = len(display_df[display_df["Result"] == "🚫 Fake"])
                        st.metric("Fake Media Detected", fake_count, f"{fake_count/total*100:.1f}%" if total > 0 else "0%")
                        
                    with col3:
                        real_count = len(display_df[display_df["Result"] == "✅ Real"])
                        st.metric("Real Media Verified", real_count, f"{real_count/total*100:.1f}%" if total > 0 else "0%")
                    
                    with col4:
                        media_counts = display_df["Type"].value_counts()
                        most_common = media_counts.idxmax().capitalize() if not media_counts.empty else "None"
                        st.metric("Most Analyzed", most_common, f"{media_counts.max()} files" if not media_counts.empty else "0 files")
                else:
                    display_empty_state()
            
            # Image tab
            with tab_image:
                image_df = display_df[display_df["Type"] == "image"]
                if not image_df.empty:
                    st.dataframe(
                        image_df.style.applymap(highlight_fake, subset=["Result"]),
                        use_container_width=True
                    )
                else:
                    st.info("No image analysis in history")
            
            # Audio tab
            with tab_audio:
                audio_df = display_df[display_df["Type"] == "audio"]
                if not audio_df.empty:
                    st.dataframe(
                        audio_df.style.applymap(highlight_fake, subset=["Result"]),
                        use_container_width=True
                    )
                else:
                    st.info("No audio analysis in history")
            
            # Video tab
            with tab_video:
                video_df = display_df[display_df["Type"] == "video"]
                if not video_df.empty:
                    st.dataframe(
                        video_df.style.applymap(highlight_fake, subset=["Result"]),
                        use_container_width=True
                    )
                else:
                    st.info("No video analysis in history")
            
            # Action buttons
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Start New Analysis", use_container_width=True):
                    st.switch_page("pages/detector.py")
            
            with col2:
                if st.button("🗑️ Clear History", use_container_width=True):
                    if st.session_state.get("confirm_delete", False):
                        try:
                            # Delete all records for this user
                            supabase.table("analysis_results").delete().eq("user_id", user.id).execute()
                            st.success("History cleared successfully")
                            st.session_state.pop("confirm_delete", None)
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error clearing history: {str(e)}")
                    else:
                        st.session_state.confirm_delete = True
                        st.warning("⚠️ Are you sure you want to clear your entire analysis history? This action cannot be undone. Click 'Clear History' again to confirm.")
            
        else:
            display_empty_state()
                
    except Exception as e:
        st.error(f"Error retrieving history: {str(e)}")
        
        # Check if the error is related to missing table
        if "does not exist" in str(e):
            display_setup_guide()

# Run the sidebar and page
if __name__ == "__main__":
    sidebar()
    history_page()