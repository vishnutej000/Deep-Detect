import os
import sys
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client
import time
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.auth import init_auth

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Database Setup",
    page_icon="🔧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Apply custom CSS
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
    h1 {
        color: #2E86C1;
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
    
    /* Progress indicator */
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #2E86C1, #3498DB);
    }
    
    .setup-step {
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3498DB;
        background-color: rgba(52, 152, 219, 0.05);
    }
    
    .setup-step h3 {
        margin-top: 0;
        color: #2E86C1;
    }
</style>
""", unsafe_allow_html=True)

def init_database():
    """Initialize the database tables needed for the application"""
    try:
        # Initialize authentication to set up supabase client in session state
        init_auth()
        supabase = st.session_state.supabase
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Check if analysis_results table exists
        status_text.text("Step 1/3: Checking if analysis_results table exists...")
        progress_bar.progress(10)
        
        table_exists = False
        try:
            # Try to query the table
            supabase.table("analysis_results").select("count").limit(1).execute()
            table_exists = True
            
            status_text.text("✅ analysis_results table already exists")
            progress_bar.progress(30)
            time.sleep(1)
            
        except Exception as e:
            if "does not exist" in str(e):
                status_text.text("Table doesn't exist, creating it now...")
                progress_bar.progress(20)
                time.sleep(1)
            else:
                st.error(f"Error checking table: {str(e)}")
                return False
        
        # Step 2: Create the table if it doesn't exist
        if not table_exists:
            status_text.text("Step 2/3: Creating analysis_results table...")
            progress_bar.progress(40)
            
            try:
                # Use raw SQL to create the table with proper schema
                sql = """
                CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
                
                CREATE TABLE IF NOT EXISTS public.analysis_results (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id UUID NOT NULL,
                    media_type TEXT NOT NULL CHECK (media_type IN ('image', 'audio', 'video')),
                    result JSONB NOT NULL,
                    file_path TEXT,
                    file_name TEXT NOT NULL,
                    file_size BIGINT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """
                
                # Execute the SQL using rpc
                try:
                    response = supabase.rpc('exec_sql', {'query': sql}).execute()
                    progress_bar.progress(60)
                except Exception as e:
                    # If rpc method fails, try direct query (depends on Supabase setup)
                    if hasattr(supabase, 'query') and callable(getattr(supabase, 'query')):
                        supabase.query(sql).execute()
                    else:
                        # If direct query isn't available, this will fail and show error
                        raise e
                
                status_text.text("Table created successfully!")
                progress_bar.progress(70)
                time.sleep(1)
                
            except Exception as e:
                st.error(f"Error creating table: {str(e)}")
                return False
        
        # Step 3: Set up permissions
        status_text.text("Step 3/3: Setting up permissions...")
        progress_bar.progress(80)
        
        try:
            # Set up RLS policies
            policies_sql = """
            -- Add permissions for authenticated users
            ALTER TABLE public.analysis_results ENABLE ROW LEVEL SECURITY;
            
            -- Drop existing policies if they exist
            DROP POLICY IF EXISTS "Users can only view their own results" ON public.analysis_results;
            DROP POLICY IF EXISTS "Users can only insert their own results" ON public.analysis_results;
            DROP POLICY IF EXISTS "Users can only update their own results" ON public.analysis_results;
            DROP POLICY IF EXISTS "Users can only delete their own results" ON public.analysis_results;
            
            -- Create a policy that allows users to only see their own results
            CREATE POLICY "Users can only view their own results" 
            ON public.analysis_results 
            FOR SELECT USING (auth.uid() = user_id);
            
            -- Create a policy that allows users to only insert their own results
            CREATE POLICY "Users can only insert their own results" 
            ON public.analysis_results 
            FOR INSERT WITH CHECK (auth.uid() = user_id);
            
            -- Create a policy that allows users to only update their own results
            CREATE POLICY "Users can only update their own results" 
            ON public.analysis_results 
            FOR UPDATE USING (auth.uid() = user_id);
            
            -- Create a policy that allows users to only delete their own results
            CREATE POLICY "Users can only delete their own results" 
            ON public.analysis_results 
            FOR DELETE USING (auth.uid() = user_id);
            """
            
            # Execute the permissions SQL
            try:
                response = supabase.rpc('exec_sql', {'query': policies_sql}).execute()
            except Exception as e:
                # If rpc method fails, try direct query (depends on Supabase setup)
                if hasattr(supabase, 'query') and callable(getattr(supabase, 'query')):
                    supabase.query(policies_sql).execute()
                else:
                    # Just log the error but continue - not all Supabase setups allow this
                    st.warning(f"Note: Could not set up row-level security. You may need to set this up manually in the Supabase dashboard.")
            
            progress_bar.progress(100)
            status_text.text("✅ Database setup complete!")
            time.sleep(1)
            
            return True
            
        except Exception as e:
            st.warning(f"Note: Could not set up permissions. This is not critical but you may want to set them up manually: {str(e)}")
            progress_bar.progress(100)
            status_text.text("✅ Database setup mostly complete (without permissions)")
            return True
            
    except Exception as e:
        st.error(f"Error initializing database: {str(e)}")
        return False

# Create a simple UI for initializing the database
def main():
    st.title("Database Setup Utility")
    st.markdown("""
    This utility will create the necessary database tables for the "Is This Fake?" application. If you're seeing 
    errors about missing tables in the history page, this will fix the issue.
    """)
    
    st.markdown("""
    <div class="setup-step">
        <h3>What this will do:</h3>
        <ul>
            <li>Create the <code>analysis_results</code> table to store your detection history</li>
            <li>Set up security permissions to protect your data</li>
            <li>Allow the app to save and display your detection results</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if user is authenticated
    if 'supabase' not in st.session_state:
        st.warning("Please log in first before running the database setup.")
        if st.button("Go to Login"):
            st.switch_page("pages/login.py")
        return
    
    if st.button("Initialize Database", key="init_db_button", use_container_width=True):
        with st.spinner("Setting up database..."):
            success = init_database()
            if success:
                st.success("✅ Database setup complete! You can now use the application.")
                st.markdown("""
                <div class="setup-step">
                    <h3>What's next?</h3>
                    <p>You can now:</p>
                    <ul>
                        <li>Return to the history page to view your analysis results</li>
                        <li>Use the detector to analyze images, audio, and video</li>
                        <li>Check your profile to see usage statistics</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Go to History", use_container_width=True):
                        st.switch_page("pages/history.py")
                with col2:
                    if st.button("Go to Detector", use_container_width=True):
                        st.switch_page("pages/detector.py")
                        
            else:
                st.error("Database setup failed. Please check the errors above.")

if __name__ == "__main__":
    main()