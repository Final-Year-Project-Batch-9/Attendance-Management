import streamlit as st
import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import json
import os
from datetime import datetime

def initialize_firebase(database_url):
    """Initialize Firebase with the provided credentials"""
    if not firebase_admin._apps:
        cred = credentials.Certificate('facerecst-firebase-adminsdk-fbsvc-e64d05e9e8.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': database_url
        })

def upload_to_firebase(df, root_node):
    """Upload dataframe to Firebase"""
    try:
        data_dict = df.to_dict('records')
        ref = db.reference(root_node)
        ref.set(data_dict)
        return True, f"Successfully uploaded {len(data_dict)} records to Firebase"
    except Exception as e:
        return False, f"Error uploading to Firebase: {str(e)}"

# Set page config
st.set_page_config(page_title="Firebase CSV Uploader", page_icon="🔥")

# Main app title
st.title("📤 Firebase CSV Uploader")

# Sidebar for Firebase configuration
st.sidebar.header("Firebase Configuration")

# Database URL input
database_url = "https://facerecst-default-rtdb.firebaseio.com/"

# Firebase credentials file upload
# credentials_file = ""

# Main content
st.write("Upload your CSV file and configure where to store it in Firebase.")

# CSV file upload
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

# Root node input
root_node = st.text_input(
    "Firebase Root Node",
    placeholder="Enter the path where data should be stored (e.g., 'data/csv_uploads')",
    help="This is the location in your Firebase Realtime Database where the data will be stored."
)

if uploaded_file and database_url and root_node:
    # Read CSV file
    try:
        df = pd.read_csv(uploaded_file)
        
        # Show data preview
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Show data info
        st.subheader("Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Size (KB)", round(uploaded_file.size/1024, 2))
        
        # Upload button
        if st.button("Upload to Firebase", type="primary"):
            with st.spinner("Uploading to Firebase..."):
                # Initialize Firebase
                # credentials_json = credentials_file.getvalue().decode('utf-8')
                initialize_firebase(database_url)
                
                # Perform upload
                success, message = upload_to_firebase(df, root_node)
                
                if success:
                    st.success(message)
                    # Add timestamp
                    st.write(f"Uploaded at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    st.error(message)
                    
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
else:
    st.info("Please provide all required inputs to proceed with the upload.")
    
    # Help section
    with st.expander("Need Help?"):
        st.markdown("""
        ### How to use this app:
        1. **Firebase Configuration** (Sidebar):
            - Enter your Firebase Database URL
            - Upload your Firebase credentials JSON file
        
        2. **Upload Data**:
            - Choose your CSV file
            - Specify the root node (database path) where data should be stored
        
        3. **Review & Upload**:
            - Preview your data
            - Click 'Upload to Firebase' when ready
        
        ### Getting Firebase Credentials:
        1. Go to Firebase Console
        2. Select your project
        3. Go to Project Settings > Service Accounts
        4. Generate new private key (JSON file)
        """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Firebase")