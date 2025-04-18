import streamlit as st
import cv2
import os
from datetime import datetime
import pandas as pd
from my_utils import alignment_procedure
from mtcnn import MTCNN
import glob
import ArcFace
import numpy as np
import keras
from keras import layers, Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import hashlib
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import random
import string

# Initialize session state variables
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False

# Firebase configuration
database_url = "https://facerecst-default-rtdb.firebaseio.com/"
credential_path = 'facerecst-firebase-adminsdk-fbsvc-e64d05e9e8.json'

def initialize_firebase(db_url=database_url, cred_path=credential_path):
    """Initialize Firebase with the provided credentials"""
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(credential_path)
            firebase_admin.initialize_app(cred, {'databaseURL': database_url})
    except Exception as e:
        st.error(f"Firebase initialization failed: {str(e)}")

def upload_to_firebase(df, root_node):
    """Upload dataframe to Firebase"""
    try:
        initialize_firebase()  # Ensure initialization before operation
        data_dict = df.to_dict('records')
        ref = db.reference(root_node)
        ref.set(data_dict)
        return True, f"Successfully uploaded {len(data_dict)} records to Firebase"
    except Exception as e:
        return False, f"Error uploading to Firebase: {str(e)}"

# Authentication functions
def initialize_users_db():
    if os.path.exists('users.csv'):
        return pd.read_csv('users.csv')
    return pd.DataFrame(columns=['username', 'password', 'role', 'email'])

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_credentials(username, password):
    users_df = initialize_users_db()
    hashed_pw = hash_password(password)
    user_row = users_df[users_df['username'] == username]
    if not user_row.empty and user_row.iloc[0]['password'] == hashed_pw:
        return True, user_row.iloc[0]['role']
    return False, None

def signup_user(username, password, role, email):
    users_df = initialize_users_db()
    if username in users_df['username'].values:
        return False, "Username already exists"
    
    new_user = {
        'username': username,
        'password': hash_password(password),
        'role': role,
        'email': email
    }
    users_df = pd.concat([users_df, pd.DataFrame([new_user])], ignore_index=True)
    users_df.to_csv('users.csv', index=False)
    
    success, message = upload_to_firebase(users_df, "data/users")
    if success:
        st.success(message)
    else:
        st.error(message)
    return True, "User registered successfully"

# Attendance functions
def initialize_attendance_df():
    if os.path.exists('attendance.csv'):
        return pd.read_csv('attendance.csv')
    return pd.DataFrame(columns=['Date', 'Subject', 'Enrollment_Number', 'Student_Name', 'Time', 'Status'])

def save_attendance(subject, enrollment_number, student_name):
    df = initialize_attendance_df()
    current_time = datetime.now()
    
    today = current_time.strftime('%Y-%m-%d')
    existing_record = df[
        (df['Date'] == today) & 
        (df['Subject'] == subject) & 
        (df['Enrollment_Number'] == enrollment_number)
    ]
    
    if existing_record.empty:
        new_record = {
            'Date': today,
            'Subject': subject,
            'Enrollment_Number': enrollment_number,
            'Student_Name': student_name,
            'Time': current_time.strftime('%H:%M:%S'),
            'Status': 'Present'
        }
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        df.to_csv('attendance.csv', index=False)
        
        success, message = upload_to_firebase(df, "data/attendance")
        if success:
            st.success(message)
        else:
            st.error(message)
        return True
    return False


# # Get absentees
def get_absentees(subject, date):
    if not os.path.exists('students_master.csv'):
        st.error("Students master list not found!")
        return None
    
    master_df = pd.read_csv('students_master.csv')
    if not os.path.exists('attendance.csv'):
        return master_df  # All students are absent
        
    attendance_df = pd.read_csv('attendance.csv')
    
    # Filter attendance for given subject and date
    present_students = attendance_df[
        (attendance_df['Subject'] == subject) & 
        (attendance_df['Date'] == date)
    ]['Enrollment_Number'].unique()
    
    # Get absentees
    absentees = master_df[~master_df['Enrollment_Number'].isin(present_students)]
    return absentees, master_df
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import time

def send_absence_email(student_email, student_name, subject_name, date):
    """Send absence notification email to student"""
    try:
        # Email configuration
        sender_email = "attendancemanagementarcface@gmail.com"
        sender_password = "agscnsdnnvhpkzik" 
        # sender_email = st.secrets["email"]["sender_email"]
        # sender_password = st.secrets["email"]["sender_password"]
        report_date = st.date_input('Select Date', 
                                value=st.session_state.report_date if st.session_state.report_date else None)
        st.session_state.report_date = report_date
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = student_email
        msg['Subject'] = f'Absence Notification - {subject_name}'
        
        # Email body
        body = f"""Dear {student_name},

This is to inform you that you were marked absent for {subject_name} on {report_date.strftime('%Y-%m-%d')}.

Please ensure regular attendance in classes.

Best regards,
College Administration"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect to SMTP server (Gmail example)
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        

        # Send email
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email to {student_name}: {str(e)}")
        return False
    
# # 4. Authentication UI
def show_login_page():
    """Display login form"""
    st.title("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login", key="login_button"):
        is_valid, role = check_credentials(username, password)
        if is_valid:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.session_state['role'] = role
            st.success(f"Logged in successfully as {role}")
            st.rerun()
        else:
            st.error("Invalid username or password")

def show_signup_page():
    """Display signup form"""
    st.title("Sign Up")
    username = st.text_input("Username", key="signup_username")
    password = st.text_input("Password", type="password", key="signup_password")
    email = st.text_input("Email", key="signup_email")
    role = st.selectbox("Role", ["student", "teacher"], key="signup_role")
    
    if st.button("Sign Up", key="signup_button"):
        if role == "teacher":
            st.warning("Username and Password will be Provided by Administration")
        else:
            success, message = signup_user(username, password, role, email)
            if success:
                st.success(message)
            else:
                st.error(message)
####################################################################################################
import random
import string

def generate_reset_token():
    """Generate a random reset token"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=32))

def send_reset_email(email, reset_token):
    """Send password reset email"""
    try:
        sender_email = "attendancemanagementarcface@gmail.com"
        sender_password = "agscnsdnnvhpkzik"
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = email
        msg['Subject'] = 'Password Reset Request'
        
        # Create reset link (you would need to set up proper URL)
        reset_link = f"your-app-url/reset-password?token={reset_token}"
        
        body = f"""
        Hello,

        A password reset was requested for your account.
        Your reset token is: {reset_token}

        Please enter this token in the application to reset your password.

        If you did not request this reset, please ignore this email.

        Best regards,
        Face Recognition Attendance System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True, "Reset email sent successfully"
    except Exception as e:
        return False, str(e)

def save_reset_token(email, token):
    """Save reset token to CSV"""
    if not os.path.exists('reset_tokens.csv'):
        reset_df = pd.DataFrame(columns=['email', 'token', 'timestamp'])
    else:
        reset_df = pd.read_csv('reset_tokens.csv')
    
    # Remove any existing tokens for this email
    reset_df = reset_df[reset_df['email'] != email]
    
    # Add new token
    new_token = {
        'email': email,
        'token': token,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    reset_df = pd.concat([reset_df, pd.DataFrame([new_token])], ignore_index=True)
    reset_df.to_csv('reset_tokens.csv', index=False)
    return True

def verify_reset_token(email, token):
    """Verify if reset token is valid"""
    if not os.path.exists('reset_tokens.csv'):
        return False
    
    reset_df = pd.read_csv('reset_tokens.csv')
    token_row = reset_df[
        (reset_df['email'] == email) & 
        (reset_df['token'] == token)
    ]
    
    if token_row.empty:
        return False
    
    # Check if token is not expired (24 hours validity)
    token_time = datetime.strptime(token_row.iloc[0]['timestamp'], '%Y-%m-%d %H:%M:%S')
    if (datetime.now() - token_time).total_seconds() > 24 * 3600:
        return False
    
    return True

def update_password(email, new_password):
    """Update user's password"""
    users_df = pd.read_csv('users.csv')
    users_df.loc[users_df['email'] == email, 'password'] = hash_password(new_password)
    users_df.to_csv('users.csv', index=False)
    
    # Update Firebase
    df = users_df
    root_node = "data/users"
    success, message = upload_to_firebase(df, root_node)
    initialize_firebase()
    return success, message

def show_forgot_password_page():
    """Display forgot password form"""
    st.markdown("""
        <style>
        .forgot-password-container {
            background-color: ;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .reset-status {
            margin-top: 1rem;
            padding: 1rem;
            border-radius: 5px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("# 🔑 Reset Password")
    
    # Initialize session state for reset flow
    if 'reset_step' not in st.session_state:
        st.session_state['reset_step'] = 'email'
    
    st.markdown("<div class='forgot-password-container'>", unsafe_allow_html=True)
    
    if st.session_state['reset_step'] == 'email':
        email = st.text_input("📧 Enter your email address")
        
        if st.button("Send Reset Link", use_container_width=True):
            if not email:
                st.error("Please enter your email address")
                return
            
            # Verify email exists in users database
            users_df = pd.read_csv('users.csv')
            if email not in users_df['email'].values:
                st.error("Email address not found")
                return
            
            # Generate and save reset token
            reset_token = generate_reset_token()
            if save_reset_token(email, reset_token):
                # Send reset email
                success, message = send_reset_email(email, reset_token)
                if success:
                    st.success("Reset instructions sent to your email")
                    st.session_state['reset_email'] = email
                    st.session_state['reset_step'] = 'verify'
                    st.rerun()
                else:
                    st.error(f"Failed to send reset email: {message}")
    
    elif st.session_state['reset_step'] == 'verify':
        st.info(f"📧 Reset token sent to {st.session_state['reset_email']}")
        token = st.text_input("🔑 Enter Reset Token")
        
        if st.button("Verify Token", use_container_width=True):
            if verify_reset_token(st.session_state['reset_email'], token):
                st.session_state['reset_step'] = 'reset'
                st.rerun()
            else:
                st.error("Invalid or expired token")
    
    elif st.session_state['reset_step'] == 'reset':
        new_password = st.text_input("🔒 Enter New Password", type="password")
        confirm_password = st.text_input("🔒 Confirm New Password", type="password")
        
        if st.button("Reset Password", use_container_width=True):
            if not new_password or not confirm_password:
                st.error("Please fill in all fields")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                success, message = update_password(st.session_state['reset_email'], new_password)
                if success:
                    st.success("Password reset successfully! Please login with your new password.")
                    # Clear reset session state
                    del st.session_state['reset_step']
                    del st.session_state['reset_email']
                    st.rerun()
                else:
                    st.error(f"Failed to reset password: {message}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Option to go back to login
    if st.button("← Back to Login"):
        if 'reset_step' in st.session_state:
            del st.session_state['reset_step']
        if 'reset_email' in st.session_state:
            del st.session_state['reset_email']
        st.rerun()
####################################################################################################

def update_student_emails():
    """Utility to update email addresses for existing students"""
    st.title("Update Student Email Addresses")
    
    if not os.path.exists('students_master.csv'):
        st.error("Students master list not found!")
        return
    
    master_df = pd.read_csv('students_master.csv')
    
    # Add Email column if it doesn't exist
    if 'Email' not in master_df.columns:
        master_df['Email'] = ""
    
    st.subheader("Current Student Records")
    
    # Display current data with email column
    st.dataframe(master_df)
    
    # Form to update a specific student's email
    with st.form("update_email_form"):
        enrollment = st.text_input("Enrollment Number")
        new_email = st.text_input("Email Address")
        update_button = st.form_submit_button("Update Email")
    
    if update_button:
        if not enrollment or not new_email:
            st.error("Please provide both enrollment number and email address")
        else:
            # Find student by enrollment number
            student_idx = master_df[master_df['Enrollment_Number'] == enrollment].index
            
            if len(student_idx) == 0:
                st.error(f"No student found with enrollment number {enrollment}")
            else:
                # Update email
                master_df.loc[student_idx, 'Email'] = new_email
                master_df.to_csv('students_master.csv', index=False)
                
                # Update Firebase
                success, message = upload_to_firebase(master_df, "data/students_master")
                initialize_firebase()
                
                if success:
                    st.success(f"Email updated successfully for {enrollment}")
                else:
                    st.error(f"Failed to update Firebase: {message}")
                
                # Refresh the display
                st.dataframe(master_df)


# # Main Application
def main():
        # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    # Show authentication pages if not logged in
    if not st.session_state['logged_in']:
        tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Forgot Password"])
        with tab1:
            show_login_page()
        with tab2:
            show_signup_page()
        with tab3:
            show_forgot_password_page()
        return

    st.write(f"# :blue[Face Recognition Attendance System]")
    # st.title('Face Recognition Attendance System')
    os.makedirs('data', exist_ok=True)
    name_list = os.listdir('data')

    # Sidebar Navigation
    st.sidebar.write(f"# :blue[Face Recognition Attendance System]")
    st.sidebar.title('Navigation')
    # Sidebar Navigation based on role
    if st.session_state['role'] == 'teacher':
        app_mode = st.sidebar.selectbox('Choose Mode',
            ['Data Collection', 'Normalize Data', 'Train Model', 'Take Attendance', 'View Reports', 'Update Student Emails'])
    else:  # student role
        app_mode = 'View Reports'
        st.sidebar.text("Mode: Take Attendance")
    
    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['role'] = None
        st.rerun()
    # app_mode = st.sidebar.selectbox('Choose Mode',
    #     ['Data Collection', 'Normalize Data', 'Train Model', 'Take Attendance', 'View Reports'])

    # Common Settings
    webcam_channel = st.sidebar.selectbox(
        'Webcam Channel:',
        ('Select Channel', '0', '1', '2', '3')
    )
    if app_mode == 'Data Collection':
        # Page configuration
        # st.set_page_config(layout="wide")
        
        # Custom CSS for better styling
        st.markdown("""
            <style>
            .main {
                padding: 2rem;
            }
            .stButton>button {
                width: 100%;
                margin-top: 1rem;
            }
            .registration-container {
                background-color: ;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            </style>
        """, unsafe_allow_html=True)

        # Header with icon
        st.markdown("# 📸 Student Registration System")
        st.markdown("---")

        # Create two columns for better layout
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
                <div class='registration-container'>
                <h3>Student Information</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Form inputs with better organization
            with st.form(key='registration_form'):
                enrollment_number = st.text_input('📝 Enrollment Number:', 
                                            placeholder="Enter enrollment number")
                name_person = st.text_input('👤 Student Name:', 
                                        placeholder="Enter student's full name")
                student_email = st.text_input('📧 Student Email:', 
                                        placeholder="Enter student's email address")
                subject_name = st.text_input('📚 Subject Name:', 
                                        placeholder="Enter subject name")
                img_number = st.number_input('📸 Number of Images:', 
                                        min_value=1, max_value=100, value=50,
                                        help="Number of images to capture for face recognition")
                
                # Submit button for the form
                submit_button = st.form_submit_button(label='Register Student',
                                                    use_container_width=True)

        with col2:
            # Camera preview section
            st.markdown("""
                <div class='registration-container'>
                <h3>Camera Preview</h3>
                </div>
            """, unsafe_allow_html=True)
            
            FRAME_WINDOW = st.image([], use_container_width=True)
            
            # Display camera status
            if not webcam_channel == 'Select Channel':
                st.success('📸 Camera is ready')
            else:
                st.warning('⚠️ Please select a camera channel')

            # Progress indicator
            if 'capture_progress' not in st.session_state:
                st.session_state.capture_progress = 0
                
        # Registration logic
        if not webcam_channel == 'Select Channel' and submit_button:
            if not enrollment_number or not name_person:
                st.error('❌ Please fill enrollment number and student name!')
            else:
                folder_name = f"{enrollment_number}_{name_person}"
                os.makedirs(f'data/{folder_name}', exist_ok=True)
                
                # Save to master list
                if not os.path.exists('students_master.csv'):
                    master_df = pd.DataFrame(columns=['Enrollment_Number', 'Student_Name', 'Email'])
                else:
                    master_df = pd.read_csv('students_master.csv')
                    # Add Email column if it doesn't exist
                    if 'Email' not in master_df.columns:
                        master_df['Email'] = ""
                
                if not master_df[master_df['Enrollment_Number'] == enrollment_number].empty:
                    st.error('❌ This enrollment number already exists!')
                else:
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    new_student = {
                        'Enrollment_Number': enrollment_number,
                        'Student_Name': name_person,
                        'Email': student_email  # Add email to the new student entry
                    }
                    master_df = pd.concat([master_df, pd.DataFrame([new_student])], ignore_index=True)
                    master_df = master_df.where(pd.notnull(master_df), None)
                    master_df.to_csv('students_master.csv', index=False)
                    df = master_df
                    root_node = "data/students_master"
                    success, message = upload_to_firebase(df, root_node)
                    initialize_firebase(database_url)
                    
                    if success:
                        st.success(f'✅ {message}')
                    else:
                        st.error(f'❌ {message}')
                    
                    # Capture Images with improved visual feedback
                    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                    cap = cv2.VideoCapture(int(webcam_channel))
                    count = 0
                    
                    try:
                        while True:
                            success, img = cap.read()
                            if not success:
                                st.error('❌ Camera not working!')
                                break

                            faces = face_classifier.detectMultiScale(img)
                            if len(faces) > 0:
                                # Save Image only when face is detected
                                cv2.imwrite(f'data/{folder_name}/{count}.jpg', img)
                                
                                # Update progress
                                progress = int((count + 1) / img_number * 100)
                                progress_bar.progress(progress)
                                status_text.text(f'📸 Capturing image {count + 1}/{img_number}')
                                count += 1

                                # Draw face rectangle with better visuals
                                for (x, y, w, h) in faces:
                                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                    cv2.putText(img, 'Face Detected', (x, y-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                            (0, 255, 0), 2)

                            FRAME_WINDOW.image(img, channels='BGR')
                            if count >= img_number:
                                st.success(f'✅ Registration completed for {name_person}')
                                # st.balloons()  # Add a celebratory effect
                                break

                    finally:
                        FRAME_WINDOW.image([])
                        cap.release()
                        cv2.destroyAllWindows()
                        
        # Add helpful instructions
        with st.expander("ℹ️ How to use"):
            st.markdown("""
                1. Fill in the student's information in the form
                2. Make sure the camera is properly selected
                3. Click 'Register Student' to begin the registration process
                4. Keep the student's face centered in the camera view
                5. Wait for all images to be captured
                6. Registration will complete automatically
            """)
    elif app_mode == 'Update Student Emails':
        update_student_emails()
    elif app_mode == 'Normalize Data':
        # Page styling
        st.markdown("""
            <style>
            .normalization-container {
                background-color: #3D3C3A;
                padding: 1rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
            }
            .stats-box {
                background-color:;
                padding: 1rem;
                border-radius: 5px;
                border-left: 4px solid #007bff;
                margin: 0.5rem 0;
            }
            </style>
        """, unsafe_allow_html=True)

        # Header section with description
        st.markdown("# 🔄 Image Data Normalization")
        st.markdown("""
            <div class='normalization-container'>
            <p>This process will normalize all collected face images to ensure consistent quality for training:
            <ul>
                <li>Detect facial landmarks</li>
                <li>Align faces based on eye positions</li>
                <li>Standardize image dimensions</li>
            </ul>
            </p>
            </div>
        """, unsafe_allow_html=True)

        # Create columns for better organization
        col1, col2 = st.columns([2, 1])

        with col1:
            # Main normalization section
            st.markdown("<div class='normalization-container'>", unsafe_allow_html=True)
            
            # Check if data exists
            if not os.path.exists("data") or len(os.listdir("data")) == 0:
                st.warning("⚠️ No data found to normalize. Please collect data first.")
                normalize_button = st.button('Start Normalization', disabled=True)
            else:
                # Display data statistics
                total_students = len(os.listdir("data"))
                total_images = sum(len(files) for _, _, files in os.walk("data"))
                
                st.markdown("""
                    <div class='stats-box'>
                    <h4>📊 Current Data Statistics</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    st.metric("Total Students", total_students)
                with stats_col2:
                    st.metric("Total Images", total_images)
                
                normalize_button = st.button('▶️ Start Normalization', use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # Progress and status section
            st.markdown("<div class='normalization-container'>", unsafe_allow_html=True)
            st.markdown("### 📋 Process Status")
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)

        if normalize_button:
            path_to_dir = "data"
            path_to_save = 'norm_data'
            os.makedirs(path_to_save, exist_ok=True)
            
            detector = MTCNN()
            class_list = os.listdir(path_to_dir)
            total_classes = len(class_list)
            
            # Initialize counters
            processed_images = 0
            failed_images = 0
            total_images = sum(len(glob.glob(os.path.join(path_to_dir, name) + '/*')) for name in class_list)
            
            # Create progress bar
            progress_bar = progress_placeholder.progress(0)
            status_text = status_placeholder.empty()
            
            for idx, name in enumerate(class_list):
                status_text.info(f"💫 Processing student: {name} ({idx + 1}/{total_classes})")
                img_list = glob.glob(os.path.join(path_to_dir, name) + '/*')
                save_folder = os.path.join(path_to_save, name)
                os.makedirs(save_folder, exist_ok=True)

                for img_path in img_list:
                    try:
                        img = cv2.imread(img_path)
                        detections = detector.detect_faces(img)

                        if len(detections) > 0:
                            right_eye = detections[0]['keypoints']['right_eye']
                            left_eye = detections[0]['keypoints']['left_eye']
                            bbox = detections[0]['box']
                            norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)
                            cv2.imwrite(f'{save_folder}/{os.path.split(img_path)[1]}', norm_img_roi)
                            processed_images += 1
                        else:
                            st.warning(f'⚠️ No face detected in {os.path.basename(img_path)}')
                            failed_images += 1
                    except Exception as e:
                        st.error(f'❌ Error processing {os.path.basename(img_path)}: {str(e)}')
                        st.write(img_path)
                        failed_images += 1
                    
                    # Update progress
                    progress = (processed_images + failed_images) / total_images
                    progress_bar.progress(progress)
                    
                    # Update status
                    status_text.info(f"""
                        📊 Progress: {processed_images + failed_images}/{total_images} images
                        ✅ Successfully processed: {processed_images}
                        ⚠️ Failed: {failed_images}
                    """)
            
            # Final status update
            if processed_images > 0:
                st.success(f"""
                    ✨ Normalization completed!
                    - Total images processed: {processed_images + failed_images}
                    - Successfully normalized: {processed_images}
                    - Failed: {failed_images}
                """)
                
                if failed_images == 0:
                    st.balloons()
            else:
                st.error("❌ No images were successfully normalized. Please check the input data.")

        # Add help section
        with st.expander("ℹ️ About Normalization"):
            st.markdown("""
                The normalization process includes:
                1. **Face Detection**: Using MTCNN to detect faces in images
                2. **Landmark Detection**: Identifying key facial points
                3. **Alignment**: Adjusting face orientation based on eye positions
                4. **Standardization**: Ensuring consistent image size and quality
                
                This step is crucial for improving the accuracy of face recognition.
            """)
    elif app_mode == 'Train Model':
        # Custom CSS for styling
        st.markdown("""
            <style>
            .training-container {
                background-color: ;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
            }
            .metric-card {
                background-color: ;
                padding: 1rem;
                border-radius: 5px;
                border-left: 4px solid #28a745;
                margin: 0.5rem 0;
            }
            </style>
        """, unsafe_allow_html=True)

        # Header section
        st.markdown("# 🧠 Train Recognition Model")
        
        # Create main columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<div class='training-container'>", unsafe_allow_html=True)
            
            # Check if normalized data exists
            if not os.path.exists("norm_data") or len(os.listdir("norm_data")) == 0:
                st.warning("⚠️ No normalized data found. Please normalize your data first.")
                start_button = st.button('Start Training', disabled=True)
            else:
                # Display dataset statistics
                total_students = len(os.listdir("norm_data"))
                total_images = sum(len(files) for _, _, files in os.walk("norm_data"))
                
                st.markdown("### 📊 Dataset Overview")
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    st.metric("Number of Students", total_students)
                with stats_col2:
                    st.metric("Total Images", total_images)
                
                # Training parameters
                st.markdown("### ⚙️ Training Parameters")
                with st.expander("Advanced Settings"):
                    epochs = st.slider("Number of Epochs", 50, 200, 100)
                    batch_size = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=32)
                    patience = st.slider("Early Stopping Patience", 5, 30, 20)
                
                start_button = st.button('▶️ Start Training', use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            # Progress and metrics section
            st.markdown("<div class='training-container'>", unsafe_allow_html=True)
            st.markdown("### 📈 Training Progress")
            metrics_placeholder = st.empty()
            progress_placeholder = st.empty()
            chart_placeholder = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)

        if start_button:
                with st.spinner('Loading ArcFace Model...'):
                    # Load ArcFace Model
                    model = ArcFace.loadModel()
                    target_size = model.layers[0].input_shape[0][1:3]
                    st.success('✅ ArcFace model loaded successfully')

                # Prepare data
                x = []
                y = []
                names = sorted(os.listdir('norm_data'))
                total_images = sum(len(glob.glob(os.path.join('norm_data', name) + '/*')) for name in names)
                
                # Create progress bar for data processing
                progress_bar = progress_placeholder.progress(0)
                status_text = metrics_placeholder.empty()
                processed_images = 0
                
                for name in names:
                    status_text.info(f'💫 Processing images for student: {name}')
                    img_list = glob.glob(os.path.join('norm_data', name) + '/*')
                    
                    for img_path in img_list:
                        img = cv2.imread(img_path)
                        img_resize = cv2.resize(img, target_size)
                        img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
                        img_pixels = np.expand_dims(img_pixels, axis=0)
                        img_norm = img_pixels/255
                        img_embedding = model.predict(img_norm, verbose=0)[0]
                        
                        x.append(img_embedding)
                        y.append(name)
                        
                        processed_images += 1
                        progress = processed_images / total_images
                        progress_bar.progress(progress)
                        status_text.info(f"""
                            📊 Data Processing Progress:
                            - Images processed: {processed_images}/{total_images}
                            - Current student: {name}
                        """)

                # Prepare training data
                status_text.info('🔄 Preparing training data...')
                df = pd.DataFrame(x, columns=np.arange(512))
                df['names'] = y
                
                x = df.drop('names', axis=1)
                y = pd.factorize(df['names'])[0]
                y = keras.utils.np_utils.to_categorical(y)

                # Split data
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
                
                status_text.success(f"""
                    ✅ Data preparation completed:
                    - Training samples: {len(x_train)}
                    - Testing samples: {len(x_test)}
                    - Number of classes: {len(names)}
                """)

                # Build model
                status_text.info('🏗️ Building model architecture...')
                model = Sequential([
                    layers.Dense(1024, activation='relu', input_shape=[512]),
                    layers.Dense(512, activation='relu'),
                    layers.Dense(len(names), activation="softmax")
                ])

                model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

                # Custom callback for Streamlit updates
                class StreamlitCallback(keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        # Update metrics
                        status_text.info(f"""
                            📊 Training Progress - Epoch {epoch + 1}/{epochs}
                            Training Accuracy: {logs['accuracy']:.4f}
                            Validation Accuracy: {logs['val_accuracy']:.4f}
                            Training Loss: {logs['loss']:.4f}
                            Validation Loss: {logs['val_loss']:.4f}
                        """)
                        
                        # Update progress bar
                        progress_bar.progress((epoch + 1) / epochs)

                # Train model
                status_text.info('🚀 Starting model training...')
                history = model.fit(x_train, y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(x_test, y_test),
                                callbacks=[
                                    StreamlitCallback(),
                                    keras.callbacks.ModelCheckpoint(
                                        'model.h5',
                                        monitor='val_accuracy',
                                        save_best_only=True,
                                        mode='max'),
                                    keras.callbacks.EarlyStopping(
                                        monitor='val_accuracy',
                                        patience=patience)
                                ])

                # Final evaluation
                final_train_acc = history.history['accuracy'][-1]
                final_val_acc = history.history['val_accuracy'][-1]
                
                st.success(f"""
                    ✨ Model training completed successfully!
                    Final Results:
                    - Training Accuracy: {final_train_acc:.4f}
                    - Validation Accuracy: {final_val_acc:.4f}
                    - Best model saved as 'model.h5'
                    
                    The model is now ready for attendance taking!
                """)


        # Add informational section
        with st.expander("ℹ️ About Model Training"):
            st.markdown("""
                This process involves:
                1. **Feature Extraction**: Using ArcFace to extract face embeddings
                2. **Data Preparation**: Splitting data into training and validation sets
                3. **Model Training**: Training a neural network for face recognition
                4. **Model Evaluation**: Monitoring accuracy and loss metrics
                
                The trained model will be used for student recognition during attendance taking.
            """)
    

    elif app_mode == 'Take Attendance':
        # Custom CSS for styling
        st.markdown("""
            <style>
            .attendance-container {
                background-color: ;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
            }
            .status-card {
                background-color: ;
                padding: 1rem;
                border-radius: 5px;
                border-left: 4px solid #17a2b8;
                margin: 0.5rem 0;
            }
            .present-list {
                max-height: 300px;
                overflow-y: auto;
                padding: 1rem;
                background-color: ;
                border-radius: 5px;
            }
            .present-item {
                padding: 0.5rem;
                margin: 0.25rem 0;
                border-radius: 3px;
                background-color: ;
            }
            </style>
        """, unsafe_allow_html=True)

        # Header section
        st.markdown("# 📸 Take Attendance")

        # Create main columns
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("<div class='attendance-container'>", unsafe_allow_html=True)
            
            # Check prerequisites
            model_exists = os.path.exists('model.h5')
            students_exist = os.path.exists('data') and len(os.listdir('data')) > 0
            
            if not model_exists:
                st.error("⚠️ Model not found! Please train the model first.")
            elif not students_exist:
                st.error("⚠️ No registered students found!")
            else:
                # Session setup
                if 'attendance_session' not in st.session_state:
                    st.session_state.attendance_session = {
                        'active': False,
                        'recognized_students': set(),
                        'start_time': None
                    }
                
                # Input section
                subject_name = st.text_input('📚 Subject Name:', placeholder="Enter subject name")
                
                # Advanced settings
                with st.expander("⚙️ Advanced Settings"):
                    confidence_threshold = st.slider(
                        'Recognition Confidence Threshold:',
                        min_value=0.0,
                        max_value=1.0,
                        value=0.6,
                        help="Higher values mean stricter recognition criteria"
                    )
                    
                    show_confidence = st.checkbox(
                        'Show Confidence Scores',
                        value=True,
                        help="Display confidence percentage on recognized faces"
                    )
                
                # Camera feed
                st.markdown("### 📹 Camera Feed")
                FRAME_WINDOW = st.image([], use_container_width=True)
                
                # Control buttons
                if not st.session_state.attendance_session['active']:
                    start_button = st.button(
                        '▶️ Start Attendance',
                        use_container_width=True,
                        disabled=not subject_name
                    )
                else:
                    stop_button = st.button(
                        '⏹️ Stop Attendance',
                        use_container_width=True
                    )
            
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # Status and metrics section
            st.markdown("<div class='attendance-container'>", unsafe_allow_html=True)
            st.markdown("### 📊 Session Status")
            status_placeholder = st.empty()
            recognition_placeholder = st.empty()
            
            # Present students list
            st.markdown("### ✅ Present Students")
            present_list_placeholder = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)

        if 'start_button' in locals() and start_button and subject_name:
            st.session_state.attendance_session['active'] = True
            st.session_state.attendance_session['start_time'] = time.time()
            st.session_state.attendance_session['recognized_students'] = set()
            
            try:
                with st.spinner('Loading recognition models...'):
                    # Load all required models
                    detector = MTCNN()
                    arcface_model = ArcFace.loadModel()
                    face_rec_model = load_model('model.h5')
                    st.success('✅ Models loaded successfully')
                
                # Get list of registered students and their folders
                registered_students = sorted(os.listdir('data'))
                
                # Create mapping of class index to student info
                class_to_student = {}
                for idx, student_folder in enumerate(registered_students):
                    try:
                        parts = student_folder.split('_', 1)
                        if len(parts) != 2:
                            continue
                        enrollment, name = parts
                        class_to_student[idx] = {
                            'enrollment': enrollment,
                            'name': name
                        }
                    except Exception as e:
                        continue
                
                cap = cv2.VideoCapture(int(webcam_channel))
                
                while st.session_state.attendance_session['active']:
                    success, img = cap.read()
                    if not success:
                        st.error('❌ Camera not working!')
                        break

                    # Update session duration
                    session_duration = time.time() - st.session_state.attendance_session['start_time']
                    status_placeholder.markdown(f"""
                        <div class='status-card'>
                        📊 Session Statistics:
                        - Duration: {int(session_duration // 60)}m {int(session_duration % 60)}s
                        - Students Present: {len(st.session_state.attendance_session['recognized_students'])}
                        </div>
                    """, unsafe_allow_html=True)

                    # Face detection
                    detections = detector.detect_faces(img)
                    
                    if len(detections) > 0:
                        for detection in detections:
                            try:
                                bbox = detection['box']
                                right_eye = detection['keypoints']['right_eye']
                                left_eye = detection['keypoints']['left_eye']
                                
                                # Process detected face
                                norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)
                                img_resize = cv2.resize(norm_img_roi, arcface_model.layers[0].input_shape[0][1:3])
                                img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
                                img_pixels = np.expand_dims(img_pixels, axis=0)
                                img_norm = img_pixels/255
                                
                                # Get face embedding and prediction
                                img_embedding = arcface_model.predict(img_norm)[0]
                                prediction = face_rec_model.predict(np.array([img_embedding]))[0]
                                
                                # Show prediction confidence
                                max_confidence = max(prediction)
                                recognition_placeholder.markdown(f"""
                                    <div class='status-card'>
                                    🎯 Recognition Status:
                                    - Confidence: {max_confidence:.1%}
                                    - Threshold: {confidence_threshold:.1%}
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                if max_confidence > confidence_threshold:
                                    class_idx = np.argmax(prediction)
                                    
                                    if class_idx in class_to_student:
                                        student = class_to_student[class_idx]
                                        enrollment = student['enrollment']
                                        name = student['name']
                                        
                                        # Mark attendance
                                        if enrollment not in st.session_state.attendance_session['recognized_students']:
                                            if save_attendance(subject_name, enrollment, name):
                                                st.session_state.attendance_session['recognized_students'].add(enrollment)
                                                st.toast(f"✅ Marked attendance for {name}")
                                        
                                        # Draw green rectangle
                                        cv2.rectangle(img, 
                                                (int(bbox[0]), int(bbox[1])), 
                                                (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), 
                                                (0, 255, 0), 2)
                                        
                                        # Display name and confidence if enabled
                                        label = f"{name}"
                                        if show_confidence:
                                            label += f" ({max_confidence:.1%})"
                                        
                                        cv2.putText(img, label,
                                                (int(bbox[0]), int(bbox[1] - 10)), 
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                                (0, 255, 0), 2)
                                else:
                                    # Draw red rectangle for unrecognized face
                                    cv2.rectangle(img, 
                                            (int(bbox[0]), int(bbox[1])), 
                                            (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), 
                                            (0, 0, 255), 2)
                                    cv2.putText(img, "Unknown",
                                            (int(bbox[0]), int(bbox[1] - 10)), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                            (0, 0, 255), 2)
                            
                            except Exception as e:
                                continue
                    
                    # Display the image
                    FRAME_WINDOW.image(img, channels='BGR')
                    
                    # Update present students list
                    present_students_html = "<div class='present-list'>"
                    for enrollment in sorted(st.session_state.attendance_session['recognized_students']):
                        for idx, student in class_to_student.items():
                            if student['enrollment'] == enrollment:
                                present_students_html += f"""
                                    <div class='present-item'>
                                        ✅ {student['name']} ({enrollment})
                                    </div>
                                """
                    present_students_html += "</div>"
                    present_list_placeholder.markdown(present_students_html, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
            finally:
                # Clean up
                if 'cap' in locals():
                    cap.release()
                cv2.destroyAllWindows()
                st.session_state.attendance_session['active'] = False
                
        elif 'stop_button' in locals() and stop_button:
            st.session_state.attendance_session['active'] = False
            st.success("✨ Attendance session completed!")
            st.balloons()

        # Add help section
        with st.expander("ℹ️ How to Take Attendance"):
            st.markdown("""
                1. **Enter Subject Name**: Type the name of the subject for attendance
                2. **Adjust Settings**: Set confidence threshold in advanced settings
                3. **Start Session**: Click 'Start Attendance' to begin
                4. **Taking Attendance**:
                    - Position students' faces in front of the camera
                    - System will automatically recognize and mark attendance
                    - Green box indicates successful recognition
                    - Red box indicates unrecognized face
                5. **End Session**: Click 'Stop Attendance' when done
                
                **Note**: Make sure students face the camera clearly for best results.
            """)

    elif app_mode == 'View Reports':
        st.header('Attendance Reports')
        
        # Initialize session state variables if they don't exist
        if 'report_generated' not in st.session_state:
            st.session_state.report_generated = False
        if 'report_date' not in st.session_state:
            st.session_state.report_date = None
        if 'selected_subject' not in st.session_state:
            st.session_state.selected_subject = None
        
        # Date Selection
        report_date = st.date_input('Select Date', 
                                value=st.session_state.report_date if st.session_state.report_date else None)
        st.session_state.report_date = report_date
        
        # Get unique subjects from attendance records
        if os.path.exists('attendance.csv'):
            attendance_df = pd.read_csv('attendance.csv')
            subjects = sorted(attendance_df['Subject'].unique())
            
            if len(subjects) > 0:
                selected_subject = st.selectbox('Select Subject', subjects, 
                                            index=subjects.index(st.session_state.selected_subject) if st.session_state.selected_subject in subjects else 0)
                st.session_state.selected_subject = selected_subject
                
                # Use a button to trigger report generation and save the state
                if st.button('Generate Report') or st.session_state.report_generated:
                    st.session_state.report_generated = True
                    try:
                        # Show present students
                        present_students = attendance_df[
                            (attendance_df['Date'] == report_date.strftime('%Y-%m-%d')) & 
                            (attendance_df['Subject'] == selected_subject)
                        ]
                        
                        # Get total students from master list with email addresses
                        master_df = pd.read_csv('students_master.csv')
                        total_students = len(master_df)
                        
                        # Display Statistics in columns
                        col1, col2, col3 = st.columns(3)
                        present_count = len(present_students)
                        absent_count = total_students - present_count
                        attendance_percentage = (present_count / total_students) * 100 if total_students > 0 else 0
                        
                        col1.metric("Total Students", total_students)
                        col2.metric("Present", present_count)
                        col3.metric("Absent", absent_count)
                        
                        # Display attendance percentage with color coding
                        if attendance_percentage >= 75:
                            st.success(f"Attendance Percentage: {attendance_percentage:.2f}%")
                        elif attendance_percentage >= 60:
                            st.warning(f"Attendance Percentage: {attendance_percentage:.2f}%")
                        else:
                            st.error(f"Attendance Percentage: {attendance_percentage:.2f}%")
                        
                        # Present Students Details
                        if not present_students.empty:
                            st.subheader('Present Students')
                            # Sort by time
                            present_students = present_students.sort_values('Time')
                            st.dataframe(
                                present_students[['Enrollment_Number', 'Student_Name', 'Time']]
                                .style.set_properties(**{'background-color': '#90EE90'})
                            )
                            
                            # Export option for present students
                            csv_present = present_students.to_csv(index=False).encode('utf-8')
                            df = csv_present
                            root_node = "data/presentstudent"
                            
                            # Only trigger firebase upload with specific button
                            if st.button("Upload Present Students to Firebase"):
                                success, message = upload_to_firebase(present_students, root_node)
                                initialize_firebase(database_url)
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)
                                    
                            st.download_button(
                                "Download Present Students List",
                                csv_present,
                                f"present_students_{selected_subject}_{report_date}.csv",
                                "text/csv",
                                key='download-present-csv'
                            )
                        
                        absentees, master_df = get_absentees(selected_subject, report_date.strftime('%Y-%m-%d'))
                        if absentees is not None and not absentees.empty:
                            st.subheader('Absent Students')
                            
                            # Ensure master_df has Email column
                            if 'Email' not in master_df.columns:
                                st.text("Email not found")
                                master_df['Email'] = ""  # Add empty Email column if it doesn't exist
                            
                            # Create DataFrame with email column
                            absent_df = pd.merge(
                                absentees,
                                master_df[['Enrollment_Number', 'Email']],
                                on='Enrollment_Number',
                                how='left'
                            )
                            # st.dataframe(absent_df)
                            
                            # Display absentee list with email addresses
                            st.dataframe(
                                absent_df[['Enrollment_Number', 'Student_Name', 'Email_x']]
                                .style.set_properties(**{'background-color': '#FFB6C1'})
                            )
                            
                            # Email notification section
                            st.subheader('Email Notifications')
                            
                            # Use session state for email form inputs
                            if 'sender_email' not in st.session_state:
                                st.session_state.sender_email = "attendancemanagementarcface@gmail.com"
                            if 'email_subject' not in st.session_state:
                                st.session_state.email_subject = f"Absence Notification - {selected_subject} ({report_date.strftime('%Y-%m-%d')})"
                            if 'email_body' not in st.session_state:
                                st.session_state.email_body = f"""Dear {{student_name}},

                        This is to inform you that you were marked absent for {selected_subject} on {report_date.strftime('%Y-%m-%d')}.

                        Please ensure regular attendance in classes.

                        Best regards,
                        College Administration"""
                            
                            email_col1, email_col2 = st.columns(2)
                            
                            with email_col1:
                                sender_email = st.text_input("Sender Email", value=st.session_state.sender_email, key="sender_email_input")
                                st.session_state.sender_email = sender_email
                            with email_col2:
                                sender_password = st.text_input("Email Password", type="password")
                            
                            email_subject = st.text_input("Email Subject", value=st.session_state.email_subject, key="email_subject_input") 
                            st.session_state.email_subject = email_subject
                            
                            email_body = st.text_area("Email Body", value=st.session_state.email_body, key="email_body_input")
                            st.session_state.email_body = email_body
                            
                            if st.button('Send Absence Notifications', key="send_emails_button"):
                                if not sender_email or not sender_password:
                                    st.error("Please provide email credentials")
                                else:
                                    with st.spinner('Sending email notifications...'):
                                        success_count = 0
                                        fail_count = 0
                                        
                                        # Progress bar
                                        progress_bar = st.progress(0)
                                        total_emails = len(absent_df)
                                        
                                        for idx, student in absent_df.iterrows():
                                            try:
                                                # Create message
                                                msg = MIMEMultipart()
                                                msg['From'] = sender_email
                                                msg['To'] = student['Email_x'] 
                                                
                                                # Fix 1: Clean the email subject by removing non-breaking spaces
                                                clean_subject = email_subject.replace('\xa0', ' ')
                                                msg['Subject'] = clean_subject
                                                
                                                # Fix 2: Clean the email body by removing non-breaking spaces
                                                clean_body = email_body.replace('\xa0', ' ')
                                                
                                                # Personalize email body
                                                personalized_body = clean_body.replace(
                                                    "{student_name}", student['Student_Name'])
                                                msg.attach(MIMEText(personalized_body, 'plain', 'utf-8'))  # Fix 3: Specify UTF-8 encoding
                                                
                                                # Connect to SMTP server (Gmail example)
                                                server = smtplib.SMTP('smtp.gmail.com', 587)
                                                server.starttls()
                                                server.login(sender_email, sender_password)
                                                
                                                # Send email
                                                server.send_message(msg)
                                                server.quit()
                                                
                                                success_count += 1
                                                st.success(f"Email sent to {student['Student_Name']}")
                                                
                                            except Exception as e:
                                                fail_count += 1
                                                st.error(f"Failed to send email to {student['Student_Name']}: {str(e)}")
                                            
                                            # Update progress bar
                                            progress_bar.progress((idx + 1) / total_emails)
                                        
                                        # Final status
                                        if success_count > 0:
                                            st.success(f"Successfully sent {success_count} email notifications")
                                        if fail_count > 0:
                                            st.error(f"Failed to send {fail_count} email notifications")
                            
                            # Export options for absent students
                            csv_absent = absent_df.to_csv(index=False).encode('utf-8')
                            df = csv_absent
                            root_node = "data/absentstudents"
                            
                            # Only trigger firebase upload with specific button
                            if st.button("Upload Absent Students to Firebase"):
                                success, message = upload_to_firebase(absent_df, root_node)
                                initialize_firebase(database_url)
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)
                                    
                            st.download_button(
                                "Download Absent Students List",
                                csv_absent,
                                f"absent_students_{selected_subject}_{report_date}.csv",
                                "text/csv",
                                key='download-absent-csv'
                            )
                        
                        # Visualization Section
                        st.subheader('Attendance Visualization')
                        
                        # Pie Chart for Present/Absent
                        fig_pie, ax_pie = plt.subplots()
                        ax_pie.pie(
                            [present_count, absent_count],
                            labels=['Present', 'Absent'],
                            autopct='%1.1f%%',
                            colors=['#90EE90', '#FFB6C1']
                        )
                        ax_pie.set_title(f'Attendance Distribution for {selected_subject}')
                        st.pyplot(fig_pie)
                        
                        # Time Analysis
                        if not present_students.empty:
                            st.subheader('Attendance Timing Analysis')
                            
                            # Convert Time to datetime
                            present_students['Time'] = pd.to_datetime(present_students['Time'])
                            present_students['Hour'] = present_students['Time'].dt.hour
                            
                            # Create hour-wise distribution
                            hour_dist = present_students.groupby('Hour').size().reset_index(name='Count')
                            
                            # Bar chart for timing distribution
                            fig_time, ax_time = plt.subplots(figsize=(10, 6))
                            ax_time.bar(hour_dist['Hour'], hour_dist['Count'])
                            ax_time.set_xlabel('Hour of Day')
                            ax_time.set_ylabel('Number of Students')
                            ax_time.set_title('Attendance Timing Distribution')
                            plt.xticks(hour_dist['Hour'])
                            st.pyplot(fig_time)
                        
                        # Monthly Trend Analysis
                        st.subheader('Monthly Attendance Trend')
                        # Get attendance for the current month
                        current_month = pd.to_datetime(report_date).strftime('%Y-%m')
                        month_attendance = attendance_df[
                            (attendance_df['Subject'] == selected_subject) &
                            (pd.to_datetime(attendance_df['Date']).dt.strftime('%Y-%m') == current_month)
                        ]
                        
                        if not month_attendance.empty:
                            # Group by date and count attendance
                            daily_counts = month_attendance.groupby('Date').size().reset_index(name='Count')
                            daily_counts['Date'] = pd.to_datetime(daily_counts['Date'])
                            
                            # Line plot for monthly trend
                            fig_trend, ax_trend = plt.subplots(figsize=(12, 6))
                            ax_trend.plot(daily_counts['Date'], daily_counts['Count'], marker='o')
                            ax_trend.set_xlabel('Date')
                            ax_trend.set_ylabel('Number of Students')
                            ax_trend.set_title('Daily Attendance Trend')
                            plt.xticks(rotation=45)
                            st.pyplot(fig_trend)
                            
                        # Export Complete Report
                        st.subheader('Export Complete Report')
                        
                        # Prepare report data
                        report_data = {
                            'Report Date': report_date.strftime('%Y-%m-%d'),
                            'Subject': selected_subject,
                            'Total Students': total_students,
                            'Present': present_count,
                            'Absent': absent_count,
                            'Attendance Percentage': f"{attendance_percentage:.2f}%"
                        }
                        
                        report_df = pd.DataFrame([report_data])
                        
                        # Export to Excel with multiple sheets
                        def create_excel_report():
                            with pd.ExcelWriter('attendance_report.xlsx', engine='xlsxwriter') as writer:
                                # Summary Sheet
                                report_df.to_excel(writer, sheet_name='Summary', index=False)
                                
                                # Present Students Sheet
                                if not present_students.empty:
                                    present_students.to_excel(writer, sheet_name='Present Students', index=False)
                                
                                # Absent Students Sheet
                                if absentees is not None and not absentees.empty:
                                    absent_df.to_excel(writer, sheet_name='Absent Students', index=False)
                            
                            with open('attendance_report.xlsx', 'rb') as f:
                                return f.read()
                        
                        excel_report = create_excel_report()
                        st.download_button(
                            "Download Complete Excel Report",
                            excel_report,
                            f"attendance_report_{selected_subject}_{report_date}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key='download-excel'
                        )
                        
                    except Exception as e:
                        st.error(f"An error occurred while generating the report: {str(e)}")
                        
            else:
                st.warning("No attendance records found. Please take attendance first.")
        else:
            st.error("No attendance records found. Please take attendance first.")
    # Add a button to reset the report display
    if st.session_state.report_generated:
        if st.button("Generate New Report"):
            st.session_state.report_generated = False
            st.rerun()

main()
