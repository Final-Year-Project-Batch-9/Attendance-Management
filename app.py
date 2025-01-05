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
from hashlib import sha256
import json

def load_users():
   try:
       with open('users.json', 'r') as f:
           return json.load(f)
   except FileNotFoundError:
       return {'users': []}

def save_users(users):
   with open('users.json', 'w') as f:
       json.dump(users, f)

def hash_password(password):
   return sha256(password.encode()).hexdigest()

def user_exists(username):
   users = load_users()
   return any(user['username'] == username for user in users['users'])

def verify_user(username, password):
   users = load_users()
   hashed_pw = hash_password(password)
   return any(user['username'] == username and user['password'] == hashed_pw 
             for user in users['users'])

def initialize_attendance_df():
   if os.path.exists('attendance.csv'):
       return pd.read_csv('attendance.csv')
   else:
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
       return True
   return False

def get_absentees(subject, date):
   if not os.path.exists('students_master.csv'):
       st.error("Students master list not found!")
       return None
   
   master_df = pd.read_csv('students_master.csv')
   if not os.path.exists('attendance.csv'):
       return master_df
       
   attendance_df = pd.read_csv('attendance.csv')
   present_students = attendance_df[
       (attendance_df['Subject'] == subject) & 
       (attendance_df['Date'] == date)
   ]['Enrollment_Number'].unique()
   
   absentees = master_df[~master_df['Enrollment_Number'].isin(present_students)]
   return absentees

def main():
   if 'logged_in' not in st.session_state:
       st.session_state.logged_in = False

   if not st.session_state.logged_in:
       st.title('Face Recognition Attendance System')
       tab1, tab2 = st.tabs(["Login", "Signup"])
       
       with tab1:
           st.header("Login")
           username = st.text_input("Username", key="login_username")
           password = st.text_input("Password", type='password', key="login_password")
           if st.button("Login"):
               if verify_user(username, password):
                   st.session_state.logged_in = True
                   st.success("Logged in successfully!")
                   st.rerun()
               else:
                   st.error("Invalid credentials")

       with tab2:
           st.header("Signup")
           new_username = st.text_input("Username", key="signup_username")
           new_password = st.text_input("Password", type='password', key="signup_password")
           confirm_password = st.text_input("Confirm Password", type='password')
           
           if st.button("Signup"):
               if not new_username or not new_password:
                   st.error("Please fill all fields")
               elif new_password != confirm_password:
                   st.error("Passwords don't match")
               elif user_exists(new_username):
                   st.error("Username already exists")
               else:
                   users = load_users()
                   users['users'].append({
                       'username': new_username,
                       'password': hash_password(new_password)
                   })
                   save_users(users)
                   st.success("Signup successful! Please login.")

   else:
       st.title('Face Recognition Attendance System')
       os.makedirs('data', exist_ok=True)
       name_list = os.listdir('data')

       if st.sidebar.button('Logout'):
           st.session_state.logged_in = False
           st.rerun()

       st.sidebar.title('Navigation')
       app_mode = st.sidebar.selectbox('Choose Mode',
           ['Data Collection', 'Normalize Data', 'Train Model'])

       webcam_channel = st.sidebar.selectbox(
           'Webcam Channel:',
           ('Select Channel', '0', '1', '2', '3')
       )

       if app_mode == 'Data Collection':
           st.header('Student Registration')
           
           enrollment_number = st.text_input('Enrollment Number:')
           name_person = st.text_input('Student Name:')
           subject_name = st.text_input('Subject Name:')
           img_number = st.number_input('Number of Images:', 50)
           FRAME_WINDOW = st.image([])

           if not webcam_channel == 'Select Channel':
               take_img = st.button('Register Student')
               if take_img:
                   if not enrollment_number or not name_person:
                       st.warning('Please fill enrollment number and student name!')
                   else:
                       folder_name = f"{enrollment_number}_{name_person}"
                       os.makedirs(f'data/{folder_name}', exist_ok=True)
                       
                       if not os.path.exists('students_master.csv'):
                           master_df = pd.DataFrame(columns=['Enrollment_Number', 'Student_Name'])
                       else:
                           master_df = pd.read_csv('students_master.csv')
                       
                       if not master_df[master_df['Enrollment_Number'] == enrollment_number].empty:
                           st.warning('This enrollment number already exists!')
                       else:
                           new_student = {
                               'Enrollment_Number': enrollment_number,
                               'Student_Name': name_person
                           }
                           master_df = pd.concat([master_df, pd.DataFrame([new_student])], ignore_index=True)
                           master_df.to_csv('students_master.csv', index=False)
                           
                           face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                           cap = cv2.VideoCapture(int(webcam_channel))
                           count = 0
                           
                           while True:
                               success, img = cap.read()
                               if not success:
                                   st.error('[INFO] Camera not working!')
                                   break

                               faces = face_classifier.detectMultiScale(img)
                               if len(faces) > 0:
                                   cv2.imwrite(f'data/{folder_name}/{count}.jpg', img)
                                   st.success(f'[INFO] Saved image {count + 1}/{img_number}')
                                   count += 1

                                   for (x, y, w, h) in faces:
                                       cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                               FRAME_WINDOW.image(img, channels='BGR')
                               if count >= img_number:
                                   st.success(f'Registration completed for {name_person}')
                                   break

                           FRAME_WINDOW.image([])
                           cap.release()
                           cv2.destroyAllWindows()

       elif app_mode == 'Normalize Data':
           st.header('Normalize Image Data')
           if st.button('Start Normalization'):
               path_to_dir = "data"
               path_to_save = 'norm_data'
               os.makedirs(path_to_save, exist_ok=True)
               
               detector = MTCNN()
               class_list = os.listdir(path_to_dir)
               
               for name in class_list:
                   st.info(f"Normalizing images for {name}")
                   img_list = glob.glob(os.path.join(path_to_dir, name) + '/*')
                   save_folder = os.path.join(path_to_save, name)
                   os.makedirs(save_folder, exist_ok=True)

                   for img_path in img_list:
                       img = cv2.imread(img_path)
                       detections = detector.detect_faces(img)

                       if len(detections) > 0:
                           right_eye = detections[0]['keypoints']['right_eye']
                           left_eye = detections[0]['keypoints']['left_eye']
                           bbox = detections[0]['box']
                           norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)
                           cv2.imwrite(f'{save_folder}/{os.path.split(img_path)[1]}', norm_img_roi)
                       else:
                           st.warning(f'No face detected in {img_path}')
                   
               st.success('Normalization completed!')

       elif app_mode == 'Train Model':
           st.header('Train Recognition Model')
           if st.button('Start Training'):
               model = ArcFace.loadModel()
               target_size = model.layers[0].input_shape[0][1:3]

               x = []
               y = []
               names = sorted(os.listdir('norm_data'))
               
               for name in names:
                   st.info(f'Processing {name}')
                   img_list = glob.glob(os.path.join('norm_data', name) + '/*')
                   
                   for img_path in img_list:
                       img = cv2.imread(img_path)
                       img_resize = cv2.resize(img, target_size)
                       img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
                       img_pixels = np.expand_dims(img_pixels, axis=0)
                       img_norm = img_pixels/255
                       img_embedding = model.predict(img_norm)[0]
                       
                       x.append(img_embedding)
                       y.append(name)

               df = pd.DataFrame(x, columns=np.arange(512))
               df['names'] = y
               
               x = df.drop('names', axis=1)
               y = pd.factorize(df['names'])[0]

               x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

               model = Sequential([
                   layers.Dense(1024, activation='relu', input_shape=[512]),
                   layers.Dense(512, activation='relu'),
                   layers.Dense(1, activation="sigmoid")
               ])

               model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])

               history = model.fit(x_train, y_train,
                                 epochs=100,
                                 batch_size=32,
                                 validation_data=(x_test, y_test),
                                 callbacks=[
                                     keras.callbacks.ModelCheckpoint(
                                         'model.h5',
                                         monitor='val_accuracy',
                                         save_best_only=True,
                                         mode='max'),
                                     keras.callbacks.EarlyStopping(
                                         monitor='val_accuracy',
                                         patience=20)
                                 ])

               st.success('Model training completed!')
           else:
               st.error("No attendance records found. Please take attendance first.")

       elif app_mode == 'Take Attendance':
           st.header('Mark Attendance')
           subject = st.text_input('Subject:')
           
           if not webcam_channel == 'Select Channel':
               FRAME_WINDOW = st.image([])
               take_attendance = st.button('Start Attendance')
               
               if take_attendance and subject:
                   cap = cv2.VideoCapture(int(webcam_channel))
                   face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                   
                   while True:
                       success, img = cap.read()
                       if not success:
                           st.error('[INFO] Camera not working!')
                           break

                       faces = face_classifier.detectMultiScale(img)
                       if len(faces) > 0:
                           for (x, y, w, h) in faces:
                               cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                       FRAME_WINDOW.image(img, channels='BGR')

                   cap.release()
                   cv2.destroyAllWindows()
               elif take_attendance:
                   st.warning('Please enter subject name!')

       elif app_mode == 'View Reports':
           st.header('Attendance Reports')
           subject = st.text_input('Subject:')
           date = st.date_input('Date:')
           
           if subject and date:
               date_str = date.strftime('%Y-%m-%d')
               absentees = get_absentees(subject, date_str)
               
               if absentees is not None:
                   if not absentees.empty:
                       st.write('Absentees:')
                       st.dataframe(absentees)
                   else:
                       st.success('Full attendance!')
               
               if os.path.exists('attendance.csv'):
                   df = pd.read_csv('attendance.csv')
                   present = df[(df['Subject'] == subject) & (df['Date'] == date_str)]
                   if not present.empty:
                       st.write('Present Students:')
                       st.dataframe(present)

if __name__ == "__main__":
   main()