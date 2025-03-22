# Attendance Management System Using Facial Recognition

### **Automated Attendance Tracking with ArcFace Algorithm**

## Table of Contents
1. [Introduction](#introduction)  
2. [Features](#features)  
3. [Technologies Used](#technologies-used)  
4. [Dataset](#dataset)  
5. [System Architecture](#system-architecture)  
6. [Model Workflow](#model-workflow)  
7. [Implementation Details](#implementation-details)  
8. [Results](#results)  
9. [Installation](#installation)  
10. [Usage](#usage)  
11. [Future Enhancements](#future-enhancements)  
---

## 1. Introduction  
Attendance management is a critical task in educational and organizational settings. Traditional methods, such as manual or biometric systems, are often prone to errors and inefficiencies.  
This project leverages the **ArcFace facial recognition algorithm** to automate attendance tracking, providing a real-time, accurate, and user-friendly solution.

---

## 2. Features  
✅ **Real-time Image Capturing**  
✅ **Real-time Face Recognition**  
✅ **Automated Attendance Marking**  
✅ **Email Notifications to Absentees**  
✅ **Download Attendance Records for a Selected Date**  

---

## 3. Technologies Used  
- **Python**: Backend logic and processing  
- **ArcFace**: Facial recognition algorithm  
- **Streamlit**: Interactive user interface and analytics dashboard  
- **OpenCV**: Webcam integration and image preprocessing  
- **Pandas**: Data manipulation and CSV storage  
- **SMTP Library**: Email notifications for absentees  

---

## 4. Dataset  
This project does not require a labeled dataset for training since it uses a **pre-trained ArcFace model with ResNet-34**. However, it requires face images for inference.  

### **Input Images**  
- The model processes **cropped and aligned face images**  
- **Face detection and alignment** are performed using **MTCNN**  

### **Example Dataset Structure**  
If you are using your own dataset, organize the images like this: 
dataset/ │── person_1/ │ ├── image1.jpg │ ├── image2.jpg │── person_2/ │ ├── image1.jpg │ ├── image2.jpg
Each person's images are stored in separate folders.  
The model extracts **512-dimensional embeddings** for each face.  

### **Pre-trained Weights**  
The model loads pre-trained ArcFace weights:  
📌 **arcface_weights.h5** (Downloaded automatically from GitHub)  

If not found, the weights are downloaded from:  
🔗 [ArcFace Weights](https://github.com/)  

---

## 5. System Architecture  
![System Architecture](https://github.com/user-attachments/assets/8db380e6-a2f9-41ad-b56b-07e8835ee221)  

---

## 6. Model Workflow  

1️⃣ **ArcFace Initialization**  
   - Setting up and loading the ArcFace algorithm for face recognition.  

2️⃣ **Streamlit Application**  
   - Creating an interactive interface for capturing images, attendance tracking, and analysis.  

3️⃣ **Image Capture**  
   - Capturing student images in real-time using a webcam.  

4️⃣ **Image Preprocessing**  
   - Real-time preprocessing (**resizing, normalization**) of images for efficient model training.  

5️⃣ **Attendance Marking**  
   - Detecting and marking student presence or absence through facial recognition.  

6️⃣ **Data Storage**  
   - Storing attendance records in a **CSV file** for further analysis.  

7️⃣ **Attendance Analysis and Notification**  
   - A Streamlit-based page for visual analysis and sending emails to absent students.  

---

## 7. Implementation Details  
- The application **automatically detects and registers faces** when a person enters the frame.  
- A **CSV file** is maintained to keep track of attendance records.  
- If a student is absent, the system automatically sends an **email notification**.  

---

## 8. Results  
🔹 **Accuracy**: Achieved an attendance recognition accuracy of **95%+** using ArcFace.  
🔹 **Efficiency**: Attendance marking is done in **real-time** with minimal processing delay.  
🔹 **Scalability**: Can be deployed across multiple classrooms or offices with minor adjustments.  

---

## 9. Installation  

### **Requirements**  
Ensure you have Python installed (**Python 3.8+ recommended**).  

### **Step 1: Clone the Repository**  
```sh
git clone https://github.com/your-username/attendance-management.git
cd attendance-management
```

### **Step 2: Install Dependencies**  
```sh
pip install -r requirements.txt
```

### **Step 3: Run the Application**  
```sh
streamlit run app.py
```

---

## 10. Usage  
1. Open the **Streamlit** interface.  
2. Click **"Start Attendance"** to begin real-time face recognition.  
3. Attendance is automatically marked in a **CSV file**.  
4. Download attendance records using the **"Download"** button.  
5. Emails will be sent to absentees at the end of the session.  

---

## 11. Future Enhancements  
🔹 **Cloud Integration**: Store attendance records on **Google Drive/AWS**.  
🔹 **Mobile App**: Develop an **Android/iOS** version.  
🔹 **Multi-Camera Support**: Handle multiple camera feeds simultaneously.  
🔹 **Live Dashboard**: Display real-time attendance statistics.  
 




