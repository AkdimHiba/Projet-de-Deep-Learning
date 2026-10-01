# 🏥 Intelligent Healthcare Protocol Monitoring System

A real-time **Computer Vision and Deep Learning** system designed to automatically monitor compliance with healthcare and safety protocols in healthcare environments.

The system analyzes video streams from a webcam or surveillance camera and detects several protocol violations, generating visual alerts in real time.

## 📌 Project Overview

Manual monitoring of healthcare protocols can be difficult, costly, and non-continuous. This project aims to provide an intelligent automated solution capable of analyzing video streams and detecting potential safety and hygiene violations.

The system currently monitors **four protocols**:

* 😷 **Face Mask Detection**
* 😴 **Drowsiness Detection**
* 📱 **Mobile Phone Usage Detection**
* 🧤 **Medical Glove Detection**

The project combines different Deep Learning and Computer Vision techniques depending on the detection task.

## 🎯 Objectives

* Automate the monitoring of healthcare protocols.
* Detect protocol violations in real time.
* Generate visual alerts when a violation is detected.
* Evaluate the performance of the different detection modules.
* Provide a modular system that can be extended with additional protocols.

## 🔍 Detection Modules

### 1. Face Mask Detection

A **MobileNetV2** model with **Transfer Learning** is used to classify the detected face into three categories:

* `Mask`
* `Mask_Worn_Incorrectly`
* `No_Mask`

This module is designed to identify whether the mask is correctly positioned on the face.

### 2. Drowsiness Detection

The drowsiness detection module uses **MediaPipe FaceMesh** to extract facial landmarks.

The **Eye Aspect Ratio (EAR)** is then calculated to identify prolonged eye closure.

The system can therefore detect potential drowsiness and generate an alert when the predefined conditions are met.

### 3. Mobile Phone Detection

**YOLOv8** is used to detect mobile phones in the video stream.

When a phone is detected in a monitored situation, the system generates a visual warning indicating potential protocol violation.

### 4. Medical Glove Detection

An additional protocol was introduced to monitor the use of medical gloves.

A **YOLOv8-based object detection model** can be trained to detect:

* `Gloves`
* `No_Gloves`
* `Improper_Gloves` *(optional depending on the dataset)*

This module aims to improve monitoring of personal protective equipment and hygiene practices.

## 🏗️ System Workflow

```text
Camera / Webcam
      │
      ▼
Video Stream
      │
      ▼
Frame Processing
      │
      ├──────────────► Face Mask Detection
      │                MobileNetV2
      │
      ├──────────────► Drowsiness Detection
      │                MediaPipe + EAR
      │
      ├──────────────► Phone Detection
      │                YOLOv8
      │
      └──────────────► Glove Detection
                       YOLOv8
                             
      ▼
Results Fusion
      │
      ▼
Protocol Violation Detection
      │
      ▼
Real-Time Visual Alerts
```

## 🧠 Technologies

| Category         | Technologies                    |
| ---------------- | ------------------------------- |
| Programming      | Python                          |
| Deep Learning    | TensorFlow, Keras               |
| Object Detection | YOLOv8                          |
| Computer Vision  | OpenCV                          |
| Facial Landmarks | MediaPipe FaceMesh              |
| Classification   | MobileNetV2 + Transfer Learning |
| Data Processing  | NumPy, Pandas                   |
| Development      | Jupyter Notebook / Google Colab |

## 📊 Model Evaluation

The individual modules can be evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

For real-time performance, the system can also be evaluated using:

* FPS (Frames Per Second)
* Processing time per frame
* Detection latency
## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/healthcare-protocol-monitoring.git
cd healthcare-protocol-monitoring
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

## ▶️ Usage

Run the main application:

```bash
python src/main.py
```

The application accesses the webcam, processes the video stream, runs the different detection modules, and displays the detected violations and alerts in real time.

## 📈 Future Improvements

Possible future improvements include:

* Multi-person tracking.
* Improved glove detection under occlusion.
* Additional Personal Protective Equipment (PPE) detection.
* Improved performance on low-end hardware.
* Web-based monitoring dashboard.
* Automatic logging of detected violations.
* Deployment on edge devices.
* Integration of additional healthcare safety protocols.

## 🎓 Academic Project

This project was developed as part of an academic **Deep Learning ** project focused on intelligent monitoring of healthcare and safety protocols.
