# SkinFinding – Real-Time Hand Gesture Recognition

## 📌 Project Overview

**SkinFinding – Real-Time Hand Gesture Recognition for Interactive Systems using Skin Color Models** is a computer vision project designed to detect and recognize hand gestures in real time using a webcam.

The system processes live camera frames, identifies the hand region, detects hand landmarks, and determines the number of raised fingers. The project combines **Python, OpenCV, and MediaPipe** to provide real-time hand gesture recognition through an interactive application.

This project demonstrates how computer vision can be used to create natural and touch-free interaction systems.

---

## 🎯 Objectives

The main objectives of this project are:

* To detect a human hand from a live webcam feed.
* To identify hand landmarks in real time.
* To recognize different hand gestures based on finger positions.
* To count the number of raised fingers.
* To provide immediate visual feedback to the user.
* To demonstrate the use of computer vision for interactive systems.

---

## ✨ Features

* 🎥 Real-time webcam processing
* ✋ Hand detection
* 📍 Hand landmark detection
* ☝️ Finger counting
* 👋 Hand gesture recognition
* 🖥️ Interactive graphical interface
* ⚡ Real-time visual feedback
* 🐍 Python-based implementation
* 📷 OpenCV-based camera and image processing
* 🤖 MediaPipe-based hand landmark detection

---

## 🛠️ Technologies Used

| Technology             | Purpose                             |
| ---------------------- | ----------------------------------- |
| **Python**             | Main programming language           |
| **OpenCV**             | Image processing and webcam capture |
| **MediaPipe**          | Hand landmark detection             |
| **NumPy**              | Numerical and array operations      |
| **Visual Studio Code** | Development environment             |
| **Git & GitHub**       | Version control and project hosting |

---

## 💻 System Requirements

### Hardware

* Computer/Laptop
* Webcam
* Minimum 4 GB RAM recommended
* Internet connection for initial package installation

### Software

* Windows/Linux/macOS
* Python 3.x
* Visual Studio Code or another Python IDE
* Git
* Required Python packages listed in `requirements.txt`

---

## 📂 Project Structure

```text
SkinFindingHandGesture/
│
├── main.py
├── main_backup.py
├── hand_landmarker.task
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
```

### File Description

**`main.py`**
Contains the main implementation of the real-time hand gesture recognition system.

**`main_backup.py`**
Backup version of the main Python implementation.

**`hand_landmarker.task`**
MediaPipe hand landmark model file used for hand detection and landmark tracking.

**`requirements.txt`**
Contains the Python dependencies required to run the project.

**`data/`**
Contains project-related data/resources used by the application.

**`.gitignore`**
Specifies files and folders that should not be uploaded to the repository, such as the Python virtual environment and generated files.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/kamakshi-cloud/SkinFindingHandGesture.git
```

### 2. Navigate to the project folder

```bash
cd SkinFindingHandGesture
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

#### Windows PowerShell

```powershell
venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
venv\Scripts\activate
```

### 5. Install the required packages

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

After installing the dependencies, activate the virtual environment and run:

```bash
python main.py
```

The application will access the webcam and begin processing the live video stream.

Make sure your computer has a working webcam and that permission to access the camera has been granted.

---

## 🔄 Working Process

The system follows these basic steps:

```text
Webcam
   ↓
Capture Live Frame
   ↓
Image Processing
   ↓
Hand Detection
   ↓
Hand Landmark Detection
   ↓
Finger Position Analysis
   ↓
Finger Counting
   ↓
Gesture Recognition
   ↓
Display Result
```

### Step 1 – Webcam Input

The webcam continuously captures live video frames.

### Step 2 – Frame Processing

Each frame is processed using computer vision techniques to prepare it for hand detection.

### Step 3 – Hand Detection

The system identifies the hand present in the camera frame.

### Step 4 – Landmark Detection

MediaPipe detects key points, or landmarks, on the hand and fingers.

### Step 5 – Finger Analysis

The positions of the detected landmarks are analyzed to determine which fingers are raised.

### Step 6 – Gesture Recognition

The detected finger configuration is used to identify the corresponding gesture.

### Step 7 – Display

The recognized hand information and finger count are displayed to the user in real time.

---

## 🧠 Computer Vision Approach

The project uses computer vision to analyze visual information obtained from a webcam.

Hand landmarks provide important positional information about the fingers and palm. By examining the relative positions of these landmarks, the system can determine finger states and recognize gestures.

The project also explores the use of **skin color models** as part of the computer vision approach for identifying hand regions.

---

## 📊 Example Gesture Recognition

The system can analyze different finger configurations such as:

| Hand Configuration   | Finger Count |
| -------------------- | -----------: |
| Closed hand          |            0 |
| One raised finger    |            1 |
| Two raised fingers   |            2 |
| Three raised fingers |            3 |
| Four raised fingers  |            4 |
| Open hand            |            5 |

The recognized result depends on the detected hand landmarks and their positions in the camera frame.

---

## 📸 Project Screenshots

### Real-Time Hand Gesture Detection

The application captures live webcam input and displays the detected hand landmarks and gesture information in real time.

![Real-Time Hand Gesture Detection](screenshots/Screenshot%202026-09-11%20172943.png)

### Gesture Recognition Interface

The application provides visual feedback for the detected hand gesture and finger count.

![Gesture Recognition Interface](screenshots/Screenshot%202026-09-11%20173744.png)
