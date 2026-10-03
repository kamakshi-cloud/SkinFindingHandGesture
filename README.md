# Skin Finding: Real-Time Hand Gesture Recognition

## Project Description

This project detects human hands in real time using a webcam and
recognizes hand gestures based on the number of fingers detected.

The system combines HSV-based skin color detection with MediaPipe
hand landmark detection.

## Technologies Used

- Python
- OpenCV
- NumPy
- MediaPipe
- Computer Vision
- HSV Skin Color Model

## Main Features

1. Real-time webcam input
2. HSV skin color detection
3. Skin mask generation
4. Noise removal using morphological operations
5. Hand landmark detection
6. Finger counting
7. Gesture recognition
8. Interactive gesture actions
9. Real-time visual feedback

## Gesture Mapping

| Fingers | Gesture | Action |
|---------|---------|--------|
| 0 | Fist | STOP |
| 1 | One Finger | SELECT |
| 2 | Two Fingers | NEXT |
| 3 | Three Fingers | PREVIOUS |
| 4 | Four Fingers | NO ACTION |
| 5 | Open Hand | START |

## Project Flow

Webcam
↓
Image Acquisition
↓
HSV Conversion
↓
Skin Color Segmentation
↓
Noise Removal
↓
Hand Detection
↓
Hand Landmarks
↓
Finger Counting
↓
Gesture Recognition
↓
Interactive Action

## How to Run

Activate the virtual environment and run:

python main.py

Press Q to close the webcam window.

## Model File

The project uses:

hand_landmarker.task

The model file must be present in the project folder.

## Project Folder

SkinFindingHandGesture/

├── main.py
├── hand_landmarker.task
├── requirements.txt
├── README.md
├── data/
└── results/