Real-Time Person Detection & Tracking

Description

This project detects and tracks people in real time using Python and OpenCV. It uses a CPU-friendly HOG detector to find humans, and a centroid tracker to assign unique IDs and follow them across frames. It also shows FPS.

Requirements

Python 3.x

Libraries:

pip install opencv-python numpy

How to Run

Run the program:

python person_tracker.py


A window will open showing:

Green bounding boxes around detected people

Red centroid dot at the center

Red ID labels for each person

FPS at top-left

Press q to quit

Notes

HOG detects full body only, not faces or hands

IDs may change if detection fails or person leaves frame

FPS is CPU dependent; should be ≥15 FPS on a laptop
