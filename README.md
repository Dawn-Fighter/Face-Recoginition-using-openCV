# Real-time Multi-Profile Emotion Detection

This project is a real-time facial recognition and emotion detection system using OpenCV and DeepFace. It detects faces from various angles (frontal and profile) and analyzes emotions from live video input.

## Features
- Real-time face detection using Haar cascades.
- Multi-profile face detection (frontal, left, and right profiles).
- Emotion recognition using DeepFace.
- Detection of facial features such as eyes and smiles.
- FPS display for performance monitoring.

## Installation

### Prerequisites
Ensure you have Python installed along with the required dependencies:

```bash
pip install opencv-python deepface numpy
```

## Usage

Run the script to start real-time face and emotion detection:

```bash
python face_recognition.py
```

Press `q` to exit the application.

## How It Works
1. Captures video frames from the webcam.
2. Converts frames to grayscale and RGB.
3. Detects faces (frontal and profile) using Haar cascades.
4. Uses DeepFace to analyze emotions in detected faces.
5. Detects additional facial features such as eyes and smiles.
6. Displays real-time FPS performance.
7. Shows detected emotions on the screen.

## Dependencies
- OpenCV
- DeepFace
- NumPy

## Example Output
The application displays a live feed with detected faces, emotions, and FPS information:

- **Green box**: Face detected
- **Blue box**: Eyes detected
- **Yellow box**: Smile detected
- **Red text**: Detected emotion

## License
This project is licensed under the MIT License.

