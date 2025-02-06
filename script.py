import cv2
import time
from deepface import DeepFace

# Load Haar cascades for various face angles and features
frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Start video capture
cap = cv2.VideoCapture(0)

# Variables for FPS calculation
prev_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame for performance
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale and RGB
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect frontal and side-profile faces
    frontal_faces = frontal_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    left_profiles = profile_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Detect right profiles from flipped frame
    flipped_gray = cv2.flip(gray_frame, 1)
    right_profiles = profile_cascade.detectMultiScale(flipped_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    right_faces = [(frame.shape[1] - x - w, y, w, h) for (x, y, w, h) in right_profiles]

    # Combine all detected faces
    all_faces = list(frontal_faces) + list(left_profiles) + list(right_faces)

    for (x, y, w, h) in all_faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the face ROI for emotion analysis
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the ROI
        emotion = "Unknown"
        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            if isinstance(result, list):
                result = result[0]
            emotion = result.get('dominant_emotion', 'Unknown')
        except Exception as e:
            print(f"Error during emotion analysis: {e}")

        # Display emotion label above the face
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Detect additional features within the face (eyes and smile)
        face_gray = gray_frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
        smile = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))

        # Draw rectangles around detected eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)

        # Draw rectangle around detected smile
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(frame, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 255), 2)

    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Real-time Multi-Profile Emotion Detection', frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
