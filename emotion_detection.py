import cv2  # for image processing
import numpy as np  # for numerical operations
from keras.models import load_model  # for loading the pre-trained CNN model

# Load the pre-trained model
model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5')

# Define the emotions
EMOTIONS = ["angry", "disgust", "scared",
            "happy", "sad", "surprised", "neutral"]

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # Process each face detected
    for (x, y, w, h) in faces:
        # Extract the face ROI
        face_roi = gray[y:y + h, x:x + w]

        # Resize the face ROI
        face_roi = cv2.resize(face_roi, (64, 64))

        # Normalize the face ROI
        face_roi = face_roi.astype("float") / 255.0

        # Reshape the face ROI
        face_roi = np.reshape(face_roi, (1, 64, 64, 1))

        # Make predictions on the face ROI
        preds = model.predict(face_roi)[0]

        # Determine the emotion with the highest probability
        emotion = EMOTIONS[np.argmax(preds)]

        # Display the emotion text on the frame
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Facial Emotion Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
