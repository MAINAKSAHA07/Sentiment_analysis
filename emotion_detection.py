import cv2
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array

# Define the path to your model and emotion categories
model_path = 'improved_emotion_detection_model.h5'  # Path to your trained model
emotion_categories = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

# Load the trained model
model = load_model(model_path)

# Initialize the LabelEncoder and fit it with the emotion categories
label_encoder = LabelEncoder()
label_encoder.fit(emotion_categories)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Start capturing video from the webcam
while True:
    # Capture each frame from the webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert the frame to grayscale (required for emotion detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use OpenCV's face detection method to find faces in the frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Get the number of faces detected
    num_faces = len(faces)

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        face = gray[y:y + h, x:x + w]
        
        # Resize the face to 48x48 (same size as the input to your trained model)
        face_resized = cv2.resize(face, (48, 48))
        
        # Normalize the pixel values
        face_resized = face_resized.astype('float32') / 255.0
        
        # Convert to an array and expand the dimensions to match the model input shape
        face_resized = img_to_array(face_resized)
        face_resized = np.expand_dims(face_resized, axis=0)
        face_resized = np.expand_dims(face_resized, axis=-1)  # Ensure single channel (grayscale)

        # Predict the emotion from the face
        emotion_probabilities = model.predict(face_resized)
        
        # Get the predicted emotion (the one with the highest probability)
        predicted_label = np.argmax(emotion_probabilities, axis=1)
        
        # Convert the predicted label to the actual emotion name
        predicted_emotion = label_encoder.inverse_transform(predicted_label)
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the predicted emotion above the rectangle
        cv2.putText(frame, predicted_emotion[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the number of detected faces on the screen
    cv2.putText(frame, f"Faces: {num_faces}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with the emotion label
    cv2.imshow('Emotion Detection', frame)

    # Break the loop and close the webcam when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
