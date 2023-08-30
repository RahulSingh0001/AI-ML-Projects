import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('cnnModel.h5')  # Make sure to replace with your model's filename

# Mapping of class indices to letters
class_to_letter = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

# Load and preprocess the image for prediction
def preprocess_image(image):
    resized = cv2.resize(image, (28, 28))  # Resize the image to match the model's input shape
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    normalized = grayscale / 255.0
    return normalized.reshape(1, 28, 28, 1)  # Reshape to match the model's input shape


# Open the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Draw a rectangle for the region of interest (hand gesture)
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    
    # Extract the region of interest
    roi = frame[100:300, 100:300]
    
    # Preprocess and predict
    preprocessed = preprocess_image(roi)
    prediction = model.predict(preprocessed)
    predicted_class = np.argmax(prediction)
    predicted_letter = class_to_letter[predicted_class]
    
    # Display the predicted letter on the frame
    cv2.putText(frame, predicted_letter, (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    
    # Display the frame
    cv2.imshow('Sign Language Alphabet Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
