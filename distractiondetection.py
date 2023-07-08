import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model_path = 'model.h5'
model = load_model(model_path)

# Define the class labels
class_labels = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9','class10']

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frames from the webcam
    ret, frame = cap.read()

    # Preprocess the frame
    frame = cv2.resize(frame, (64, 64))
    frame = image.img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0

    # Perform object detection
    predictions = model.predict(frame)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_class]

    # Display the predictions
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
