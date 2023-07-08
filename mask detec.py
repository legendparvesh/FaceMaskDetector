import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
MODEL_PATH = 'mymodel.h5'
LABELS = ['mask', 'no mask']

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Open a video capture object (webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (150, 150))
    normalized_frame = resized_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Run the inference
    predictions = model.predict(input_frame)
    print(predictions,)
    predicted_class = np.argmax(predictions)
    print(predicted_class)
    if predictions>0.7:
        print('mask')
        label='mask'
    else:
        print('no mask')
        label='no mask'
    # Get the predicted label
    #label = LABELS[predicted_class]

    # Display the label on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()
