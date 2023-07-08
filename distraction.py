# Import the necessary libraries
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

# Set the path to your dataset
dataset_path = 'C:/Users/rames/Downloads/state-farm-distracted-driver-detection/imgs/train'
num_classes=10
# Define the input shape of your images
input_shape = (64, 64, 3)  # Adjust the size according to your images

# Load the images and labels from the dataset
def load_dataset(path):
    images = []
    labels = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = Image.open(image_path).resize(input_shape[:2])
            images.append(np.array(image))
            labels.append(label)
    return np.array(images), np.array(labels)

# Load and preprocess the dataset
images, labels = load_dataset(dataset_path)
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
labels = to_categorical(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build your model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set the path to save the model
model_save_path = 'model.h5'

# Define a callback to save the best model during training
checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, callbacks=[checkpoint])

# Save the final trained model
model.save(model_save_path)
