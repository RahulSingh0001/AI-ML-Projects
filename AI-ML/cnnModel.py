import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load the CSV file containing training data
training_data = pd.read_csv('sign_mnist_train.csv')  # Replace with your CSV file's filename

# Extract labels and pixel values from the training data
labels = training_data['label'].values
pixels = training_data.iloc[:, 1:].values

# Reshape pixel values to image dimensions (assuming the pixels are in a flat format)
num_samples = pixels.shape[0]
pixels_reshaped = pixels.reshape(num_samples, 28, 28, 1)  # Assuming 28x28 images

# Normalize pixel values to the range [0, 1]
pixels_normalized = pixels_reshaped / 255.0

# Convert labels to one-hot encoded format
labels_encoded = to_categorical(labels)

# Split the data into training and validation sets
training_images, validation_images, training_labels, validation_labels = train_test_split(
    pixels_normalized, labels_encoded, test_size=0.2, random_state=42)

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(25, activation='softmax')  # Adjusted to 25 classes
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


# Train the model
model.fit(training_images, training_labels, epochs=10, validation_data=(validation_images, validation_labels))

# Save the trained model
model.save('cnnModel.keras')
