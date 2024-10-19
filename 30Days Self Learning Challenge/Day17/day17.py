import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Step 1: Generate synthetic time series data
def generate_synthetic_data(num_samples=1000, time_steps=1441):
    # Generating random time series data
    X = np.random.rand(num_samples, time_steps, 1)  # Shape: (num_samples, time_steps, 1)
    # Creating binary labels based on some condition (for demonstration purposes)
    y = (np.mean(X, axis=1) > 0.5).astype(int)  # Labels are 1 if mean of the sample > 0.5, else 0
    return X, y

# Load your synthetic time series data
X, y = generate_synthetic_data()

# Step 2: Split the dataset into training and test sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, y_train = X[:split_index], y[:split_index]
X_test, y_test = X[split_index:], y[split_index:]



# real magic is here real computation learning algorithm that we CALL CNNS
# Step 3: Create the CNN model
cnn_model = models.Sequential([
    layers.Input(shape=(1441, 1)),  # Input layer with shape (1441, 1)
    layers.Conv1D(16, kernel_size=3, activation='relu'),  # 1st Conv1D layer
    layers.MaxPooling1D(pool_size=2),  # 1st MaxPooling layer
    layers.Conv1D(32, kernel_size=3, activation='relu'),  # 2nd Conv1D layer
    layers.MaxPooling1D(pool_size=2),  # 2nd MaxPooling layer
    layers.Flatten(),  # Flatten the output
    layers.Dense(1, activation='sigmoid')  # Dense layer for binary classification
])

# Step 4: Compile the model
cnn_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Step 5: Train the model and store the history
history = cnn_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Step 6: Plot training & validation loss
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Step 7: Plot training & validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
