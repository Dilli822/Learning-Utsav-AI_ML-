import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import MobileNet
import matplotlib.pyplot as plt

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert class vectors to binary class matrices (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Function to create a custom CNN model
def create_custom_cnn():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Create and compile the custom CNN model
custom_model = create_custom_cnn()
custom_model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Train the custom CNN model
custom_history = custom_model.fit(x_train, y_train, epochs=15, 
                                   validation_data=(x_test, y_test), 
                                   batch_size=64)

# Load MobileNet model, excluding the top layer (for transfer learning)
base_model = MobileNet(input_shape=(32, 32, 3), include_top=False, weights='imagenet')

# Freeze the base model
base_model.trainable = False

# Create a new model for MobileNet
mobile_net_model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),  # Custom dense layer
    layers.Dropout(0.5),                  # Dropout for regularization
    layers.Dense(10, activation='softmax') # Output layer
])

# Compile the MobileNet model
mobile_net_model.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])

# Train the MobileNet model
mobile_history = mobile_net_model.fit(x_train, y_train, epochs=15, 
                                       validation_data=(x_test, y_test), 
                                       batch_size=64)

# Evaluate both models
custom_test_loss, custom_test_acc = custom_model.evaluate(x_test, y_test)
mobile_test_loss, mobile_test_acc = mobile_net_model.evaluate(x_test, y_test)

print(f"Custom CNN Test accuracy: {custom_test_acc:.4f}")
print(f"MobileNet Test accuracy: {mobile_test_acc:.4f}")

# Plot training & validation accuracy values for both models
plt.figure(figsize=(12, 10))

# Custom CNN Accuracy
plt.subplot(2, 2, 1)
plt.plot(custom_history.history['accuracy'], label='Custom Train Accuracy')
plt.plot(custom_history.history['val_accuracy'], label='Custom Validation Accuracy')
plt.title('Custom CNN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# MobileNet Accuracy
plt.subplot(2, 2, 2)
plt.plot(mobile_history.history['accuracy'], label='MobileNet Train Accuracy')
plt.plot(mobile_history.history['val_accuracy'], label='MobileNet Validation Accuracy')
plt.title('MobileNet Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Custom CNN Loss
plt.subplot(2, 2, 3)
plt.plot(custom_history.history['loss'], label='Custom Train Loss')
plt.plot(custom_history.history['val_loss'], label='Custom Validation Loss')
plt.title('Custom CNN Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# MobileNet Loss
plt.subplot(2, 2, 4)
plt.plot(mobile_history.history['loss'], label='MobileNet Train Loss')
plt.plot(mobile_history.history['val_loss'], label='MobileNet Validation Loss')
plt.title('MobileNet Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
