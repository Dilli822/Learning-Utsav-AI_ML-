import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0

# Convert labels to categorical one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define a smaller Custom CNN model
def create_custom_cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Define a simple ResNet model
def create_resnet():
    def residual_block(x, filters):
        shortcut = x
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(filters, (3, 3), padding='same')(x)
        x = layers.add([x, shortcut])  # Add shortcut
        x = layers.Activation('relu')(x)
        return x

    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)

    for _ in range(3):  # Stack 3 residual blocks
        x = residual_block(x, 16)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation='softmax')(x)

    model = models.Model(inputs, x)
    return model

# Compile and train both models
def train_model(model, model_name):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, 
                        validation_data=(x_test, y_test),
                        epochs=5, 
                        batch_size=64, 
                        verbose=2)
    return history

# Create and train models
custom_cnn = create_custom_cnn()
resnet = create_resnet()

print("Training Custom CNN...")
custom_cnn_history = train_model(custom_cnn, "Custom CNN")

print("\nTraining ResNet...")
resnet_history = train_model(resnet, "ResNet")

# Plotting the results
def plot_results(custom_cnn_history, resnet_history):
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(custom_cnn_history.history['accuracy'], label='Custom CNN Train')
    plt.plot(custom_cnn_history.history['val_accuracy'], label='Custom CNN Validation')
    plt.plot(resnet_history.history['accuracy'], label='ResNet Train')
    plt.plot(resnet_history.history['val_accuracy'], label='ResNet Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(custom_cnn_history.history['loss'], label='Custom CNN Train')
    plt.plot(custom_cnn_history.history['val_loss'], label='Custom CNN Validation')
    plt.plot(resnet_history.history['loss'], label='ResNet Train')
    plt.plot(resnet_history.history['val_loss'], label='ResNet Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plotting the results
plot_results(custom_cnn_history, resnet_history)
