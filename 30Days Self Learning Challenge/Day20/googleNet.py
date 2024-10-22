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
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # Reduced filters
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # Reduced filters
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))  # Reduced filters
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))  # Reduced units
    model.add(layers.Dense(10, activation='softmax'))
    return model

# Define a smaller GoogLeNet model (Inception v1)
def create_googlenet():
    inputs = layers.Input(shape=(32, 32, 3))

    # First convolution layer
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)  # Reduced filters
    x = layers.MaxPooling2D((2, 2))(x)

    # Second convolution layer
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)  # Reduced filters
    x = layers.MaxPooling2D((2, 2))(x)

    # Inception modules
    def inception_module(x, filters):
        branch1x1 = layers.Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)
        
        branch3x3 = layers.Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
        branch3x3 = layers.Conv2D(filters[1], (3, 3), padding='same', activation='relu')(branch3x3)
        
        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = layers.Conv2D(filters[2], (1, 1), padding='same', activation='relu')(branch_pool)
        
        return layers.concatenate([branch1x1, branch3x3, branch_pool], axis=-1)

    # Adding smaller inception modules
    x = inception_module(x, [16, 32, 16])  # Reduced filters
    x = inception_module(x, [32, 64, 32])  # Reduced filters
    
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
                        epochs=10, 
                        batch_size=64, 
                        verbose=2)
    return history

# Create and train models
custom_cnn = create_custom_cnn()
googlenet = create_googlenet()

print("Training Custom CNN...")
custom_cnn_history = train_model(custom_cnn, "Custom CNN")

print("\nTraining GoogLeNet...")
googlenet_history = train_model(googlenet, "GoogLeNet")

# Plotting the results
def plot_results(custom_cnn_history, googlenet_history):
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(custom_cnn_history.history['accuracy'], label='Custom CNN Train')
    plt.plot(custom_cnn_history.history['val_accuracy'], label='Custom CNN Validation')
    plt.plot(googlenet_history.history['accuracy'], label='GoogLeNet Train')
    plt.plot(googlenet_history.history['val_accuracy'], label='GoogLeNet Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(custom_cnn_history.history['loss'], label='Custom CNN Train')
    plt.plot(custom_cnn_history.history['val_loss'], label='Custom CNN Validation')
    plt.plot(googlenet_history.history['loss'], label='GoogLeNet Train')
    plt.plot(googlenet_history.history['val_loss'], label='GoogLeNet Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Plotting the results
plot_results(custom_cnn_history, googlenet_history)
