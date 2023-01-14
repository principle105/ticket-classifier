from tensorflow import keras
from tensorflow.python.keras import layers as L


def build_model(input_shape):
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            L.Lambda(lambda x: x / 255.0),

            L.Conv2D(32, kernel_size=(16, 16), activation="relu"),
            L.MaxPooling2D(pool_size=(8, 8)),

            L.Conv2D(64, kernel_size=(16, 16), activation="relu"),
            L.MaxPooling2D(pool_size=(8, 8)),

            L.Flatten(),

            L.Dropout(0.5),
            L.Dense(64),

            L.Dropout(0.5),
            L.Dense(64),

            L.Dense(
                1, activation="sigmoid"
            ),  # Sigmoid is good for a binary classification problem.
        ]
    )