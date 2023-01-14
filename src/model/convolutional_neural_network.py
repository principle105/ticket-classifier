from tensorflow import keras
from tensorflow.python.keras import layers as L


def build_model(input_shape):
    return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            L.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            L.MaxPooling2D(pool_size=(2, 2)),
            L.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            L.MaxPooling2D(pool_size=(2, 2)),
            L.Flatten(),
            L.Dropout(0.5),
            L.Dense(
                1, activation="sigmoid"
            ),  # Sigmoid is good for a binary classification problem.
        ]
    )


model = build_model((64, 64, 3))
