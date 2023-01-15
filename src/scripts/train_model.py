import tensorflow as tf

from config import (
    BATCH_SIZE,
    DATA_DIR,
    EPOCHS,
    LOG_DIR,
    MODEL_INPUT_SHAPE,
    MODEL_PATH,
    VALIDATION_SPLIT,
)
from model.convolutional_neural_network import build_model


def train_model():
    (
        training_dataset,
        testing_dataset,
    ) = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        image_size=MODEL_INPUT_SHAPE,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        subset="both",
        seed=0,
    )

    model = build_model((*MODEL_INPUT_SHAPE, 3))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)

    model.fit(
        training_dataset,
        epochs=EPOCHS,
        validation_data=testing_dataset,
        callbacks=[tensorboard_callback],
    )
    model.save_weights(MODEL_PATH)
