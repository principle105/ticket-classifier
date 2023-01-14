import tensorflow as tf

from config import (BATCH_SIZE, DATA_DIR, EPOCHS, MODEL_INPUT_SHAPE,
                    MODEL_PATH, VALIDATION_SPLIT)
from model.convolutional_neural_network import build_model


def train_model():
    training_dataset, testing_dataset = tf.keras.preprocessing.image_dataset_from_directory(DATA_DIR, image_size=MODEL_INPUT_SHAPE, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT, subset="both", seed=0)

    model = build_model((*MODEL_INPUT_SHAPE, 3))
    model.compile('adam', 'binary_crossentropy')

    model.fit(training_dataset, epochs=EPOCHS, validation_data=testing_dataset)
    model.save_weights(MODEL_PATH)