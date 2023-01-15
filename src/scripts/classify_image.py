from utils.image_utils import load_image
from model.convolutional_neural_network import build_model
from config import MODEL_INPUT_SHAPE, MODEL_PATH
import numpy as np


def classify_image(filepath):
    # Load the image into a format the model can understand.
    im = load_image(filepath)
    im = np.expand_dims(im, axis=0)

    # Load the model.
    model = build_model((*MODEL_INPUT_SHAPE, 3))
    model.load_weights(MODEL_PATH)

    # Calculate the results.
    prob = model.predict(im)[0][0]

    result = round(prob)
    confidence = prob if result else (1 - prob)
    confidence = round(confidence * 100, 2)  # Convert the confidence to a percentage.

    if prob >= 0.5:
        print(f"The ticket is real with {confidence}% confidence.")

    else:
        print(f"The ticket is fake with {confidence}% confidence.")
