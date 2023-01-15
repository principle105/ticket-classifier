import numpy as np
from PIL import Image

from config import MODEL_INPUT_SHAPE


# Loads an image file, then resizes it to the correct size for the model.
def load_image(filepath: str):
    im = Image.open(filepath)
    im = im.convert("RGB")
    im = im.resize(
        reversed(MODEL_INPUT_SHAPE)
    )  # This has to be reversed as the model input shape is (height, width)
    im = np.asarray(im)

    return im
