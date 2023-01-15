import os
import random

import cv2
import imutils
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from tqdm import tqdm

from utils.file_utils import create_dirs


def _get_translation_amt(size):
    # Dividing by 10 to avoid major translations
    return random.randint(-int(size / 10), int(size / 10))


def _get_save_path(augmented_path: str, img_name: str, name: str, index: int):
    return f"{augmented_path}/{img_name.split('_')[1].split('.')[0]}_{name}_{index}.jpg"


def _save_img(file_name: str, modifier=None):
    if modifier is None:
        cv2.imwrite(file_name)
        return

    cv2.imwrite(
        file_name,
        cv2.cvtColor(modifier, cv2.COLOR_RGB2BGR),
    )


def data_augmentation(dataset_path: str, augmented_path: str, n_augmentations: int):
    # Ensuring that the directories exist
    create_dirs(dataset_path)
    create_dirs(augmented_path)

    # Iterating through each image
    for img_name in tqdm(os.listdir(dataset_path)):

        img_path = f"{dataset_path}/{img_name}"
        img = plt.imread(img_path)
        img_height, img_width, _ = img.shape

        # Opening the image for shading
        img_for_shading = Image.open(img_path)

        # Data augmentation loop
        for i in range(n_augmentations):

            # Horizontal Translation

            translation_amt = _get_translation_amt(img_width)
            translated = imutils.translate(img, translation_amt, 0)

            _save_img(
                _get_save_path(augmented_path, img_name, "horiTranslate", i), translated
            )

            # Vertical Translation

            translation_amt = _get_translation_amt(img_height)
            translated = imutils.translate(img, 0, translation_amt)

            # Name and save the translated image into the augmented folder
            _save_img(
                _get_save_path(augmented_path, img_name, "vertiTranslate", i),
                translated,
            )

            # Rotation

            # Randomly rotate the image
            rotation = random.randint(0, 360)
            rotated = imutils.rotate(img, angle=rotation)

            _save_img(_get_save_path(augmented_path, img_name, "rotate", i), rotated)

            # Brightness Adjust

            # Randomly discolour the image
            brightness = random.uniform(0.5, 1.5)

            # Image brightness enhancer
            enhancer = ImageEnhance.Brightness(img_for_shading)
            im_output = enhancer.enhance(brightness)

            # Save the rotated image into the augmented folder
            im_output.save(_get_save_path(augmented_path, img_name, "shaded", i))
