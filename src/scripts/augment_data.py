from config import (N_AUGMENTATIONS, REAL_FOOD_TICKET_PATH_DATASET,
                    REAL_FOOD_TICKET_PATH_PREPROCESSED)
from preprocessing.data_augmentation import data_augmentation


def augment_data():
	data_augmentation(REAL_FOOD_TICKET_PATH_DATASET, REAL_FOOD_TICKET_PATH_PREPROCESSED, N_AUGMENTATIONS)