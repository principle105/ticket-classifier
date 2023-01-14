import argparse
import logging

from config import (N_AUGMENTATIONS, REAL_FOOD_TICKET_PATH_DATASET,
                    REAL_FOOD_TICKET_PATH_PREPROCESSED)
from preprocessing.data_augmentation import data_augmentation

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(
	prog="U of T Hackathon Fake Meal Ticket Classifier",
	description="Utilities for detecting if a meal ticket is real or fake."
)

parser.add_argument(
	"task",
	choices=["augment_data"]
)

args = parser.parse_args()
run_type = args.run_type

if run_type == "augment_data":
	data_augmentation(REAL_FOOD_TICKET_PATH_DATASET, REAL_FOOD_TICKET_PATH_PREPROCESSED)