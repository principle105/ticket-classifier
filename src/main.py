import argparse
import logging

from scripts.augment_data import augment_data

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
run_type = args.task

if run_type == "augment_data":
	augment_data()