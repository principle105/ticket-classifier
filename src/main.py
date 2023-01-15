import logging

import typer
from pyfiglet import figlet_format
from rich import print

from scripts.augment_data import augment_data
from scripts.train_model import train_model
from scripts.classify_image import classify_image

# Initializing cli
app = typer.Typer()

logging.basicConfig(level=logging.INFO)


@app.command(name="augment_data")
def _augment_data():
    print("[green]Augmenting Data![/green]")
    augment_data()

@app.command(name="train_model")
def _train_model():
    print("[green]Training Model![/green]")
    train_model()

@app.command(name="classify_image")
def _classify_image(filepath: str):
    print("[green]Classifying Image![/green]")
    classify_image(filepath)


@app.command()
def info():
    text = figlet_format("Fake Meal Ticket Classifier")
    print(
        f"[green]{text}[/green]\n\nUtilities for detecting if a meal ticket is real or fake."
    )


if __name__ == "__main__":
    app()
