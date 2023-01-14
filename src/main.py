import logging

import typer
from pyfiglet import figlet_format

from scripts.augment_data import augment_data

# Initializing cli
app = typer.Typer()

logging.basicConfig(level=logging.INFO)


@app.command(name="augment_data")
def _augment_data():
    typer.echo("Augmenting data...")
    augment_data()


@app.command()
def info():
    typer.echo(figlet_format("Fake Meal Ticket Classifier"))


if __name__ == "__main__":
    app()
