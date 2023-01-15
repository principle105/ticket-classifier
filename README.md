# U of T Fake Meal Ticket Classifier

# Installation

1. Install the dependencies with `pip3 install -r requirements.txt`.

# Usage

```
python3 -m tensorboard.main --logdir ./logs
python3 src/main.py [OPTIONS]
```

Options:

-   `augment_data` - Increases the size of the data by creating modified version of the data
-   `train_model` - Trains the model
-   `classify_image [FILEPATH]` - Determines if a meal ticket is real or fake.
-   `info` - Gives information about the project
