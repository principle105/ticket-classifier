from config import DATA_DIR
from preprocessing.image_loader import load_data


def train_model():
    dataset = load_data(DATA_DIR)

    for data, labels in dataset.take(1):
        print(data.shape)
        print(labels.shape)