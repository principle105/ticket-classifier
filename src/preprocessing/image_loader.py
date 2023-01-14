import tensorflow as tf


def load_data(data_dir):
	dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir, image_size=(180, 180), batch_size=64)

	return dataset