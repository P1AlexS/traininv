import logging
import os
import cv2
import random
import numpy as np
from zenml import step

class DataIngestion:
    def __init__(self, data_directory: str, image_size: int):
        self.data_directory = data_directory
        self.categories = ['cat', 'dog']
        self.image_size = image_size

    def load_and_preprocess_images(self):
        logging.info(f"Ingesting the data from {self.data_directory}")
        data = []
        for category in self.categories:
            folder = os.path.join(self.data_directory, category)
            label = self.categories.index(category)
            logging.info(f"Loading images from {folder}")
            for img in os.listdir(folder):
                img_path = os.path.join(folder, img)
                img_arr = cv2.imread(img_path)
                img_arr = cv2.resize(img_arr, (self.image_size, self.image_size))
                data.append([img_arr, label])

        logging.info("Shuffling the data")
        random.shuffle(data)

        logging.info("Splitting features and labels")
        x = []
        y = []

        for features, labels in data:
            x.append(features)
            y.append(labels)

        x = np.array(x)
        y = np.array(y)

        logging.info("Normalizing the image data")
        x = x / 255.0

        logging.info("Data loading and preprocessing completed")
        return x, y

@step
def ingest_data(data_directory: str, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        ingest_data_instance = DataIngestion(data_directory, image_size)
        x, y = ingest_data_instance.load_and_preprocess_images()
        return x, y
    except Exception as e:
        logging.error(f"Data Ingestion Error: {e}")
        raise e

