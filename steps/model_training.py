import logging
import os
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from zenml import step




@step
def train_model(x: np.ndarray, y: np.ndarray, image_size: int, epochs: int, validation_split: float, batch_size: int) -> None:

    pass