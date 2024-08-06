import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class Prediction:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.image_size = 100

    def preprocess_image(self, img_path):
        img_arr = cv2.imread(img_path)
        img_arr = cv2.resize(img_arr, (self.image_size, self.image_size))
        img_arr = np.array(img_arr) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)
        return img_arr

    def predict_image(self, img_path):
        img_arr = self.preprocess_image(img_path)
        prediction = self.model.predict(img_arr)
        predicted_class = np.argmax(prediction, axis=1)[0]
        if predicted_class == 0:
            return "cat"
        elif predicted_class == 1:
            return "dog"
        else:
            return "Unknown class"