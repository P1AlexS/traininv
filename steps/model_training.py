import logging
import os
import numpy as np
import mlflow
import mlflow.keras
import dagshub
from mlflow.models import infer_signature
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from zenml import step

class Training:
    def __init__(self, image_size: int, epochs: int, validation_split: float, batch_size: int):
        self.image_size = image_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.batch_size = batch_size

    def build_and_train_model(self, x: np.ndarray, y: np.ndarray):
      with mlflow.start_run() as run:
          mlflow.log_param("image_size", self.image_size)
          mlflow.log_param("epochs", self.epochs)
          mlflow.log_param("validation_split", self.validation_split)
          mlflow.log_param("batch_size", self.batch_size)

      model = Sequential([
          Input(shape=(self.image_size, self.image_size, 3)),
          Conv2D(64, (3, 3), activation='relu'),
          MaxPooling2D((2, 2)),
          Conv2D(64, (3, 3), activation='relu'),
          MaxPooling2D((2, 2)),
          Flatten(),
          Dense(128, activation='relu'),
          Dense(2, activation='softmax')
      ])
      model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

      logging.info("Starting model training")
      history = model.fit(x, y, epochs=self.epochs, validation_split=self.validation_split, batch_size=self.batch_size)

      for epoch, loss in enumerate(history.history['loss']):
          mlflow.log_metric("loss", loss, step=epoch)

      for epoch, accuracy in enumerate(history.history['accuracy']):
          mlflow.log_metric("accuracy", accuracy, step=epoch)

      directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
      model_path = os.path.join(directory, 'model.keras')
      model.save(model_path)
      logging.info(f"Model saved at {model_path}")

      mlflow.keras.log_model(model, "model")
      mlflow.end_run()

@step
def train_model(x: np.ndarray, y: np.ndarray, image_size: int, epochs: int, validation_split: float, batch_size: int) -> None:
    try:
        training_instance = Training(image_size, epochs, validation_split, batch_size)
        training_instance.build_and_train_model(x, y)
    except Exception as e:
        logging.error(f"Training Error: {e}")
        raise e
