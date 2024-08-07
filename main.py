import os
from pipelines.training_pipeline import training_pipeline
from catOrDogClassifier.training import Training
from catOrDogClassifier.prediction import Prediction

# training = Training()
# training.train_model()
# directory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
# model_path = os.path.join(directory, 'model.keras')
#
# predictor = Prediction(model_path)
#
# directory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'testing'))
# image_path = os.path.join(directory, '301.jpg')
#
# prediction_result = predictor.predict_image(image_path)
# print(prediction_result)
image_size = 100
epochs = 5
validation_split = 0.1
batch_size = 32

directory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))

training_pipeline(directory, image_size=image_size, epochs=epochs, validation_split=validation_split, batch_size=batch_size)


