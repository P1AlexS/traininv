import os
from catOrDogClassifier.training import Training
from catOrDogClassifier.prediction import Prediction




# training = Training()
# training.train_model()

directory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
model_path = os.path.join(directory, 'model.keras')

predictor = Prediction(model_path)

directory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'testing'))
image_path = os.path.join(directory, '301.jpg')

prediction_result = predictor.predict_image(image_path)
print(prediction_result)
