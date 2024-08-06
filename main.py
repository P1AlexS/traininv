import os
from catOrDogClassifier.training import training
from catOrDogClassifier.prediction import Prediction


directory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
model_path = os.path.join(directory, 'model.h5')

predictor = Prediction(model_path)

directory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'testing'))
image_path = os.path.join(directory, '307.jpg')

prediction_result = predictor.predict_image(image_path)
print(prediction_result)
