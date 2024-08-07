import os
from pipelines.training_pipeline import training_pipeline

# predictor = Prediction(model_path)
#
# directory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'testing'))
# image_path = os.path.join(directory, '301.jpg')
#
# prediction_result = predictor.predict_image(image_path)
# print(prediction_result)
image_size = 50
epochs = 1
validation_split = 0.1
batch_size = 32

directory = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))

training_pipeline(directory, image_size=image_size, epochs=epochs, validation_split=validation_split, batch_size=batch_size)


