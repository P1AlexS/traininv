from zenml import pipeline
from steps.data_ingestion import ingest_data
from steps.model_training import train_model
from steps.model_evaluation import evaluate_model
@pipeline()
def training_pipeline(data_path: str, image_size: int, epochs: int, validation_split: float, batch_size: int):
    x, y = ingest_data(data_path, image_size)
    train_model(x, y, image_size, epochs, validation_split, batch_size)
    evaluate_model()
