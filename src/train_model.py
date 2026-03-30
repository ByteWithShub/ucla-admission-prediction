# Model training module for UCLA Admission Prediction project
import os
import joblib
from sklearn.neural_network import MLPClassifier
from src.logger import setup_logger

logger = setup_logger()


def train_mlp_classifier(xtrain, ytrain):
    try:
        model = MLPClassifier(
            hidden_layer_sizes=(3,),
            batch_size=50,
            max_iter=200,
            random_state=123
        )
        model.fit(xtrain, ytrain)
        logger.info("MLPClassifier trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise


def save_model(model, file_path: str):
    try:
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, file_path)
        logger.info(f"Model saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise