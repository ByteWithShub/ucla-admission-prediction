# Model evaluation module for UCLA Admission Prediction project
import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from src.logger import setup_logger

logger = setup_logger()


def evaluate_model(model, xtest, ytest):
    try:
        ypred = model.predict(xtest)
        accuracy = accuracy_score(ytest, ypred)
        cm = confusion_matrix(ytest, ypred)

        logger.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")
        return accuracy, cm, ypred
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise


def save_loss_curve(model):
    try:
        os.makedirs("outputs", exist_ok=True)
        loss_df = pd.DataFrame({"Loss": model.loss_curve_})
        loss_df.to_csv("outputs/loss_curve.csv", index=False)
        logger.info("Loss curve saved successfully")
    except Exception as e:
        logger.error(f"Error saving loss curve: {e}")
        raise