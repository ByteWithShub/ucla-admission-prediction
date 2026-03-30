# Model prediction module for UCLA Admission Prediction project
import pandas as pd
from src.logger import setup_logger

logger = setup_logger()


def prepare_input_data(input_data: dict) -> pd.DataFrame:
    try:
        input_df = pd.DataFrame([input_data])
        logger.info("Input data prepared successfully")
        return input_df
    except Exception as e:
        logger.error(f"Error preparing input data: {e}")
        raise