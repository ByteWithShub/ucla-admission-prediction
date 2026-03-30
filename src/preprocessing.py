#Preprocessing module for UCLA Admission Prediction project
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from src.logger import setup_logger

logger = setup_logger()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()

        # Convert target into binary classification
        df["Admit_Chance"] = (df["Admit_Chance"] >= 0.8).astype(int)

        # Drop unnecessary column
        if "Serial_No" in df.columns:
            df = df.drop("Serial_No", axis=1)

        # Convert selected variables to categorical
        df["University_Rating"] = df["University_Rating"].astype("object")
        df["Research"] = df["Research"].astype("object")

        logger.info("Data cleaned successfully")
        return df
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        raise


def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = pd.get_dummies(
            df,
            columns=["University_Rating", "Research"],
            dtype=int
        )
        logger.info("Categorical encoding completed")
        return df
    except Exception as e:
        logger.error(f"Error encoding data: {e}")
        raise


def split_features_target(df: pd.DataFrame):
    try:
        X = df.drop("Admit_Chance", axis=1)
        y = df["Admit_Chance"]
        logger.info("Features and target split completed")
        return X, y
    except Exception as e:
        logger.error(f"Error splitting features and target: {e}")
        raise


def scale_data(xtrain, xtest):
    try:
        scaler = MinMaxScaler()
        xtrain_scaled = scaler.fit_transform(xtrain)
        xtest_scaled = scaler.transform(xtest)
        logger.info("Feature scaling completed")
        return xtrain_scaled, xtest_scaled, scaler
    except Exception as e:
        logger.error(f"Error scaling data: {e}")
        raise